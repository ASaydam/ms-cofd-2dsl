"""
Status of the production: tasks per run and state, the queue, the footprint, failures and the compute used so far.

    python3 hpc/status.py                 summary
    python3 hpc/status.py --failures 20   with the last 20 failures and their reasons
    python3 hpc/status.py --jobs 20       with the 20 running jobs closest to the end of their solver run
    python3 hpc/status.py --json          the same numbers as JSON (hpc/publish_status.py, the dashboard)

Runs with the Python of the cluster (standard library and PyYAML), outside of the container. The compute used and
the throughput are read from results/<run>/summary.csv, one line per finished task (written by hpc/run_task.py).
The progress of every running job comes from the end of its solver log (job_progress).
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ledger  # noqa: E402
import topup  # noqa: E402

LABELS = {'0.25x': '2 mm', 'base_fo': '1 mm, first order', 'base_lu': '1 mm', '4x': '0.5 mm', '16x': '0.25 mm'}

# simpleFoam log and controlDict of a case folder
RE_TIME = re.compile(r"^Time = ([\d.eE+-]+)\s*$", re.MULTILINE)
RE_RESIDUAL = re.compile(r"Solving for (Ux|Uy|p), Initial residual = ([\d.eE+-]+)")
RE_EXECUTION = re.compile(r"^ExecutionTime = ([\d.eE+-]+) s", re.MULTILINE)
RE_END = re.compile(r"^End\s*$", re.MULTILINE)
RE_END_TIME = re.compile(r"^endTime\s+([\d.eE+-]+)\s*;", re.MULTILINE)
TARGET_DECADES = 6   # residualControl 1e-6 for U and p (system/fvSolution of the templates)
# pace of a running job (dashboard): the residual is judged after the first WARMUP part of the iteration cap (a run
# started from mapped fields first rises for ~1000 iterations); slower than FLAT decades per 1000 iterations is flat
# (a decade in more than 20 000 iterations); a log not written for FROZEN_S seconds belongs to a stuck job
WARMUP = 0.2
FLAT = -0.05
FROZEN_S = 1200


def parse_time(text):
    """ISO time as written by run_task.py (+0300 or +03:00), None when it cannot be read"""
    try:
        if len(text) > 5 and text[-5] in '+-' and text[-3] != ':':
            text = text[:-2] + ':' + text[-2:]
        return datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def fit_line(x, y):
    """least squares (intercept, slope) of y over x"""
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    sxx = sum((a - mean_x) ** 2 for a in x)
    slope = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / sxx if sxx else 0.0
    return mean_y - slope * mean_x, slope


def hours_left(rows, blockage, remaining, running, model, target):
    """
    estimated solver hours of the runs of a level that are not done: per case intercept + per_blockage * blockage
    ratio (at least min), with the pilot fit (status.hours_model in config/slurm.yaml), or with a fit to the finished
    runs once half of the level is done (the first runs to finish are the fast ones); a running task counts with what
    is left after its elapsed time (at least a tenth). Without a model: the mean of the finished runs.
    :param rows: rows of results/<run>/summary.csv (finished runs)
    :param blockage: {case: blockage ratio} of the manifest
    :param remaining: cases of the dataset whose run is not done
    :param running: {case: elapsed seconds} of the running jobs of the level
    :param model: {intercept, per_blockage, min} or None
    :param target: number of cases of the level
    :return: hours (None without any estimate), source of the estimate
    """
    finished = {row['case']: float(row['execution_time_s'] or 0) / 3600 for row in rows if row['case'] in blockage}
    if len(finished) >= 30 and len(finished) >= target / 2:
        intercept, slope = fit_line([blockage[case] for case in finished], list(finished.values()))
        low, source = min(finished.values()), 'fit to the finished runs'
    elif model:
        intercept, slope, low, source = model['intercept'], model['per_blockage'], model['min'], 'pilot fit'
    elif finished:
        return round(len(remaining) * sum(finished.values()) / len(finished), 1), 'mean of the finished runs'
    else:
        return None, None
    total = 0.0
    for case in remaining:
        hours = max(intercept + slope * blockage.get(case, 0.0), low)
        if case in running:
            hours = max(hours - running[case] / 3600, hours / 10)
        total += hours
    return round(total, 1), source


def seconds(text):
    """a squeue time ([days-]hours:minutes:seconds, minutes:seconds) in seconds, None for UNLIMITED and the like"""
    try:
        days, _, clock = text.strip().rpartition('-')
        parts = [int(part) for part in clock.split(':')]
        while len(parts) < 3:
            parts.insert(0, 0)
        return (int(days) if days else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    except ValueError:
        return None


def running_jobs(prefix):
    """our running jobs (squeue): [{job, run, case, node, elapsed_s, limit_s}]"""
    try:
        result = subprocess.run(['squeue', '-u', os.environ.get('USER', ''), '-h', '-t', 'RUNNING',
                                 '-o', '%i|%j|%M|%l|%N'], capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    jobs = []
    for line in result.stdout.splitlines():
        job, name, elapsed, limit, node = line.split('|')
        if name.startswith(prefix + '_'):
            run, _, case = name[len(prefix) + 1:].partition('__')
            jobs.append({'job': job, 'run': run, 'case': case, 'node': node, 'elapsed_s': seconds(elapsed),
                         'limit_s': seconds(limit)})
    return jobs


def job_progress(folder, mapped, now=None, tail=1 << 16):
    """
    where the task of a running job is, from its case folder:

        stage               starting, meshing, mapping (initial fields from base_lu), solving or finishing (arrays,
                            attributes, archive, upload)
        iteration, cap      last iteration of the solver and its endTime
        residuals           initial residuals of Ux, Uy and p of the last iteration; residual: the largest of them
        s_per_iteration     solver seconds per iteration
        decades_per_1000    trend of log10(residual) per 1000 iterations (least squares over the last ~100 iterations)
        eta_iterations/_s   iterations and solver seconds until the residual reaches 1e-6 at that trend
        log_age_s           seconds since the log was last written (a stuck job stops writing)

    Only the last `tail` bytes of the log are read (~110 iterations of a 0.25 mm run).
    :param mapped: True for runs that start from the fields of another run (init_from)
    """
    folder = Path(folder)
    now = (now or datetime.now().astimezone()).timestamp()
    log = folder / 'log.simpleFoam'
    if not (folder / 'mesh_variant').exists():
        return {'stage': 'meshing' if (folder / 'log.gmsh').exists() else 'starting'}
    if not log.exists():
        return {'stage': 'mapping' if mapped and not (folder / 'mapped').exists() else 'starting'}

    size = log.stat().st_size
    with open(log, 'rb') as file:
        file.seek(max(0, size - tail))
        text = file.read().decode(errors='replace')
    progress = {'stage': 'solving', 'log_age_s': max(round(now - log.stat().st_mtime), 0)}  # clocks differ a little
    marks = list(RE_TIME.finditer(text))
    ended = [match.start() for match in RE_END.finditer(text)]
    if ended and (not marks or ended[-1] > marks[-1].start()):
        progress['stage'] = 'finishing'

    samples = []  # (iteration, initial residuals, execution time) of every complete iteration
    for mark, following in zip(marks, marks[1:] + [None]):
        block = text[mark.end():following.start() if following else len(text)]
        residuals = {}
        for field, value in RE_RESIDUAL.findall(block):
            residuals.setdefault(field, float(value))  # the first p solution of the iteration (residualControl)
        execution = RE_EXECUTION.search(block)
        if len(residuals) == 3 and execution:
            samples.append((float(mark.group(1)), residuals, float(execution.group(1))))
    if not samples:
        return progress

    iteration, residuals, _ = samples[-1]
    progress.update({'iteration': int(iteration), 'residuals': [residuals[f] for f in ('Ux', 'Uy', 'p')],
                     'residual': max(residuals.values())})
    control = folder / 'system' / 'controlDict'
    cap = RE_END_TIME.search(control.read_text(errors='replace')) if control.exists() else None
    progress['cap'] = int(float(cap.group(1))) if cap else None

    # speed and trend within the last solver run (the execution time starts again when a run is continued)
    start = max([i for i in range(1, len(samples)) if samples[i][2] < samples[i - 1][2]], default=0)
    recent = samples[start:]
    if len(recent) >= 10 and recent[-1][0] > recent[0][0]:
        progress['s_per_iteration'] = round((recent[-1][2] - recent[0][2]) / (recent[-1][0] - recent[0][0]), 3)
        _, slope = fit_line([s[0] for s in recent], [math.log10(max(max(s[1].values()), 1e-30)) for s in recent])
        progress['decades_per_1000'] = round(1000 * slope, 4)
        if slope < 0:
            left = max((math.log10(progress['residual']) + TARGET_DECADES) / -slope, 0)
            progress['eta_iterations'] = round(left)
            progress['eta_s'] = round(left * progress['s_per_iteration'])
    return progress


def pace(job):
    """
    judgement of a running job (job_progress): its stage when it is not solving, else frozen (log not written for
    FROZEN_S), warming up or flat (residual trend slower than FLAT, before or after the warm-up), cap (the trend
    reaches 1e-6 only after the iteration cap) or ok
    """
    if job['stage'] != 'solving':
        return job['stage']
    if job.get('log_age_s', 0) > FROZEN_S:
        return 'frozen'
    if job.get('decades_per_1000') is None:
        return 'ok'
    early = bool(job.get('cap')) and job['iteration'] < WARMUP * job['cap']
    if job['decades_per_1000'] > FLAT:
        return 'warming up' if early else 'flat'
    if not early and job.get('cap') and job['iteration'] + job.get('eta_iterations', 0) > job['cap']:
        return 'cap'
    return 'ok'


def collect(root, slurm, dataset, jobs=None, nodes=None, used=None, backlog=None, now=None, running=None):
    """all numbers of the status as a dictionary (jobs, nodes, used, backlog, running: given in tests, else asked from
    SLURM and BeeGFS)"""
    now = now or datetime.now().astimezone()
    runs = [run for run in dataset['runs'] if run in slurm.get('only_runs', dataset['runs'])]
    states = ledger.read_states(runs, root)
    if jobs is None:
        jobs, _ = topup.our_jobs(slurm['job_name_prefix'])
    active, unsteady, reserves = topup.dataset_cases(root, dataset, states, topup.cases_file_of(root))
    members = set(active) - set(unsteady)      # the cases of the dataset
    cases = len(members)

    status = {'generated': now.isoformat(timespec='seconds'), 'cases': cases, 'runs': [], 'unsteady': unsteady,
              'reserves': len(reserves),
              'window': (topup.current_window(slurm, now) or {}).get('qos', 'none'),
              'queue': {'running': sum(s == 'RUNNING' for _, s, _ in jobs.values()),
                        'pending': sum(s == 'PENDING' for _, s, _ in jobs.values())}}
    recent = []  # (finished, run, summary row) of the last finished tasks of every run
    summaries = {}
    held = ledger.held_tasks(root)
    for run in runs:
        counts = Counter(state for (r, c), (state, _) in states.items() if r == run and c in members)
        summary = root / 'results' / run / 'summary.csv'
        rows = summaries[run] = list(csv.DictReader(open(summary))) if summary.exists() else []
        hours = sum(float(row['execution_time_s'] or 0) for row in rows) / 3600
        finished = [t for t in (parse_time(row.get('finished', '')) for row in rows) if t]
        latest = {row['case']: row for row in rows}
        capped = {row['case'] for row in rows if row['status'] == 'end_time'}
        for row in rows[-15:]:
            if parse_time(row.get('finished', '')):
                recent.append((parse_time(row['finished']), run, row))
        queued = Counter(s for (r, _), (_, s, _) in jobs.items() if r == run)
        status['runs'].append({
            'run': run, 'label': LABELS.get(run, run),
            'target': cases,
            'done': counts['done'], 'running': queued['RUNNING'], 'pending': queued['PENDING'],
            'submitted': counts['submitted'], 'failed': counts['failed'],
            'held': sum(1 for r, case in held if r == run and case in members),
            # not converged: cases of the dataset whose last run stopped at its cap (a task run again with a higher cap
            # or other settings counts with its last run; dropped cases are in unsteady); the compute counts every run
            'end_time': sum(1 for case, row in latest.items() if case in members and row['status'] == 'end_time'),
            # converged when run again after stopping at the cap
            'rerun_converged': sum(1 for case, row in latest.items() if case in members and case in capped
                                   and row['status'] == 'converged'),
            'core_hours': round(hours, 2), 'mean_hours': round(hours / len(rows), 4) if rows else None,
            **{f'done_{h}h': sum(t > now - timedelta(hours=h) for t in finished) for h in (1, 6, 24)}})
    status['core_hours'] = round(sum(r['core_hours'] for r in status['runs']), 1)

    limits = slurm['footprint']
    status['quota_gib'] = round(topup.used_gib(root) if used is None else used, 1)
    status['backlog_gib'] = round(topup.backlog_gib(root) if backlog is None else backlog, 2)
    status['pause_gib'] = limits['pause_used_gib']
    running = running_jobs(slurm['job_name_prefix']) if running is None else running
    status['nodes'] = dict(Counter(job['node'] for job in running)) if nodes is None else nodes

    # the running jobs of this data root with their progress (a pilot with the same job names is left out: its
    # tasks are not running in this ledger)
    status['jobs'] = []
    for job in running:
        state, content = states.get((job['run'], job['case']), (None, ''))
        if not (state == 'submitted' or (state == 'running' and content.split()[:1] == [job['job']])):
            continue
        folder = root / dataset['paths']['cases'] / job['run'] / job['case']
        mapped = bool(dataset['runs'].get(job['run'], {}).get('init_from'))
        progress = {**job, **job_progress(folder, mapped, now)}
        status['jobs'].append({**progress, 'pace': pace(progress)})

    # compute left per level, from the blockage ratio of every case not done yet
    blockage = {row['case_name']: float(row['blockage_ratio'])
                for row in ledger.read_manifest(root / 'geometry' / 'manifest.csv') if row.get('blockage_ratio')}
    models = (slurm.get('status') or {}).get('hours_model') or {}
    for entry in status['runs']:
        run = entry['run']
        remaining = [case for case in members if states.get((run, case), (None, ''))[0] != 'done']
        running_cases = {job['case']: job['elapsed_s'] or 0 for job in status['jobs'] if job['run'] == run}
        entry['hours_left'], entry['hours_source'] = hours_left(summaries[run], blockage, remaining, running_cases,
                                                                models.get(run), entry['target'])
    status['finished'] = [{'run': run, 'case': row['case'], 'status': row['status'],
                           'iterations': int(float(row['iterations'])) if row.get('iterations') else None,
                           'hours': round(float(row['execution_time_s'] or 0) / 3600, 3),
                           'finished': t.isoformat(timespec='seconds')}
                          for t, run, row in sorted(recent, key=lambda item: item[0], reverse=True)[:15]]
    failed = [(run, case, content) for (run, case), (state, content) in states.items() if state == 'failed']
    status['failures'] = [{'run': run, 'case': case, 'reason': content[:200]} for run, case, content in failed[-20:]]
    status['interrupted'] = sum(1 for (run, case), (state, _) in states.items()
                                if state in ('submitted', 'running') and (run, case) not in jobs
                                and (run, case) not in held)
    return status


def main():
    parser = argparse.ArgumentParser(description="Status of the production")
    parser.add_argument("--failures", type=int, default=5, help="number of failures to list")
    parser.add_argument("--jobs", type=int, default=0, help="number of running jobs to list")
    parser.add_argument("--json", action="store_true", help="print the status as JSON")
    args = parser.parse_args()

    root = ledger.data_root()
    slurm, dataset = topup.load_settings(root)
    status = collect(root, slurm, dataset)
    if args.json:
        print(json.dumps(status, indent=1))
        return

    print(f"{status['generated'][:16].replace('T', ' ')}  data root {root}  ({status['cases']} cases)")
    print(f"qos window: {status['window']}  queue: {status['queue']['running']} running, "
          f"{status['queue']['pending']} pending")
    print(f"\n{'run':8s} {'done':>6s} {'target':>7s} {'running':>8s} {'submitted':>10s} {'failed':>7s} "
          f"{'core-h':>8s} {'mean h':>7s} {'end_time':>9s} {'done 1h':>8s} {'done 24h':>9s} {'left h':>8s}")
    for r in status['runs']:
        print(f"{r['run']:8s} {r['done']:6d} {r['target']:7d} {r['running']:8d} {r['submitted']:10d} {r['failed']:7d} "
              f"{r['core_hours']:8.1f} {r['mean_hours'] or 0:7.2f} {r['end_time']:9d} {r['done_1h']:8d} "
              f"{r['done_24h']:9d} {r['hours_left'] if r['hours_left'] is not None else '-':>8}")
    left = sum(r['hours_left'] or 0 for r in status['runs'])
    print(f"solver core-hours so far: {status['core_hours']:.0f}, left (estimate from the blockage ratio): {left:.0f}")

    flag = '  WARNING' if status['quota_gib'] > status['pause_gib'] else ''
    flag = '  ALARM: close to the quota' if status['quota_gib'] > 800 else flag
    print(f"\nquota used: {status['quota_gib']:.0f} GiB (pause at {status['pause_gib']}), not uploaded: "
          f"{status['backlog_gib']:.1f} GiB (pause at {slurm['footprint']['pause_backlog_gib']}){flag}")
    print("nodes: " + ', '.join(f"{node} {n}" for node, n in sorted(status['nodes'].items())))
    jobs = status['jobs']
    print(f"running jobs: {', '.join(f'{n} {p}' for p, n in Counter(j['pace'] for j in jobs).most_common())} "
          f"(flat: residual not falling after the warm-up; cap: falling too slowly to reach 1e-6 before the cap)")
    if args.jobs:
        print(f"\n{'run':8s} {'case':28s} {'iteration':>15s} {'residual':>9s} {'s/it':>6s} {'dec/1000':>8s} "
              f"{'eta h':>6s} {'log age':>8s}")
        for j in sorted(jobs, key=lambda j: j.get('eta_s', math.inf))[:args.jobs]:
            print(f"{j['run']:8s} {j['case']:28s} {j.get('iteration', '-'):>7} / {j.get('cap') or '-':<6} "
                  f"{j.get('residual', math.nan):9.2e} {j.get('s_per_iteration', math.nan):6.2f} "
                  f"{j.get('decades_per_1000', math.nan):8.3f} {j.get('eta_s', math.nan) / 3600:6.2f} "
                  f"{j.get('log_age_s', math.nan):8.0f}  {j['pace']}")
    seeds = root / 'seeds'
    if seeds.exists():
        print(f"base_lu solutions kept for the fine runs: {sum(1 for _ in seeds.iterdir())}")

    unsteady = status['unsteady']
    if unsteady:
        print(f"\n{len(unsteady)} cases without a steady solution (base_lu at its iteration cap; dropped with all runs, "
              f"{status['reserves']} reserve cases in use): "
              f"{', '.join(unsteady[:10])}{' ...' if len(unsteady) > 10 else ''}")
    if status['failures'] and args.failures:
        print(f"\nfailures (last {len(status['failures'])}):")
        for f in status['failures'][-args.failures:]:
            print(f"  {f['run']:8s} {f['case']:28s} {f['reason'][:150]}")
    if status['interrupted']:
        print(f"\n{status['interrupted']} tasks without a job (interrupted), topup.py submits them again")
    again = [f"{r['rerun_converged']} {r['run']}" for r in status['runs'] if r['rerun_converged']]
    if again:
        print(f"\nconverged when run again after stopping at the cap: {', '.join(again)}")
    held = [f"{r['held']} {r['run']}" for r in status['runs'] if r['held']]
    if held:
        print(f"\non hold (hold.txt, not submitted): {', '.join(held)}")


if __name__ == "__main__":
    main()
