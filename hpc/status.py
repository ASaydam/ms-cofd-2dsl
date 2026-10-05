"""
Status of the production: tasks per run and state, the queue, the footprint, failures and the compute used so far.

    python3 hpc/status.py                 summary
    python3 hpc/status.py --failures 20   with the last 20 failures and their reasons
    python3 hpc/status.py --json          the same numbers as JSON (hpc/publish_status.py, the dashboard)

Runs with the Python of the cluster (standard library and PyYAML), outside of the container. The compute used and
the throughput are read from results/<run>/summary.csv, one line per finished task (written by hpc/run_task.py).
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ledger  # noqa: E402
import topup  # noqa: E402

LABELS = {'0.25x': '2 mm', 'base_fo': '1 mm, first order', 'base_lu': '1 mm', '4x': '0.5 mm', '16x': '0.25 mm'}


def parse_time(text):
    """ISO time as written by run_task.py (+0300 or +03:00), None when it cannot be read"""
    try:
        if len(text) > 5 and text[-5] in '+-' and text[-3] != ':':
            text = text[:-2] + ':' + text[-2:]
        return datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def running_nodes(prefix):
    """{node: number of our running jobs} (squeue)"""
    try:
        result = subprocess.run(['squeue', '-u', os.environ.get('USER', ''), '-h', '-t', 'RUNNING', '-o', '%N|%j'],
                                capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    return dict(Counter(line.split('|')[0] for line in result.stdout.splitlines()
                        if line.split('|')[-1].startswith(prefix + '_')))


def collect(root, slurm, dataset, jobs=None, nodes=None, used=None, backlog=None, now=None):
    """all numbers of the status as a dictionary (jobs, nodes, used, backlog: given in tests, else asked from SLURM
    and BeeGFS)"""
    now = now or datetime.now().astimezone()
    runs = [run for run in dataset['runs'] if run in slurm.get('only_runs', dataset['runs'])]
    manifest_path = root / 'geometry' / 'manifest.csv'
    cases = len(ledger.read_manifest(manifest_path)) if manifest_path.exists() else 0
    states = ledger.read_states(runs, root)
    if jobs is None:
        jobs, _ = topup.our_jobs(slurm['job_name_prefix'])
    dependencies = ledger.dependencies_of(dataset['runs'])
    unsteady = ledger.unsteady_cases(states, dependencies)

    status = {'generated': now.isoformat(timespec='seconds'), 'cases': cases, 'runs': [], 'unsteady': unsteady,
              'window': (topup.current_window(slurm, now) or {}).get('qos', 'none'),
              'queue': {'running': sum(s == 'RUNNING' for _, s, _ in jobs.values()),
                        'pending': sum(s == 'PENDING' for _, s, _ in jobs.values())}}
    for run in runs:
        counts = Counter(state for (r, _), (state, _) in states.items() if r == run)
        summary = root / 'results' / run / 'summary.csv'
        rows = list(csv.DictReader(open(summary))) if summary.exists() else []
        hours = sum(float(row['execution_time_s'] or 0) for row in rows) / 3600
        finished = [t for t in (parse_time(row.get('finished', '')) for row in rows) if t]
        queued = Counter(s for (r, _), (_, s, _) in jobs.items() if r == run)
        status['runs'].append({
            'run': run, 'label': LABELS.get(run, run),
            'target': cases - len(unsteady) if run in dependencies else cases,
            'done': counts['done'], 'running': queued['RUNNING'], 'pending': queued['PENDING'],
            'submitted': counts['submitted'], 'failed': counts['failed'],
            'end_time': sum(row['status'] == 'end_time' for row in rows),
            'core_hours': round(hours, 2), 'mean_hours': round(hours / len(rows), 4) if rows else None,
            **{f'done_{h}h': sum(t > now - timedelta(hours=h) for t in finished) for h in (1, 6, 24)}})
    status['core_hours'] = round(sum(r['core_hours'] for r in status['runs']), 1)

    limits = slurm['footprint']
    status['quota_gib'] = round(topup.used_gib(root) if used is None else used, 1)
    status['backlog_gib'] = round(topup.backlog_gib(root) if backlog is None else backlog, 2)
    status['pause_gib'] = limits['pause_used_gib']
    status['nodes'] = running_nodes(slurm['job_name_prefix']) if nodes is None else nodes
    failed = [(run, case, content) for (run, case), (state, content) in states.items() if state == 'failed']
    status['failures'] = [{'run': run, 'case': case, 'reason': content[:200]} for run, case, content in failed[-20:]]
    status['interrupted'] = sum(1 for (run, case), (state, _) in states.items()
                                if state in ('submitted', 'running') and (run, case) not in jobs)
    return status


def main():
    parser = argparse.ArgumentParser(description="Status of the production")
    parser.add_argument("--failures", type=int, default=5, help="number of failures to list")
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
          f"{'core-h':>8s} {'mean h':>7s} {'end_time':>9s} {'done 1h':>8s} {'done 24h':>9s}")
    for r in status['runs']:
        print(f"{r['run']:8s} {r['done']:6d} {r['target']:7d} {r['running']:8d} {r['submitted']:10d} {r['failed']:7d} "
              f"{r['core_hours']:8.1f} {r['mean_hours'] or 0:7.2f} {r['end_time']:9d} {r['done_1h']:8d} "
              f"{r['done_24h']:9d}")
    print(f"solver core-hours so far: {status['core_hours']:.0f}")

    flag = '  WARNING' if status['quota_gib'] > status['pause_gib'] else ''
    flag = '  ALARM: close to the quota' if status['quota_gib'] > 800 else flag
    print(f"\nquota used: {status['quota_gib']:.0f} GiB (pause at {status['pause_gib']}), not uploaded: "
          f"{status['backlog_gib']:.1f} GiB (pause at {slurm['footprint']['pause_backlog_gib']}){flag}")
    print("nodes: " + ', '.join(f"{node} {n}" for node, n in sorted(status['nodes'].items())))
    seeds = root / 'seeds'
    if seeds.exists():
        print(f"base_lu solutions kept for the fine runs: {sum(1 for _ in seeds.iterdir())}")

    unsteady = status['unsteady']
    if unsteady:
        print(f"\n{len(unsteady)} cases without a steady solution (base_lu at its iteration cap, finer meshes not run): "
              f"{', '.join(unsteady[:10])}{' ...' if len(unsteady) > 10 else ''}")
    if status['failures'] and args.failures:
        print(f"\nfailures (last {len(status['failures'])}):")
        for f in status['failures'][-args.failures:]:
            print(f"  {f['run']:8s} {f['case']:28s} {f['reason'][:150]}")
    if status['interrupted']:
        print(f"\n{status['interrupted']} tasks without a job (interrupted), topup.py submits them again")


if __name__ == "__main__":
    main()
