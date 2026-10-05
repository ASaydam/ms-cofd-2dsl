"""
Keeps the production queue filled, within the limits of config/slurm.yaml. It is not a daemon: every job calls it
when it ends (hpc/job_task.sh), and it is called during monitoring. Each call does one squeue and returns.

    python3 hpc/topup.py                 submit what is ready, up to the limits
    python3 hpc/topup.py --dry-run       only print what would be submitted
    python3 hpc/topup.py --cases FILE    only the cases listed in FILE (one name per line)

A data root other than the production (e.g. a pilot, MSCOFD_DATA_ROOT) can hold two files of its own:
    cases.txt     the cases of this root, used when --cases is not given (also by the calls of the jobs)
    slurm.yaml    entries that replace the top-level entries of config/slurm.yaml (e.g. footprint, job_name_prefix),
                  and only_runs: the runs of this root (default: all runs of config/dataset.yaml)

What it does:
1. one call at a time (a lock that holds across nodes, see ledger.DirectoryLock) and at most one every
   min_interval_s seconds: a call waits for its turn, and
   returns at once when another call started after it was made (that call saw the same states), so the state
   change of a job that just ended is always followed by a full call
2. the QoS window of the current date (temporary allocation, then the normal limits); pending jobs of an earlier
   window are moved to the current QoS
3. footprint guard: no submissions while the quota usage or the upload backlog is above the limits
4. tasks whose job is gone without a done / failed state (preempted, time limit, node failure) and failed tasks
   are submitted again, up to max_attempts submissions; they continue where they stopped (hpc/run_task.py)
5. ready tasks (hpc/ledger.py) are submitted in their order until running + pending jobs reach max_queued

Runs with the Python of the cluster (standard library and PyYAML), outside of the container.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ledger  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def load_settings(root=None):
    """config/slurm.yaml and config/dataset.yaml, with the top-level entries of <root>/slurm.yaml and
    <root>/dataset.yaml in place of their own"""
    settings = []
    for name in ('slurm.yaml', 'dataset.yaml'):
        with open(REPO / 'config' / name) as file:
            values = yaml.safe_load(file)
        if root is not None and (Path(root) / name).exists():
            with open(Path(root) / name) as file:
                values.update(yaml.safe_load(file) or {})
        settings.append(values)
    return tuple(settings)


def cases_file_of(root, path=None):
    """the file of the cases to run: --cases, else <root>/cases.txt when it exists, else None (all cases)"""
    if path:
        return Path(path)
    return Path(root) / 'cases.txt' if (Path(root) / 'cases.txt').exists() else None


def last_call(stamp):
    """start time of the last call (seconds since the epoch, written by the call), 0 when there was none"""
    try:
        return float(stamp.read_text())
    except (FileNotFoundError, ValueError):
        return 0.0


def current_window(slurm, now=None):
    now = now or datetime.now().astimezone()
    for window in slurm['windows']:
        if datetime.fromisoformat(window['start']) <= now < datetime.fromisoformat(window['end']):
            return window
    return None


def our_jobs(prefix):
    """
    the jobs of the production in the queue
    :return: {(run, case): (job id, state, qos)}, number of jobs (a task with two jobs counts twice)
    """
    result = subprocess.run(['squeue', '-u', os.environ.get('USER', ''), '-h', '-o', '%i|%j|%T|%q'],
                            capture_output=True, text=True, check=True)
    jobs, count = {}, 0
    for line in result.stdout.splitlines():
        job_id, name, state, qos = line.split('|')
        if not name.startswith(prefix + '_'):
            continue
        run, _, case = name[len(prefix) + 1:].partition('__')
        jobs[(run, case)] = (job_id, state, qos)
        count += 1
    return jobs, count


def select_tasks(states, jobs, cases, runs, attempts_of, max_attempts, dependencies=None):
    """
    the tasks to submit: failed tasks and interrupted ones (their job is gone without a done / failed state) with
    submissions left, then the ready tasks (ledger.ready_tasks); a task is in only one of the lists
    :param attempts_of: function (run, case) -> number of submissions so far
    :param dependencies: {run: run it starts from} (ledger.dependencies_of)
    :return: retry, ready, exhausted (interrupted tasks without submissions left: (run, case, marker content))
    """
    selected = set(cases)
    retry, exhausted = [], []
    for (run, case), (state, content) in states.items():
        if case not in selected:
            continue
        interrupted = state in ('submitted', 'running') and (run, case) not in jobs
        if interrupted or state == 'failed':
            if attempts_of(run, case) < max_attempts:
                retry.append((run, case))
            elif interrupted:
                exhausted.append((run, case, content))
    return retry, ledger.ready_tasks(cases, runs, states, dependencies), exhausted


def used_gib(root):
    """quota usage of the user in GiB (beegfs-ctl), the last known value when beegfs-ctl is not available"""
    cache = root / 'state' / 'quota_used_gib'
    try:
        result = subprocess.run(['beegfs-ctl', '--getquota', '--uid', os.environ.get('USER', ''), '--csv'],
                                capture_output=True, text=True, timeout=60)
        for line in result.stdout.splitlines():
            fields = line.split(',')
            if fields and fields[0] == os.environ.get('USER'):
                value = int(fields[2]) / 2 ** 30
                cache.write_text(f"{value:.1f}\n")
                return value
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return float(cache.read_text()) if cache.exists() else 0.0


def backlog_gib(root):
    """archives waiting for the upload, in GiB"""
    return sum(path.stat().st_size for path in (root / 'archives').glob('*/*.tar.zst*')) / 2 ** 30


def attempts_of(root, run, case):
    path = root / 'state' / run / f"{case}.attempts"
    return int(path.read_text()) if path.exists() else 0


def excluded_nodes(root, slurm):
    """nodes the jobs must not use: exclude_nodes of config/slurm.yaml and the nodes jobs found the container
    cannot start on (<root>/state/bad_nodes, one node per line, written by hpc/job_task.sh)"""
    nodes = set(slurm.get('exclude_nodes') or [])
    bad = Path(root) / 'state' / 'bad_nodes'
    if bad.exists():
        nodes.update(line.split()[0] for line in bad.read_text().splitlines() if line.strip())
    return sorted(nodes)


def submit(root, slurm, window, run, case, dry_run):
    settings = slurm['runs'][run]
    logs = root / 'logs' / run
    logs.mkdir(parents=True, exist_ok=True)
    exclude = excluded_nodes(root, slurm)
    command = ['sbatch', '--parsable', f"--job-name={slurm['job_name_prefix']}_{run}__{case}",
               f"--account={slurm['account']}", f"--partition={slurm['partition']}", f"--qos={window['qos']}",
               '--ntasks=1', '--cpus-per-task=1', f"--mem={settings['mem']}", f"--time={settings['time']}",
               '--requeue', f"--output={logs / (case + '-%j.out')}", '--export=ALL'] + \
              ([f"--exclude={','.join(exclude)}"] if exclude else []) + \
              [str(REPO / 'hpc' / 'job_task.sh'), run, case]
    if dry_run:
        print(' '.join(command))
        return
    job_id = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip().split(';')[0]
    attempts = attempts_of(root, run, case) + 1
    ledger.state_folder(run, root).mkdir(parents=True, exist_ok=True)
    (root / 'state' / run / f"{case}.attempts").write_text(f"{attempts}\n")
    ledger.set_state(run, case, 'submitted', f"{job_id} attempt={attempts}", root=root)


def main():
    parser = argparse.ArgumentParser(description="Submit the ready tasks of the production")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="do not wait for min_interval_s")
    parser.add_argument("--cases", help="file with the case names to run (default: all cases of the manifest)")
    args = parser.parse_args()

    root = ledger.data_root()
    slurm, dataset = load_settings(root)
    (root / 'state').mkdir(parents=True, exist_ok=True)

    # 1. one call at a time, not more often than min_interval_s
    arrival = time.time()
    with ledger.DirectoryLock(root / 'state' / 'topup.lockdir'):
        stamp = root / 'state' / 'topup.last'
        if not args.force and not args.dry_run:
            last = last_call(stamp)
            if last > arrival:
                return
            time.sleep(max(0.0, slurm['min_interval_s'] - (time.time() - last)))
        if not args.dry_run:
            stamp.write_text(f"{time.time():.3f}\n")
        top_up(root, slurm, dataset, args)


def top_up(root, slurm, dataset, args):
    """steps 2 to 5, in the lock"""
    # 2. QoS of the current date
    window = current_window(slurm)
    if window is None:
        print("no QoS window for the current date, nothing submitted")
        return
    jobs, queued = our_jobs(slurm['job_name_prefix'])
    for (run, case), (job_id, state, qos) in jobs.items():
        if state == 'PENDING' and qos != window['qos'] and not args.dry_run:
            subprocess.run(['scontrol', 'update', f"jobid={job_id}", f"qos={window['qos']}"], check=False)

    # 3. footprint guard
    used, backlog = used_gib(root), backlog_gib(root)
    footprint = slurm['footprint']
    if used > footprint['pause_used_gib'] or backlog > footprint['pause_backlog_gib']:
        print(f"paused: {used:.0f} GiB used (limit {footprint['pause_used_gib']}), "
              f"{backlog:.0f} GiB not uploaded (limit {footprint['pause_backlog_gib']})")
        return

    # 4. interrupted and failed tasks
    runs = [run for run in dataset['runs'] if run in slurm.get('only_runs', dataset['runs'])]
    manifest = ledger.read_manifest(root / 'geometry' / 'manifest.csv')
    cases = ledger.case_order(manifest)
    cases_file = cases_file_of(root, args.cases)
    if cases_file:
        selected = set(Path(cases_file).read_text().split())
        cases = [case for case in cases if case in selected]
    states = ledger.read_states(runs, root)
    retry, ready, exhausted = select_tasks(states, jobs, cases, runs, lambda run, case: attempts_of(root, run, case),
                                           slurm['max_attempts'],
                                           ledger.dependencies_of(dataset['runs']))
    for run, case, content in exhausted:
        if not args.dry_run:
            ledger.set_state(run, case, 'failed', f"interrupted, {slurm['max_attempts']} submissions ({content})",
                             root=root)

    # 5. submit until max_queued
    free = window['max_queued'] - queued
    candidates = retry + ready
    submitted = 0
    for run, case in candidates[:max(free, 0)]:
        submit(root, slurm, window, run, case, args.dry_run)
        submitted += 1
    print(f"{datetime.now():%Y-%m-%d %H:%M} qos {window['qos']}: {queued} queued, {submitted} submitted "
          f"({len(retry)} again), {len(candidates) - submitted} ready and waiting; {used:.0f} GiB used, "
          f"{backlog:.1f} GiB not uploaded")


if __name__ == "__main__":
    main()
