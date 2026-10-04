"""
Status of the production: tasks per run and state, the queue, the footprint, failures and the compute used so far.

    python3 hpc/status.py                 summary
    python3 hpc/status.py --failures 20   with the last 20 failures and their reasons

Runs with the Python of the cluster (standard library and PyYAML), outside of the container. The compute used is
read from results/<run>/summary.csv, one line per finished task (written by hpc/run_task.py).
"""

import argparse
import csv
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ledger  # noqa: E402
import topup  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Status of the production")
    parser.add_argument("--failures", type=int, default=5, help="number of failures to list")
    args = parser.parse_args()

    slurm, dataset = topup.load_settings()
    root = ledger.data_root()
    runs = list(dataset['runs'])
    manifest_path = root / 'geometry' / 'manifest.csv'
    cases = len(ledger.read_manifest(manifest_path)) if manifest_path.exists() else 0
    states = ledger.read_states(runs, root)
    jobs = topup.our_jobs(slurm['job_name_prefix'])
    window = topup.current_window(slurm)

    print(f"{datetime.now():%Y-%m-%d %H:%M}  data root {root}  ({cases} cases)")
    print(f"qos window: {window['qos'] if window else 'none'}  queue: "
          f"{sum(state == 'RUNNING' for _, state, _ in jobs.values())} running, "
          f"{sum(state == 'PENDING' for _, state, _ in jobs.values())} pending")

    print(f"\n{'run':8s} {'done':>6s} {'running':>8s} {'submitted':>10s} {'failed':>7s} {'left':>6s} "
          f"{'core-h':>8s} {'mean h':>7s} {'end_time':>9s}")
    total_hours = 0.0
    for run in runs:
        counts = Counter(state for (r, _), (state, _) in states.items() if r == run)
        summary = root / 'results' / run / 'summary.csv'
        rows = list(csv.DictReader(open(summary))) if summary.exists() else []
        hours = sum(float(row['execution_time_s'] or 0) for row in rows) / 3600
        total_hours += hours
        capped = sum(row['status'] == 'end_time' for row in rows)
        left = cases - counts['done']
        print(f"{run:8s} {counts['done']:6d} {counts['running']:8d} {counts['submitted']:10d} {counts['failed']:7d} "
              f"{left:6d} {hours:8.1f} {hours / len(rows) if rows else 0:7.2f} {capped:9d}")
    print(f"solver core-hours so far: {total_hours:.0f}")

    used, backlog = topup.used_gib(root), topup.backlog_gib(root)
    limits = slurm['footprint']
    flag = '  WARNING' if used > limits['pause_used_gib'] else ''
    flag = '  ALARM: close to the quota' if used > 800 else flag
    print(f"\nquota used: {used:.0f} GiB (pause at {limits['pause_used_gib']}), not uploaded: {backlog:.1f} GiB "
          f"(pause at {limits['pause_backlog_gib']}){flag}")
    seeds = root / 'seeds'
    if seeds.exists():
        print(f"base_lu solutions kept for the fine runs: {sum(1 for _ in seeds.iterdir())}")

    failed = [(run, case, content) for (run, case), (state, content) in states.items() if state == 'failed']
    if failed and args.failures:
        print(f"\nfailures ({len(failed)}):")
        for run, case, content in failed[-args.failures:]:
            print(f"  {run:8s} {case:28s} {content[:150]}")

    stuck = [(run, case) for (run, case), (state, _) in states.items()
             if state in ('submitted', 'running') and (run, case) not in jobs]
    if stuck:
        print(f"\n{len(stuck)} tasks without a job (interrupted), topup.py submits them again")
    try:
        log = subprocess.run(['tail', '-1', str(root / 'state' / 'topup.log')], capture_output=True, text=True).stdout
        if log:
            print(f"last topup: {log.strip()}")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
