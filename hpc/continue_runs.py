"""
Continues runs that stopped at their iteration cap with the residual still falling, from their latest time step to a
higher cap (docs/decisions.md):

    python3 hpc/continue_runs.py <run> <cap> <case> [<case> ...]
    python3 hpc/continue_runs.py <run> <cap> --capped        every case of the dataset whose last run stopped at its cap

For every case, one after the other:
1. the results and the Drive archive of the stopped run are kept as superseded (results/<run>/superseded/
   <case>.cap<iterations>*, <remote>/<run>/superseded/<case>.cap<iterations>.tar.zst; archive_index.csv updated)
2. the case folder is restored from that archive into the work folder (the attribute work folder is made again)
3. the cap is written to iterations.txt (an existing line of the task is replaced)
4. the state of the task is cleared last, so topup.py submits it only now; hpc/run_task.py then continues the run from
   its latest time step

Runs on the login node, with a Python that has zstandard (the cluster's anaconda) and rclone (module load rclone),
after sourcing hpc/env.sh (MSCOFD_DATA_ROOT, MSCOFD_REMOTE).
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ledger  # noqa: E402
import topup  # noqa: E402


def capped_cases(root, dataset, run):
    """cases of the dataset whose last <run> stopped at its cap (results/<run>/summary.csv)"""
    states = ledger.read_states(list(dataset['runs']), root)
    cases, discarded, _ = topup.dataset_cases(root, dataset, states, topup.cases_file_of(root))
    members = set(cases) - set(discarded)
    with open(root / 'results' / run / 'summary.csv', newline='') as file:
        latest = {row['case']: row for row in csv.DictReader(file)}
    return [case for case in cases if case in members and latest.get(case, {}).get('status') == 'end_time'
            and states.get((run, case), ('', ''))[0] == 'done']


def set_cap(root, run, case, cap):
    """the line "<run> <case> <cap>" in iterations.txt, in place of an earlier one of the task"""
    path = root / 'iterations.txt'
    lines = path.read_text().splitlines() if path.exists() else []
    lines = [line for line in lines if line.split()[:2] != [run, case]] + [f"{run} {case} {cap}"]
    temporary = path.with_name('.iterations.txt.tmp')
    temporary.write_text('\n'.join(lines) + '\n')
    os.replace(temporary, path)


def restore(archive, work, case):
    """the case folder (not the attribute work folder) from a .tar.zst archive into work/"""
    import zstandard

    restored = 0
    with open(archive, 'rb') as raw, zstandard.ZstdDecompressor().stream_reader(raw) as stream, \
            tarfile.open(fileobj=stream, mode='r|') as tar:
        for member in tar:
            if member.name.startswith('/') or '..' in Path(member.name).parts:
                raise ValueError(f"unsafe archive member {member.name}")
            if member.name == case or member.name.startswith(case + '/'):
                tar.extract(member, work)
                restored += 1
    return restored


def continue_case(root, remote, run, case, cap):
    results, work = root / 'results' / run, root / 'work' / run
    if (work / case).exists():
        raise FileExistsError(f"{work / case} exists")
    iterations = json.loads((results / f"{case}_provenance.json").read_text())['iterations']
    tag = f"cap{int(float(iterations))}"

    # 1. keep the stopped run
    (results / 'superseded').mkdir(exist_ok=True)
    kept = f"{remote}/{run}/superseded/{case}.{tag}.tar.zst"
    subprocess.run(['rclone', 'moveto', f"{remote}/{run}/{case}.tar.zst", kept], check=True)
    for suffix in ('.npz', '_attributes.json', '_provenance.json', '_surface.csv'):
        if (results / f"{case}{suffix}").exists():
            os.replace(results / f"{case}{suffix}", results / 'superseded' / f"{case}.{tag}{suffix}")
    archives = root / 'archives'
    with ledger.DirectoryLock(archives / '.index.lock'):
        with open(archives / 'archive_index.csv', newline='') as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            if row['run'] == run and row['case'] == case and '/superseded/' not in row['remote']:
                row['file'], row['remote'] = f"superseded/{case}.{tag}.tar.zst", kept
        temporary = archives / '.archive_index.csv.tmp'
        with open(temporary, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, archives / 'archive_index.csv')

    # 2. the case folder
    download = work / f".restore-{case}.tar.zst"
    subprocess.run(['rclone', 'copyto', kept, str(download)], check=True)
    try:
        restored = restore(download, work, case)
    finally:
        download.unlink()

    # 3. and 4.
    set_cap(root, run, case, cap)
    for name in (f"{case}.done", f"{case}.attempts"):
        try:
            (root / 'state' / run / name).unlink()
        except FileNotFoundError:
            pass
    return iterations, restored


def main():
    parser = argparse.ArgumentParser(description="Continue runs that stopped at their cap to a higher cap")
    parser.add_argument('run')
    parser.add_argument('cap', type=int)
    parser.add_argument('cases', nargs='*')
    parser.add_argument('--capped', action='store_true', help="every case whose last run stopped at its cap")
    args = parser.parse_args()

    root = ledger.data_root()
    _, dataset = topup.load_settings(root)
    cases = capped_cases(root, dataset, args.run) if args.capped else args.cases
    print(f"{len(cases)} {args.run} runs continued to {args.cap} iterations")
    for case in cases:
        try:
            iterations, restored = continue_case(root, os.environ['MSCOFD_REMOTE'], args.run, case, args.cap)
            print(f"  {case}: from {iterations} iterations, {restored} files restored", flush=True)
        except Exception as error:  # the other cases go on; this one keeps its state
            print(f"  {case}: NOT continued, {type(error).__name__}: {error}", flush=True)


if __name__ == '__main__':
    main()
