"""
Task ledger of the production runs: which tasks exist, in which order they are run, and their state.

A task is one run of one case (config/dataset.yaml, runs: base_fo, base_lu, 4x, 16x). The state of a task is a
marker file in <data root>/state/<run>/, so it can be read and changed by the jobs and by the queue manager
(hpc/topup.py) without a database:

    <case>.submitted   submitted to SLURM, the file holds the job id
    <case>.running     the job has started, the file holds the job id and the host
    <case>.done        finished, the solver converged (or stopped at the endTime cap, see the provenance file)
    <case>.failed      failed (the file holds the reason); interrupted tasks are submitted again, see topup.py

This module only uses the Python standard library, it runs on the login node as well as in the container.
"""

import csv
import os
from pathlib import Path

STATES = ('submitted', 'running', 'done', 'failed')

# runs whose result another run starts from (mapFields)
DEPENDENCIES = {'4x': 'base_lu', '16x': 'base_lu'}

# submission order of the runs that are ready: the long runs first, so the last days are not spent on a tail of
# 16x runs (base_lu is needed by the others, it is submitted as soon as it is ready anyway)
RUN_PRIORITY = ['base_lu', '16x', '4x', 'base_fo']


def data_root():
    return Path(os.environ.get('MSCOFD_DATA_ROOT', Path(__file__).resolve().parent.parent / 'outputs'))


def state_folder(run, root=None):
    return Path(root or data_root()) / 'state' / run


def read_states(runs, root=None):
    """
    state of every task that has one: {(run, case): (state, content of the marker file)}
    """
    states = {}
    for run in runs:
        folder = state_folder(run, root)
        if not folder.exists():
            continue
        for name in os.listdir(folder):
            case, _, state = name.rpartition('.')
            if state in STATES:
                states[(run, case)] = (state, (folder / name).read_text().strip())
    return states


def set_state(run, case, state, content='', root=None):
    """
    replaces the marker of a task (the new marker is written first, then the old ones are removed)
    """
    folder = state_folder(run, root)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f".{case}.{state}.tmp"
    temporary.write_text(f"{content}\n")
    os.replace(temporary, folder / f"{case}.{state}")
    for other in STATES:
        if other != state:
            try:
                (folder / f"{case}.{other}").unlink()
            except FileNotFoundError:
                pass


def case_order(manifest_rows):
    """
    case names in an interleaved order: index 1 of every size and geometry, then index 2, ... so that the finished
    cases stay balanced over the sizes and geometries at any time
    """
    return [row['case_name'] for row in sorted(manifest_rows, key=lambda row: (int(row['index']), row['size'],
                                                                                row['geometry']))]


def read_manifest(path):
    with open(path, newline='') as file:
        return list(csv.DictReader(file))


def ready_tasks(cases, runs, states):
    """
    tasks that can be submitted now, in submission order: no state yet (or failed and resubmittable, decided by the
    caller) and the run they start from is done
    :param cases: case names in case_order
    :param runs: runs of config/dataset.yaml
    :param states: read_states() without the tasks the caller does not want to submit
    """
    ready = []
    for run in sorted(runs, key=RUN_PRIORITY.index):
        for case in cases:
            if (run, case) in states:
                continue
            source = DEPENDENCIES.get(run)
            if source and states.get((source, case), ('',))[0] != 'done':
                continue
            ready.append((run, case))
    return ready


def main():
    """
    command line for the job scripts: python3 hpc/ledger.py set <run> <case> <state> [<content>]
    """
    import sys
    if len(sys.argv) < 5 or sys.argv[1] != 'set' or sys.argv[4] not in STATES:
        sys.exit("usage: ledger.py set <run> <case> <state> [<content>]")
    set_state(sys.argv[2], sys.argv[3], sys.argv[4], ' '.join(sys.argv[5:]))


if __name__ == "__main__":
    main()
