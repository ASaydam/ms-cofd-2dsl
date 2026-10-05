"""
Publishes the status of the production (hpc/status.py) to the GitHub repository of the dashboard. topup.py calls it
after every call, it publishes at most every status.every_min minutes (config/slurm.yaml) and only for the production
data root (MSCOFD_HOME):

    status.json    the latest status
    history.csv    one line per publication: time, finished tasks per run, running and pending jobs, quota, cases
                   without a steady solution

The repository is cloned to <MSCOFD_HOME>/status-repo (branch status.branch) and pushed over SSH with a deploy key
that can write only to this repository: <MSCOFD_HOME>/config/github_status_key, with the host keys of github.com in
config/github_known_hosts (hpc/git_ssh.sh). Errors are printed and never stop the caller.

    python3 hpc/publish_status.py           publish now (e.g. to test)
"""

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ledger  # noqa: E402
import status as status_module  # noqa: E402
import topup  # noqa: E402

HISTORY_FIELDS = ['time', 'running', 'pending', 'quota_gib', 'unsteady']


def git(repository, *args, timeout=90):
    environment = dict(os.environ, GIT_SSH=str(Path(__file__).resolve().parent / 'git_ssh.sh'))
    result = subprocess.run(['git', *args], cwd=repository, env=environment, capture_output=True, text=True,
                            timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {(result.stderr or result.stdout).strip()[:300]}")
    return result.stdout


def history_row(status):
    row = {'time': status['generated'], 'running': status['queue']['running'],
           'pending': status['queue']['pending'], 'quota_gib': status['quota_gib'], 'unsteady': len(status['unsteady'])}
    row.update({f"done_{r['run']}": r['done'] for r in status['runs']})
    return row


def publish(root, slurm, dataset):
    """collects the status, writes status.json and a line of history.csv in the clone, commits and pushes"""
    settings = slurm['status']
    home = Path(os.environ['MSCOFD_HOME'])
    repository = home / 'status-repo'
    if not (repository / '.git').exists():
        git(home, 'clone', '-q', '--branch', settings['branch'], '--single-branch', settings['repo'], str(repository))
        git(repository, 'config', 'user.name', 'KUACC status')
        git(repository, 'config', 'user.email', 'status@ms-cofd-2dsl.invalid')

    status = status_module.collect(root, slurm, dataset)
    (repository / 'status.json').write_text(json.dumps(status, indent=1) + '\n')
    row = history_row(status)
    history = repository / 'history.csv'
    fields = HISTORY_FIELDS + [f"done_{r['run']}" for r in status['runs']]
    new = not history.exists()
    with open(history, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        if new:
            writer.writeheader()
        writer.writerow(row)
    git(repository, 'add', 'status.json', 'history.csv')
    git(repository, 'commit', '-q', '-m', f"status {status['generated']}")
    git(repository, 'push', '-q', 'origin', settings['branch'])
    return status


def maybe_publish(root, slurm, dataset, force=False):
    """publishes when status is configured, root is the production data root and the last publication is older than
    status.every_min minutes; never raises"""
    try:
        settings = slurm.get('status')
        home = os.environ.get('MSCOFD_HOME')
        if not settings or not home or Path(root).resolve() != Path(home).resolve():
            return
        stamp = Path(root) / 'state' / 'status.last'
        if not force and stamp.exists() and time.time() - stamp.stat().st_mtime < 60 * settings['every_min']:
            return
        stamp.touch()
        status = publish(root, slurm, dataset)
        print(f"status published: {sum(r['done'] for r in status['runs'])} tasks done")
    except Exception as error:  # the dashboard must never stop the production
        print(f"status not published: {type(error).__name__}: {error}")


if __name__ == "__main__":
    root = ledger.data_root()
    slurm, dataset = topup.load_settings(root)
    maybe_publish(root, slurm, dataset, force=True)
