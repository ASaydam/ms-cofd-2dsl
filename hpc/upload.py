"""
Moves the archives of finished tasks to the remote storage (rclone), so the cluster quota is never filled.

    python -m hpc.upload --pending       uploads every archive left in <data root>/archives (e.g. after a remote
                                         failure), in the container or wherever rclone and MSCOFD_REMOTE are set

An archive is copied to <MSCOFD_REMOTE>/<run>/<case>.tar.zst, the MD5 of the remote copy is compared with the
local file, and only then the local archive is deleted and a line is added to archives/archive_index.csv:

    run, case, file, bytes, md5, remote, uploaded

Without MSCOFD_REMOTE the archive stays in archives/ (the footprint guard of hpc/topup.py counts it).
"""

import argparse
import csv
import fcntl
import hashlib
import os
import subprocess
import time
from pathlib import Path

INDEX_FIELDS = ['run', 'case', 'file', 'bytes', 'md5', 'remote', 'uploaded']


def md5_of(path, block=1 << 24):
    digest = hashlib.md5()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(block), b''):
            digest.update(chunk)
    return digest.hexdigest()


def remote_md5(target):
    """MD5 of a remote file (rclone md5sum), None when it cannot be read"""
    result = subprocess.run(['rclone', 'md5sum', target], capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.split()[0]


def append_index(archives_folder, row):
    """adds a line to archive_index.csv, locked because many jobs finish at the same time"""
    index = Path(archives_folder) / 'archive_index.csv'
    with open(index, 'a', newline='') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        if file.tell() == 0:
            csv.writer(file, lineterminator='\n').writerow(INDEX_FIELDS)
        csv.DictWriter(file, fieldnames=INDEX_FIELDS, lineterminator='\n').writerow(row)
        fcntl.flock(file, fcntl.LOCK_UN)


def upload_archive(archive, run, case, retries=3):
    """
    uploads one archive and deletes it when the remote copy has the same MD5
    :return: True when uploaded, False when no remote is configured
    """
    remote = os.environ.get('MSCOFD_REMOTE', '').rstrip('/')
    archive = Path(archive)
    if not remote:
        print(f"MSCOFD_REMOTE not set, {archive.name} stays in {archive.parent}")
        return False

    target = f"{remote}/{run}/{archive.name}"
    local_md5 = md5_of(archive)
    for attempt in range(1, retries + 1):
        subprocess.run(['rclone', 'copyto', str(archive), target, '--retries', '3', '--low-level-retries', '10',
                        '--drive-chunk-size', '64M'], check=False)
        if remote_md5(target) == local_md5:
            row = {'run': run, 'case': case, 'file': archive.name, 'bytes': archive.stat().st_size, 'md5': local_md5,
                   'remote': target, 'uploaded': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
            append_index(archive.parent.parent, row)
            archive.unlink()
            return True
        time.sleep(30 * attempt)
    raise RuntimeError(f"upload of {archive.name} to {target} failed or the MD5 does not match")


def main():
    parser = argparse.ArgumentParser(description="Upload the archives of finished tasks")
    parser.add_argument("--pending", action="store_true", help="upload every archive left in the archives folder")
    args = parser.parse_args()

    root = Path(os.environ['MSCOFD_DATA_ROOT'])
    if args.pending:
        for archive in sorted((root / 'archives').glob('*/*.tar.zst')):
            run, case = archive.parent.name, archive.name[:-len('.tar.zst')]
            print(f"{run} {case}: {'uploaded' if upload_archive(archive, run, case) else 'kept'}")


if __name__ == "__main__":
    main()
