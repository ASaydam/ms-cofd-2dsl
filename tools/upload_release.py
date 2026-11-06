"""
Upload of the release (arrays and tables) to a Hugging Face dataset repository, read where it is on Google Drive:

    python -m tools.upload_release --release <release folder> --repo <user or org>/ms-cofd-2dsl --output <folder> \
        [--md5 <MD5 of checksums.md5>] [--batch N] [--limit N]
    python -m tools.upload_release --release ... --repo ... --output ... --verify-only

Meant for Google Colab with Drive mounted (tools/colab/upload_release.ipynb), like tools/upload_raw_archives.py, whose
staging, commit and listing functions it uses. Every file of the release's checksums.md5 is uploaded at its path in the
release, so `md5sum -c checksums.md5` works in a download of the repository.

The repository:
    README.md               the dataset card (docs/hf/release_README.md), with the licence
    checksums.md5           the release's checksums.md5, unchanged
    DATACARD.md, ...        every file listed in checksums.md5, at its path (<run>/<split>.h5, metadata.parquet, ...)

The MD5 of checksums.md5 is checked first (--md5), so only the release it identifies is uploaded. A new repository is
created private, and the card and checksums.md5 are its first commit. The run never changes its visibility: it is made
public by hand, on its settings page, once the comparison at the end of a run (the listing of the repository against
checksums.md5, done.csv and the card) reports no problem. A card that differs from the one in the repository is
committed again at the next start.

The files follow in commits (HfApi.create_commit) of at most batch files and batch_bytes of one folder; a larger file
goes alone. The top-level files go first, then the runs from the smallest grid to the largest. Each file is read from
Drive once: it is copied to a local staging folder, and its MD5 (checked against checksums.md5) and SHA-256 are
computed in the same pass. The commit uploads the staged copy, so the uploaded bytes are the checked ones;
huggingface_hub's own SHA-256 of the copy is compared with ours. A file whose MD5 does not match is not uploaded and
goes to errors.csv. The staged copies of a batch are removed when its commit is done (or has failed), so at most one
batch is on local disk; a batch that does not fit into the free space of the staging folder stops the run before it
is copied. Nothing in the release folder is changed.

Output (a new folder):
    <output>/done.csv       path, bytes, md5_ok, sha256, commit: one row per file in the repository (commit is empty
                            for a file that was found in the repository)
    <output>/errors.csv     path, error
    <output>/verify.csv     path, problem, detail (--verify-only, and at the end of every run)

A file in done.csv is skipped when the script runs again, so a disconnected Colab session resumes where it stopped.
A file the repository holds that done.csv does not (a commit that went through while the session stopped) is read
and recorded without uploading it again when its size and SHA-256 match. When the Drive mount drops, the run stops
with DriveLost: a partly copied file is removed, and the batch is copied again at the next start.

--verify-only lists the repository with the LFS metadata of its files (size and SHA-256, also for files stored with
xet) and compares every file with done.csv and checksums.md5. Missing, extra and mismatched files, and files in the
repository but not in done.csv (unchecked), are printed and written to verify.csv. Nothing is uploaded.

Needs huggingface_hub >= 0.30 (hf_xet is used when it is installed). The token is that of run(token=...), or of
huggingface_hub's login or HF_TOKEN; it is never written or printed.
"""

import argparse
import csv
import hashlib
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.extract_line_samples import DriveLost, append_rows, mount_lost, read_rows, shards_of
from tools.upload_raw_archives import (REPO_TYPE, check_staging, commit, drive_answers, lfs_of, repo_tree, same_content,
                                       stage_archive)

DONE_FIELDS = ['path', 'bytes', 'md5_ok', 'sha256', 'commit']
ERROR_FIELDS = ['path', 'error']
VERIFY_FIELDS = ['path', 'problem', 'detail']
CARD, CHECKSUMS = 'README.md', 'checksums.md5'
RUN_ORDER = ['0.25x', 'base_fo', 'base_lu', '4x', '16x']      # smallest grid first
CARD_FILE = Path(__file__).resolve().parent.parent / 'docs' / 'hf' / 'release_README.md'
SPACE_MARGIN = 2e9                                            # bytes left free on the staging disk


def read_checksums(release, md5=None):
    """
    the release's checksums.md5
    :param md5: the expected MD5 of checksums.md5 (None: not checked)
    :return: (its bytes, rows sorted for the upload: path, run (its folder, '' at the top level), md5, bytes)
    :raise ValueError: when its MD5 differs, a path is not a plain relative path, or a listed file is missing
    """
    release = Path(release)
    data = (release / CHECKSUMS).read_bytes()
    if md5 is not None and hashlib.md5(data).hexdigest() != md5:
        raise ValueError(f"{CHECKSUMS} has the MD5 {hashlib.md5(data).hexdigest()}, not {md5}: not this release")
    rows = []
    for line in data.decode().splitlines():
        if not line.strip():
            continue
        checksum, path = line.split(None, 1)
        path = path.lstrip('*')                               # md5sum's binary mark
        parts = path.split('/')
        if path.startswith('/') or '..' in parts or path in (CARD, CHECKSUMS) or len(parts) > 2:
            raise ValueError(f"unexpected path in {CHECKSUMS}: {path}")
        file = release / path
        if not file.is_file():
            raise ValueError(f"{path} is listed in {CHECKSUMS} but not in {release}")
        rows.append({'path': path, 'run': parts[0] if len(parts) == 2 else '', 'md5': checksum,
                     'bytes': file.stat().st_size})

    def order(row):
        run = row['run']
        return (run != '', RUN_ORDER.index(run) if run in RUN_ORDER else len(RUN_ORDER), run, row['path'])
    return data, sorted(rows, key=order)


def staged_path(staging, row):
    return Path(staging) / row['path']


def clear_staging(staging, rows):
    """removes the staged and partial copies of these rows, and the folders they leave empty"""
    for row in rows:
        path = staged_path(staging, row)
        path.unlink(missing_ok=True)
        path.with_name(path.name + '.part').unlink(missing_ok=True)
    for run in {row['run'] for row in rows if row['run']}:
        try:
            (Path(staging) / run).rmdir()
        except OSError:                                       # not empty, or not there
            pass


def check_space(staging, rows):
    """the batch must fit into the free space of the staging disk, with SPACE_MARGIN to spare"""
    Path(staging).mkdir(parents=True, exist_ok=True)
    free, need = shutil.disk_usage(staging).free, sum(int(r['bytes']) for r in rows)
    if need + SPACE_MARGIN > free:
        raise ValueError(f"the batch of {rows[0]['path']} needs {need / 1e9:.1f} GB, the staging folder has "
                         f"{free / 1e9:.1f} GB free ({SPACE_MARGIN / 1e9:.0f} GB are kept free)")


def commit_front(api, repo, tree, card, checksums):
    """
    the card (with the licence) and checksums.md5, committed when the repository does not hold them as they are; in a
    new repository this is the first commit
    :param card, checksums: the contents (bytes)
    :return: the commit id, or None when both are up to date
    """
    from huggingface_hub import CommitOperationAdd
    files = {CARD: card, CHECKSUMS: checksums}
    changed = [name for name, data in files.items() if name not in tree or not same_content(tree[name], data)]
    if not changed:
        return None
    message = "Dataset card, licence and checksums" if CARD not in tree else f"Update {', '.join(changed)}"
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=files[name]) for name in changed]
    return api.create_commit(repo, operations=operations, commit_message=message, repo_type=REPO_TYPE).oid


def prepare(release, staging, row, tree):
    """
    copies one file to the staging folder: the MD5 is checked against checksums.md5 and the SHA-256 kept
    :param tree: the repository listing at the start (repo_tree)
    :return: (row of done.csv, CommitOperationAdd of the staged copy), the operation None when the repository holds
             the file already
    :raise ValueError: when the MD5 does not match checksums.md5 (the file is not uploaded)
    """
    from huggingface_hub import CommitOperationAdd
    staged = staged_path(staging, row)
    md5, sha256, size = stage_archive(Path(release) / row['path'], staged)
    if md5 != row['md5']:
        raise ValueError(f"MD5 {md5} does not match {CHECKSUMS} ({row['md5']}); not uploaded")
    done = {'path': row['path'], 'bytes': size, 'md5_ok': True, 'sha256': sha256, 'commit': ''}
    item = tree.get(row['path'])
    if item is not None and (lfs_of(item) == (size, sha256) or lfs_of(item) == (size, None)
                             and same_content(item, staged.read_bytes())):   # a small file stored without LFS
        return done, None
    operation = CommitOperationAdd(path_in_repo=row['path'], path_or_fileobj=str(staged))
    info = getattr(operation, 'upload_info', None)          # from huggingface_hub's own read of the staged copy
    if info is not None and (info.sha256.hex() != sha256 or info.size != size):
        raise ValueError("the staged copy differs from the bytes read (SHA-256 of huggingface_hub); not uploaded")
    return done, operation


def compare(tree, rows, done, front, content=None):
    """
    the repository against done.csv and checksums.md5
    :param tree: the repository listing (repo_tree)
    :param rows: the files of checksums.md5 (read_checksums)
    :param done: {path: row of done.csv}
    :param front: {path: bytes} of the card and checksums.md5
    :param content: function of a path that returns the release file's bytes, for the small files the Hub stores
                    without LFS (their listing has a git blob id, no SHA-256); None: such a file is a mismatch
    :return: rows of verify.csv; problem is missing, extra, mismatch, or unchecked (a file in the repository whose
             SHA-256 was never compared with an MD5-checked read, because it is not in done.csv)
    """
    problems = []

    def add(path, problem, detail=''):
        problems.append({'path': path, 'problem': problem, 'detail': detail})

    for name, data in front.items():
        if name not in tree:
            add(name, 'missing')
        elif not same_content(tree[name], data):
            add(name, 'mismatch', "differs from the local file")
    expected = set(front) | {'.gitattributes'}
    for row in rows:
        path = row['path']
        expected.add(path)
        if path not in tree:
            add(path, 'missing')
            continue
        size, sha256 = lfs_of(tree[path])
        record = done.get(path)
        details = []
        if size != int(row['bytes']):
            details.append(f"size {size}, release {row['bytes']}")
        elif sha256 is None:
            if content is None:
                details.append("no LFS or xet SHA-256 in the listing")
            elif not same_content(tree[path], content(path)):
                details.append("git blob differs from the release file")
        if record is not None:
            if sha256 is not None and sha256 != record['sha256']:
                details.append(f"SHA-256 {sha256}, done.csv {record['sha256']}")
            if size != int(record['bytes']):
                details.append(f"size {size}, done.csv {record['bytes']}")
        if details:
            add(path, 'mismatch', '; '.join(details))
        elif record is None:
            add(path, 'unchecked', "not in done.csv")
    for path in sorted(set(tree) - expected):
        add(path, 'extra')
    return problems


def verify(api, repo, rows, done, front, output, release):
    """
    compare (one listing of the repository); the problems are printed and written to <output>/verify.csv
    :param release: the release folder, read for the small files the Hub stores without LFS
    """
    tree = repo_tree(api, repo)
    problems = compare(tree, rows, done, front, content=lambda path: (Path(release) / path).read_bytes())
    try:
        with open(Path(output) / 'verify.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=VERIFY_FIELDS)
            writer.writeheader()
            writer.writerows(problems)
    except OSError as error:
        if mount_lost(error):
            raise DriveLost(str(error)) from error
        raise
    counts = {name: sum(p['problem'] == name for p in problems) for name in ('missing', 'extra', 'mismatch',
                                                                              'unchecked')}
    print(f"{len(tree)} files in {repo}, {len(rows)} files in {CHECKSUMS}: {counts['missing']} missing, "
          f"{counts['extra']} extra, {counts['mismatch']} mismatched, {counts['unchecked']} unchecked", flush=True)
    return problems


def run(release, repo, output, card=CARD_FILE, md5=None, batch=20, batch_bytes=10e9, workers=4, limit=None,
        verify_only=False, token=None, staging=None):
    """
    upload every file of the release that is not yet in done.csv, then compare the repository with the release; with
    verify_only only the comparison
    :param release: the release folder (holds checksums.md5)
    :param repo: repository id, <user or org>/<name>; created private (made public by hand)
    :param card: the dataset card (docs/hf/release_README.md)
    :param md5: the expected MD5 of checksums.md5 (None: not checked)
    :param limit: upload at most this many files (a test run)
    :param token: Hugging Face token with write access (None: huggingface_hub's login or HF_TOKEN)
    :param staging: local folder (not on Drive) for the copies of one batch
    :return: the rows of verify.csv
    :raise DriveLost: when the mounted Drive stops answering (the batch is copied again next time)
    """
    from huggingface_hub import HfApi
    output = Path(output)
    staging = Path(staging or Path(tempfile.gettempdir()) / 'hf-upload-staging')
    try:
        output.mkdir(parents=True, exist_ok=True)
        checksums, rows = read_checksums(release, md5)
        card = Path(card).read_bytes()
        done = {r['path']: r for r in read_rows(output / 'done.csv')}
        if not verify_only:
            check_staging(release, staging)
    except OSError as error:
        if mount_lost(error):
            raise DriveLost(str(error)) from error
        raise
    front = {CARD: card, CHECKSUMS: checksums}
    api = HfApi(token=token)
    if verify_only:
        return verify(api, repo, rows, done, front, output, release)

    api.create_repo(repo, repo_type=REPO_TYPE, private=True, exist_ok=True)     # an existing one is not changed
    tree = repo_tree(api, repo)
    first = commit_front(api, repo, tree, card, checksums)
    if first:
        print(f"card and {CHECKSUMS} committed ({first})", flush=True)
    todo = [r for r in rows if r['path'] not in done]
    if limit is not None:
        todo = todo[:limit]
    clear_staging(staging, rows)                              # copies left by a session that stopped
    print(f"{len(rows)} files in {CHECKSUMS} ({sum(r['bytes'] for r in rows) / 1e9:.1f} GB), {len(done)} done, "
          f"{len(tree)} files in the repository, {len(todo)} to upload", flush=True)

    def one(row):
        try:
            return (row, *prepare(release, staging, row, tree), None)
        except Exception as error:                            # noqa: BLE001  (recorded, the other files go on)
            return row, None, None, error

    total_bytes, started, failed = 0, time.time(), False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for part in shards_of(todo, batch, batch_bytes):
            check_space(staging, part)
            found, uploads, operations, error_rows, lost = [], [], [], [], None
            try:
                for row, done_row, operation, error in pool.map(one, part):
                    if error is not None:
                        if mount_lost(error):
                            lost = error                      # not an error of the file: it is copied again later
                        else:
                            error_rows.append({'path': row['path'], 'error': f"{type(error).__name__}: {error}"})
                    elif operation is None:
                        found.append(done_row)
                    else:
                        uploads.append(done_row)
                        operations.append(operation)
                if lost is not None:
                    raise DriveLost(str(lost)) from lost
                if operations:
                    names = [r['path'] for r in uploads]
                    message = (f"Add {names[0]}" if len(names) == 1 else f"Add {len(names)} files of {part[0]['run']}/"
                               if part[0]['run'] else f"Add {len(names)} top-level files")
                    oid = commit(api, repo, operations, message, release)
                    for r in uploads:
                        r['commit'] = oid
            finally:
                clear_staging(staging, part)                  # at most one batch on local disk
            try:
                if found or uploads:
                    append_rows(output / 'done.csv', DONE_FIELDS, found + uploads)
                if error_rows:
                    append_rows(output / 'errors.csv', ERROR_FIELDS, error_rows)
            except OSError as error:
                if mount_lost(error):
                    raise DriveLost(str(error)) from error
                raise
            failed = failed or bool(error_rows)
            for r in found + uploads:
                done[r['path']] = {k: str(v) for k, v in r.items()}
            total_bytes += sum(r['bytes'] for r in uploads)
            elapsed = time.time() - started
            print(f"{part[0]['path']} ... {len(uploads)} uploaded, {len(found)} found in the repository, "
                  f"{len(error_rows)} errors | {total_bytes / 1e9:.1f} GB in {elapsed / 3600:.2f} h "
                  f"({total_bytes / 1e6 / max(elapsed, 1):.0f} MB/s)", flush=True)

    drive_answers(release)
    problems = verify(api, repo, rows, done, front, output, release)
    if problems or failed:
        print(f"{repo} does not match the release yet", flush=True)
    elif api.repo_info(repo, repo_type=REPO_TYPE).private:
        print(f"{repo} holds the whole release and stays private: make it public on its settings page", flush=True)
    else:
        print(f"{repo} holds the whole release", flush=True)
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--release', required=True, help="the release folder, with checksums.md5")
    parser.add_argument('--repo', required=True, help="Hugging Face dataset repository, <user or org>/<name>")
    parser.add_argument('--output', required=True)
    parser.add_argument('--card', default=str(CARD_FILE))
    parser.add_argument('--md5', default=None, help="the expected MD5 of checksums.md5")
    parser.add_argument('--batch', type=int, default=20)
    parser.add_argument('--batch-bytes', type=float, default=10e9)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--staging', default=None, help="local folder for the copies of one batch")
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    problems = run(args.release, args.repo, args.output, args.card, args.md5, args.batch, args.batch_bytes,
                   args.workers, args.limit, args.verify_only, staging=args.staging)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
