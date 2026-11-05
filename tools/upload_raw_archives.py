"""
Upload of the run archives of the dataset to a public Hugging Face dataset repository, read where they are on
Google Drive:

    python -m tools.upload_raw_archives --archives <archives folder> --index archive_index.csv \
        --cases metadata.parquet --repo <user or org>/ms-cofd-2dsl-raw --output <folder> [--batch N] [--limit N]
    python -m tools.upload_raw_archives --index ... --cases ... --repo ... --output ... --verify-only

Meant for Google Colab with Drive mounted (tools/colab/upload_raw_archives.ipynb), like tools/extract_line_samples.py.
Only the 22,500 runs of the dataset are uploaded: the live archives of archive_index.csv (live_rows of
tools/extract_line_samples.py) of the cases in --cases, the case_name column of the release's metadata.parquet or
metadata.csv. The 18 dropped cases are not in the release, so their base_lu archives stay out; superseded attempts
are not live. Every case of the list must have a live archive of every run.

The repository:
    README.md               the dataset card (docs/hf/raw_README.md), with the licence
    index.csv               run, case, file, bytes, md5 of every archive of the dataset
    <run>/<case>.tar.zst    the archives

A new repository is created private. The card and index.csv are its first commit, before any archive, and only then
is it made public, so it is never public without its card and licence. A private repository is made public the same
way; one that is public already is left as it is. A card or index that differs from the one in the repository is
committed again at the next start, so an edited card is published by running the script again.

The archives follow in commits (HfApi.create_commit) of at most batch archives and batch_bytes of one run. Each
archive is read from Drive once: it is copied to a local staging folder, and its MD5 (checked against
archive_index.csv) and SHA-256 are computed in the same pass. The commit uploads the staged copy, so the uploaded
bytes are the checked ones; huggingface_hub's own SHA-256 of the copy is compared with ours. An archive whose MD5 does
not match is not uploaded and goes to errors.csv. The staged copies of a batch are removed when its commit is done
(or has failed), so at most one batch is on local disk. Nothing in the archives folder is changed
(HfApi.upload_large_folder is not used: it writes its cache into the folder it uploads).

Output (a new folder):
    <output>/done.csv       run, case, bytes, md5_ok, sha256, commit: one row per archive in the repository
                            (commit is empty for an archive that was found in the repository, see below)
    <output>/errors.csv     run, case, error
    <output>/verify.csv     path, run, case, problem, detail (--verify-only)

An archive in done.csv is skipped when the script runs again, so a disconnected Colab session resumes where it
stopped. The repository is listed once at the start. An archive it holds that done.csv does not (a commit that went
through while the session stopped) is read and recorded without uploading it again when its size and SHA-256 match.
When the Drive mount drops, the run stops with DriveLost: a partly copied archive is removed, and the batch is copied
again at the next start.

--verify-only lists the repository with the LFS metadata of its files (size and SHA-256, also for files stored with
xet) and compares every file with done.csv and the index. Missing, extra and mismatched files, and archives that are
in the repository but not in done.csv (unchecked), are printed and written to verify.csv. Nothing is uploaded.

Needs huggingface_hub >= 0.30 (hf_xet is used when it is installed), and pyarrow for a .parquet case list. The token
is that of run(token=...), or of huggingface_hub's login or HF_TOKEN; it is never written or printed.
"""

import argparse
import csv
import hashlib
import io
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.extract_line_samples import (CHUNK, DriveLost, HashingReader, append_rows, live_rows, mount_lost,
                                        read_rows, shards_of)

DONE_FIELDS = ['run', 'case', 'bytes', 'md5_ok', 'sha256', 'commit']
ERROR_FIELDS = ['run', 'case', 'error']
VERIFY_FIELDS = ['path', 'run', 'case', 'problem', 'detail']
INDEX_FIELDS = ['run', 'case', 'file', 'bytes', 'md5']        # index.csv of the repository: no remote paths
CARD, INDEX = 'README.md', 'index.csv'
REPO_TYPE = 'dataset'
CARD_FILE = Path(__file__).resolve().parent.parent / 'docs' / 'hf' / 'raw_README.md'


def case_names(path):
    """the cases of the dataset: the case_name column of the release's metadata.parquet or metadata.csv"""
    path = Path(path)
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        return set(pq.read_table(path, columns=['case_name']).column('case_name').to_pylist())
    with open(path, newline='') as f:
        return {r['case_name'] for r in csv.DictReader(f)}


def select_rows(index, cases):
    """
    the archives to upload: the live archives of archive_index.csv of the cases of the dataset
    :param cases: the case names of the dataset (case_names)
    :return: (index rows sorted by run and case, one per run and case: the last one of an archive indexed twice;
              the (run, case) of the live archives of other cases, which are left out)
    :raise ValueError: when a case of the list has no live archive of one of the runs of the index
    """
    live = live_rows(index)
    runs = sorted({r['run'] for r in live})
    rows = {(r['run'], r['case']): r for r in live if r['case'] in cases}
    others = sorted({(r['run'], r['case']) for r in live if r['case'] not in cases})
    missing = [(run, case) for run in runs for case in sorted(cases) if (run, case) not in rows]
    if missing or not rows:
        raise ValueError(f"{len(missing)} runs of the {len(cases)} listed cases have no live archive in the index, "
                         f"e.g. {missing[:5]} (is the case list the release's metadata?)")
    return [rows[key] for key in sorted(rows)], others


def index_csv(rows):
    """index.csv of the repository: run, case, file, bytes, md5 of the archives"""
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=INDEX_FIELDS, extrasaction='ignore', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode()


def repo_path(row):
    return f"{row['run']}/{row['file']}"


def staged_path(staging, row):
    return Path(staging) / row['run'] / row['file']


def stage_archive(source, target):
    """
    copies one archive to the staging folder and computes its MD5, SHA-256 and size in the same pass; the copy is
    written as <target>.part and renamed when complete, a partial copy is removed
    :return: (md5, sha256, bytes)
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    sha256 = hashlib.sha256()
    try:
        with open(source, 'rb') as raw, open(partial, 'wb') as copy:
            hashing = HashingReader(raw)
            for data in iter(lambda: hashing.read(CHUNK), b''):
                sha256.update(data)
                copy.write(data)
        os.replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return hashing.md5.hexdigest(), sha256.hexdigest(), hashing.bytes


def clear_staging(staging, rows):
    """removes the staged and partial copies of these rows, and the run folders they leave empty"""
    for row in rows:
        path = staged_path(staging, row)
        path.unlink(missing_ok=True)
        path.with_name(path.name + '.part').unlink(missing_ok=True)
    for run in {row['run'] for row in rows}:
        try:
            (Path(staging) / run).rmdir()
        except OSError:                                   # not empty, or not there
            pass


def check_staging(archives, staging):
    """the staging folder must be apart from the archives folder, since its copies are deleted"""
    archives, staging = Path(archives).resolve(), Path(staging).resolve()
    if staging == archives or archives in staging.parents or staging in archives.parents:
        raise ValueError(f"the staging folder {staging} must be outside the archives folder {archives}")


def repo_tree(api, repo):
    """{path: file} of every file of the repository, from one listing; LFS and xet files have lfs.size, lfs.sha256"""
    return {item.path: item for item in api.list_repo_tree(repo, repo_type=REPO_TYPE, recursive=True)
            if hasattr(item, 'size')}                    # folders have no size


def lfs_of(item):
    """(size, sha256) of a file of the listing; sha256 is None for a file not stored with LFS or xet"""
    if getattr(item, 'lfs', None) is None:
        return item.size, None
    return item.lfs.size, item.lfs.sha256


def same_content(item, data):
    """True when a file of the listing holds these bytes (its LFS SHA-256, or its git blob id)"""
    size, sha256 = lfs_of(item)
    if sha256 is not None:
        return size == len(data) and sha256 == hashlib.sha256(data).hexdigest()
    return item.blob_id == hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def drive_answers(archives, error=None):
    """raises DriveLost when the error, or a listing of the archives folder, shows that the mounted Drive is gone"""
    if mount_lost(error) or 'Transport endpoint' in str(error or ''):
        raise DriveLost(str(error)) from error
    try:
        os.listdir(archives)
    except OSError as lost:
        if mount_lost(lost):
            raise DriveLost(str(lost)) from error


def commit(api, repo, operations, message, archives, retries=3):
    """
    create_commit, tried again after a failure (the Hub can time out or limit the rate of commits); huggingface_hub
    skips files that are unchanged in the repository, so repeating a commit that went through does no harm
    :return: the commit id
    :raise DriveLost: when the mounted Drive stops answering
    """
    for attempt in range(retries + 1):
        try:
            return api.create_commit(repo, operations=operations, commit_message=message, repo_type=REPO_TYPE).oid
        except Exception as error:                        # noqa: BLE001  (tried again, raised after the last try)
            drive_answers(archives, error)
            if attempt == retries:
                raise
            print(f"commit failed ({type(error).__name__}: {error}); trying again ({attempt + 1} of {retries})",
                  flush=True)
            time.sleep(60 * (attempt + 1))


def commit_front(api, repo, tree, card, index):
    """
    the card (with the licence) and index.csv, committed when the repository does not hold them as they are; in a
    new repository this is the first commit, before any archive
    :param card, index: the contents (bytes)
    :return: the commit id, or None when both are up to date
    """
    from huggingface_hub import CommitOperationAdd
    files = {CARD: card, INDEX: index}
    changed = [name for name, data in files.items() if name not in tree or not same_content(tree[name], data)]
    if not changed:
        return None
    message = "Dataset card, licence and index" if CARD not in tree else f"Update {', '.join(changed)}"
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=files[name]) for name in changed]
    return api.create_commit(repo, operations=operations, commit_message=message, repo_type=REPO_TYPE).oid


def make_public(api, repo):
    """
    the repository public: HfApi.update_repo_settings(private=False), or update_repo_visibility in versions of
    huggingface_hub that do not have it
    """
    try:
        api.update_repo_settings(repo, private=False, repo_type=REPO_TYPE)
    except (AttributeError, TypeError):
        api.update_repo_visibility(repo, private=False, repo_type=REPO_TYPE)


def prepare(archives, staging, row, tree):
    """
    copies one archive to the staging folder: the MD5 is checked against the index and the SHA-256 kept
    :param tree: the repository listing at the start (repo_tree)
    :return: (row of done.csv, CommitOperationAdd of the staged copy), the operation None when the repository holds
             the archive already
    :raise ValueError: when the MD5 does not match the index (the archive is not uploaded)
    """
    from huggingface_hub import CommitOperationAdd
    staged = staged_path(staging, row)
    md5, sha256, size = stage_archive(Path(archives) / row['run'] / row['file'], staged)
    if md5 != row['md5']:
        raise ValueError(f"MD5 {md5} does not match the index ({row['md5']}); not uploaded")
    done = {'run': row['run'], 'case': row['case'], 'bytes': size, 'md5_ok': True, 'sha256': sha256, 'commit': ''}
    item = tree.get(repo_path(row))
    if item is not None and lfs_of(item) == (size, sha256):
        return done, None
    operation = CommitOperationAdd(path_in_repo=repo_path(row), path_or_fileobj=str(staged))
    info = getattr(operation, 'upload_info', None)      # from huggingface_hub's own read of the staged copy
    if info is not None and (info.sha256.hex() != sha256 or info.size != size):
        raise ValueError("the staged copy differs from the bytes read (SHA-256 of huggingface_hub); not uploaded")
    return done, operation


def compare(tree, rows, done, front):
    """
    the repository against done.csv and the index
    :param tree: the repository listing (repo_tree)
    :param rows: the index rows of the dataset (select_rows)
    :param done: {(run, case): row of done.csv}
    :param front: {path: bytes} of the files next to the archives (the card and index.csv)
    :return: rows of verify.csv; problem is missing, extra, mismatch, or unchecked (an archive in the repository
             whose SHA-256 was never compared with an MD5-checked read, because it is not in done.csv)
    """
    problems = []

    def add(path, problem, detail='', row=None):
        problems.append({'path': path, 'run': row['run'] if row else '', 'case': row['case'] if row else '',
                         'problem': problem, 'detail': detail})

    for name, data in front.items():
        if name not in tree:
            add(name, 'missing')
        elif not same_content(tree[name], data):
            add(name, 'mismatch', "differs from the local file")
    expected = set(front) | {'.gitattributes'}
    for row in rows:
        path = repo_path(row)
        expected.add(path)
        if path not in tree:
            add(path, 'missing', row=row)
            continue
        size, sha256 = lfs_of(tree[path])
        record = done.get((row['run'], row['case']))
        details = []
        if size != int(row['bytes']):
            details.append(f"size {size}, index {row['bytes']}")
        if sha256 is None:
            details.append("no LFS or xet SHA-256 in the listing")
        if record is not None:
            if sha256 is not None and sha256 != record['sha256']:
                details.append(f"SHA-256 {sha256}, done.csv {record['sha256']}")
            if size != int(record['bytes']):
                details.append(f"size {size}, done.csv {record['bytes']}")
        if details:
            add(path, 'mismatch', '; '.join(details), row)
        elif record is None:
            add(path, 'unchecked', "not in done.csv", row)
    for path in sorted(set(tree) - expected):
        add(path, 'extra')
    return problems


def verify(api, repo, rows, done, card, output):
    """compare (one listing of the repository); the problems are printed and written to <output>/verify.csv"""
    tree = repo_tree(api, repo)
    problems = compare(tree, rows, done, {CARD: card, INDEX: index_csv(rows)})
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
    print(f"{len(tree)} files in {repo}, {len(rows)} archives in the index: {counts['missing']} missing, "
          f"{counts['extra']} extra, {counts['mismatch']} mismatched, {counts['unchecked']} unchecked", flush=True)
    return problems


def run(archives, index, cases, repo, output, card=CARD_FILE, batch=100, batch_bytes=10e9, workers=6, limit=None,
        verify_only=False, token=None, staging=None):
    """
    upload every archive of the dataset that is not yet in done.csv, or with verify_only compare the repository
    with done.csv and the index
    :param cases: file with the case names of the dataset (the release's metadata.parquet or metadata.csv)
    :param repo: repository id, <user or org>/<name>; created private, made public after the card is in it
    :param card: the dataset card (docs/hf/raw_README.md)
    :param limit: upload at most this many archives (a test run)
    :param token: Hugging Face token with write access (None: huggingface_hub's login or HF_TOKEN)
    :param staging: local folder (not on Drive) for the copies of one batch
    :return: the rows of verify.csv with verify_only, else the rows added to done.csv
    :raise DriveLost: when the mounted Drive stops answering (the batch is copied again next time)
    """
    from huggingface_hub import HfApi
    output = Path(output)
    staging = Path(staging or Path(tempfile.gettempdir()) / 'hf-upload-staging')
    try:
        output.mkdir(parents=True, exist_ok=True)
        rows, others = select_rows(index, case_names(cases))
        card = Path(card).read_bytes()
        done = {(r['run'], r['case']): r for r in read_rows(output / 'done.csv')}
        if not verify_only:
            check_staging(archives, staging)
    except OSError as error:
        if mount_lost(error):
            raise DriveLost(str(error)) from error
        raise
    api = HfApi(token=token)
    if verify_only:
        return verify(api, repo, rows, done, card, output)

    api.create_repo(repo, repo_type=REPO_TYPE, private=True, exist_ok=True)     # an existing one is not changed
    tree = repo_tree(api, repo)
    front = commit_front(api, repo, tree, card, index_csv(rows))
    if front:
        print(f"card and index committed ({front})", flush=True)
    if api.repo_info(repo, repo_type=REPO_TYPE).private:                       # it holds the card now
        make_public(api, repo)
        print(f"{repo} is public now", flush=True)
    todo = [r for r in rows if (r['run'], r['case']) not in done]
    if limit is not None:
        todo = todo[:limit]
    clear_staging(staging, rows)                          # copies left by a session that stopped
    print(f"{len(rows)} archives of the dataset ({len(others)} live archives of other cases left out), {len(done)} "
          f"done, {len(tree)} files in the repository, {len(todo)} to upload", flush=True)

    def one(row):
        try:
            return (row, *prepare(archives, staging, row, tree), None)
        except Exception as error:                        # noqa: BLE001  (recorded, the other archives go on)
            return row, None, None, error

    written, total_bytes, started = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for part in shards_of(todo, batch, batch_bytes):
            found, uploads, operations, error_rows, lost = [], [], [], [], None
            try:
                for row, done_row, operation, error in pool.map(one, part):
                    if error is not None:
                        if mount_lost(error):
                            lost = error                  # not an error of the archive: it is copied again later
                        else:
                            error_rows.append({'run': row['run'], 'case': row['case'],
                                               'error': f"{type(error).__name__}: {error}"})
                    elif operation is None:
                        found.append(done_row)
                    else:
                        uploads.append(done_row)
                        operations.append(operation)
                if lost is not None:
                    raise DriveLost(str(lost)) from lost
                if operations:
                    message = (f"Add {part[0]['run']}: {uploads[0]['case']} to {uploads[-1]['case']} "
                               f"({len(operations)} archive{'s' if len(operations) > 1 else ''})")
                    oid = commit(api, repo, operations, message, archives)
                    for r in uploads:
                        r['commit'] = oid
            finally:
                clear_staging(staging, part)              # at most one batch on local disk
            try:
                if found or uploads:
                    append_rows(output / 'done.csv', DONE_FIELDS, found + uploads)
                if error_rows:
                    append_rows(output / 'errors.csv', ERROR_FIELDS, error_rows)
            except OSError as error:
                if mount_lost(error):
                    raise DriveLost(str(error)) from error
                raise
            written += found + uploads
            total_bytes += sum(r['bytes'] for r in uploads)
            elapsed = time.time() - started
            print(f"{part[0]['run']:>8s} {part[0]['case']} ... {len(uploads)} uploaded, {len(found)} found in the "
                  f"repository, {len(error_rows)} errors | {total_bytes / 1e9:.1f} GB in {elapsed / 3600:.2f} h "
                  f"({total_bytes / 1e6 / max(elapsed, 1):.0f} MB/s)", flush=True)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--archives', help="folder with <run>/<case>.tar.zst (not needed with --verify-only)")
    parser.add_argument('--index', required=True, help="archive_index.csv of the production")
    parser.add_argument('--cases', required=True, help="metadata.parquet or metadata.csv of the release")
    parser.add_argument('--repo', required=True, help="Hugging Face dataset repository, <user or org>/<name>")
    parser.add_argument('--output', required=True)
    parser.add_argument('--card', default=str(CARD_FILE))
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--batch-bytes', type=float, default=10e9)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--staging', default=None, help="local folder for the copies of one batch")
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only and not args.archives:
        parser.error("--archives is needed for the upload")
    result = run(args.archives, args.index, args.cases, args.repo, args.output, args.card, args.batch,
                 args.batch_bytes, args.workers, args.limit, args.verify_only, staging=args.staging)
    return 1 if args.verify_only and result else 0


if __name__ == "__main__":
    sys.exit(main())
