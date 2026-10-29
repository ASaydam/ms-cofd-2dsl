"""
Line samples of every run, read from the run archives (the sampled lines are kept only in the archives, and the wake
velocity within the obstacle's y extent is computed from them):

    python -m tools.extract_line_samples --archives <archives folder> --index archive_index.csv --output <folder>

Meant for Google Colab with Drive mounted (tools/colab/extract_line_samples.ipynb), so the 1.58 TB of archives are
read where they are. Each archive <run>/<case>.tar.zst is streamed (zstd and tar in streaming mode, nothing is
unpacked to disk). The five sets of the attribute run, <case>_attributes/postProcessing/lines/<time>/<set>_p_U.xy,
are kept, and the MD5 of the whole archive is checked against archive_index.csv. Only the live archives of the index
are read (the 22,500 runs of the dataset and the 18 base_lu runs of the dropped cases), not the superseded attempts.

Output:
    <output>/<run>/lines_<first case>_<n>.npz  for every case of the shard: <case>/time and, for every set,
                                               <case>/<set> of shape (points, 4): y (m), p, u_x, u_y (float32)
    <output>/done.csv                          run, case, time, md5_ok, bytes, shard: one row per finished archive
    <output>/errors.csv                        run, case, error

The points of a set are those OpenFOAM wrote: the wall-to-wall lines have all 2401, and the lines through the
centroid stop at the obstacle. A case in done.csv is skipped when the script runs again, so a disconnected Colab
session resumes where it stopped.

The Drive mount of Colab can drop during a long run ("Transport endpoint is not connected"). A shard is therefore
written to a local staging folder first, with its rows of done.csv next to it, and then copied to the output; when
the mount drops, the run stops with DriveLost and the staged shards are copied at the next start. A shard holds at
most shard_size cases and shard_bytes of archives, so that a drop loses little reading.

Needs numpy and zstandard.
"""

import argparse
import csv
import errno
import hashlib
import io
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SETS = ('wakeLine', 'centroidBelow', 'centroidAbove', 'inletSection', 'outletSection')
MEMBER = re.compile(r'(?:^|/)[^/]+_attributes/postProcessing/lines/([^/]+)/(\w+)_p_U\.xy$')
DONE_FIELDS = ['run', 'case', 'time', 'md5_ok', 'bytes', 'shard']
CHUNK = 1 << 22


class DriveLost(RuntimeError):
    """the mounted Drive stopped answering; remount it and run again"""


def mount_lost(error):
    return isinstance(error, OSError) and (error.errno == errno.ENOTCONN or 'Transport endpoint' in str(error))


class HashingReader:
    """file object that keeps the MD5 and the byte count of everything read through it"""

    def __init__(self, raw):
        self.raw, self.md5, self.bytes = raw, hashlib.md5(), 0

    def read(self, size=-1):
        data = self.raw.read(size)
        self.md5.update(data)
        self.bytes += len(data)
        return data

    def drain(self):
        while self.read(CHUNK):
            pass


def parse_set(text):
    """an OpenFOAM raw set (columns y, p, Ux, Uy, Uz) as float32 y, p, u_x, u_y"""
    data = np.loadtxt(io.BytesIO(text), dtype=np.float64, ndmin=2)
    return data[:, :4].astype(np.float32)


def read_archive(path):
    """
    the five line sets of one archive
    :return: (time, {set: array}, md5 of the archive, bytes)
    """
    import zstandard
    found = {}
    with open(path, 'rb') as raw:
        hashing = HashingReader(raw)
        reader = zstandard.ZstdDecompressor().stream_reader(hashing, read_size=CHUNK, read_across_frames=True)
        with tarfile.open(fileobj=reader, mode='r|') as tar:
            for member in tar:
                match = MEMBER.search(member.name)
                if match and member.isfile() and match.group(2) in SETS:
                    found.setdefault(match.group(1), {})[match.group(2)] = parse_set(tar.extractfile(member).read())
        hashing.drain()
    if len(found) != 1:
        raise ValueError(f"line sets of {len(found)} time folders: {sorted(found)}")
    (time_name, sets), = found.items()
    missing = [s for s in SETS if s not in sets]
    if missing:
        raise ValueError(f"sets missing: {missing}")
    return time_name, sets, hashing.md5.hexdigest(), hashing.bytes


def live_rows(index):
    """the live archives of archive_index.csv, i.e. the rows of <case>.tar.zst outside superseded/"""
    with open(index, newline='') as f:
        rows = [r for r in csv.DictReader(f)
                if r['file'] == f"{r['case']}.tar.zst" and '/superseded/' not in r['remote']]
    return sorted(rows, key=lambda r: (r['run'], r['case']))


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def append_rows(path, fields, rows):
    new = not path.exists()
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def shards_of(rows, shard_size, shard_bytes):
    """consecutive rows of one run, at most shard_size of them and shard_bytes of archives (at least one row)"""
    shards = []
    for row in rows:
        last = shards[-1] if shards else None
        if (last and last[0]['run'] == row['run'] and len(last) < shard_size
                and sum(int(r['bytes']) for r in last) + int(row['bytes']) <= shard_bytes):
            last.append(row)
        else:
            shards.append([row])
    return shards


def publish(staging, output):
    """copy the staged shards to the output and append their rows to done.csv; a shard leaves staging when copied"""
    for rows_path in sorted(staging.glob('*.csv')):
        npz = rows_path.with_suffix('.npz')
        rows = read_rows(rows_path)
        target = output / rows[0]['shard']
        try:
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(npz, target)
            append_rows(output / 'done.csv', DONE_FIELDS, rows)
        except OSError as error:
            if mount_lost(error):
                raise DriveLost(str(error)) from error
            raise
        npz.unlink()
        rows_path.unlink()


def run(archives, index, output, workers=6, shard_size=250, shard_bytes=20e9, limit=None, staging=None):
    """
    read every live archive not yet in done.csv
    :param limit: read at most this many archives (a test run)
    :param staging: local folder for finished shards before they are copied to the output
    :raise DriveLost: when the mounted Drive stops answering (the finished shards stay in staging)
    """
    archives, output = Path(archives), Path(output)
    staging = Path(staging or Path(tempfile.gettempdir()) / 'line-samples-staging')
    staging.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir(parents=True, exist_ok=True)
        publish(staging, output)
        done = {(r['run'], r['case']) for r in read_rows(output / 'done.csv')}
        todo = [r for r in live_rows(index) if (r['run'], r['case']) not in done]
    except OSError as error:
        if mount_lost(error):
            raise DriveLost(str(error)) from error
        raise
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(done)} archives done, {len(todo)} to read", flush=True)

    def one(row):
        try:
            time_name, sets, md5, size = read_archive(archives / row['run'] / row['file'])
            return row, time_name, sets, md5 == row['md5'], size, None
        except Exception as error:                        # noqa: BLE001  (recorded, the other archives go on)
            return row, None, None, False, 0, error

    total_bytes, started = 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for shard in shards_of(todo, shard_size, shard_bytes):
            arrays, done_rows, error_rows, lost = {}, [], [], None
            for row, time_name, sets, md5_ok, size, error in pool.map(one, shard):
                if error is not None:
                    if mount_lost(error):
                        lost = error                      # not an error of the archive: it is read again later
                    else:
                        error_rows.append({'run': row['run'], 'case': row['case'],
                                           'error': f"{type(error).__name__}: {error}"})
                    continue
                arrays[f"{row['case']}/time"] = np.array(time_name)
                for name, values in sets.items():
                    arrays[f"{row['case']}/{name}"] = values
                done_rows.append({'run': row['run'], 'case': row['case'], 'time': time_name, 'md5_ok': md5_ok,
                                  'bytes': size})
                total_bytes += size
            if done_rows:
                name = f"lines_{done_rows[0]['case']}_{len(done_rows)}"
                for r in done_rows:
                    r['shard'] = f"{shard[0]['run']}/{name}.npz"
                stem = staging / f"{shard[0]['run']}_{name}"
                np.savez_compressed(stem.with_suffix('.npz'), **arrays)
                with open(stem.with_suffix('.csv'), 'w', newline='') as f:    # written last: marks the shard complete
                    writer = csv.DictWriter(f, fieldnames=DONE_FIELDS)
                    writer.writeheader()
                    writer.writerows(done_rows)
            if lost is not None:
                raise DriveLost(str(lost)) from lost
            try:
                publish(staging, output)
                if error_rows:
                    append_rows(output / 'errors.csv', ['run', 'case', 'error'], error_rows)
            except OSError as error:
                if mount_lost(error):
                    raise DriveLost(str(error)) from error
                raise
            elapsed = time.time() - started
            bad = sum(not r['md5_ok'] for r in done_rows)
            print(f"{shard[0]['run']:>8s} {shard[0]['case']} ... {len(done_rows)} read, {len(error_rows)} errors, "
                  f"{bad} MD5 mismatches | {total_bytes / 1e9:.1f} GB in {elapsed / 3600:.2f} h "
                  f"({total_bytes / 1e6 / max(elapsed, 1):.0f} MB/s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--archives', required=True, help="folder with <run>/<case>.tar.zst")
    parser.add_argument('--index', required=True, help="archive_index.csv of the production")
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--shard-size', type=int, default=250)
    parser.add_argument('--shard-bytes', type=float, default=20e9)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--staging', default=None)
    args = parser.parse_args()
    run(args.archives, args.index, args.output, args.workers, args.shard_size, args.shard_bytes, args.limit,
        args.staging)


if __name__ == "__main__":
    sys.exit(main())
