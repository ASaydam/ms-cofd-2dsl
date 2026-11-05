"""
Brings the per-run results of the data root in line with the code before the re-freeze, without CFD:

    python -m tools.patch_results --wake <wake_band.csv>               data root: MSCOFD_DATA_ROOT
    python -m tools.patch_results --wake <wake_band.csv> --dry-run     count only, change nothing

1. results/<run>/<case>.npz: channel 3 (wall_distance) is set to extraction/build_arrays.py wall_distance,
   min(j, ny - 1 - j). The other channels and the exported time stay as they are.
2. results/<run>/<case>_attributes.json: wake_min_u_x is set to the smallest u_x of the wake line within the
   obstacle's y extent (tools/openfoam_attributes.py), from <wake_band.csv>: one row per run and case with the columns
   run, case, line_min (minimum of the whole line), metadata (the value in the file), band_min (the new value), made
   by tools/wake_band.py from the line samples of the run archives (tools/extract_line_samples.py). New values are
   rounded to the 6 significant digits of OpenFOAM's sets. Only the runs whose minimum of the whole
   line lies outside the extent (band_min != line_min) change. A file is changed only when it holds the row's old
   value, so a wrong row is reported instead of applied; the original goes to <backup>/<run>/.

Files already patched are skipped, so the script can run again. Every file is written to a temporary file first,
which replaces the original only when complete. Prints the counts per run.
"""

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.build_arrays import wall_distance  # noqa: E402
from utils import data_root  # noqa: E402

RUNS = ['0.25x', 'base_fo', 'base_lu', '4x', '16x']


def replace_atomically(path, write):
    """write(file) into a temporary file next to path, then replace path with it"""
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'wb') as file:
        write(file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def patch_npz(path, dry_run=False):
    """:return: True when channel 3 was (or would be) rewritten"""
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    sample = arrays['sample']
    walls = wall_distance(*sample.shape[1:]).astype(sample.dtype)
    if np.array_equal(sample[3], walls):
        return False
    if not dry_run:
        sample[3] = walls
        replace_atomically(path, lambda file: np.savez_compressed(file, **arrays))
    return True


def wake_rows(path):
    """{(run, case): (old value, new value)} of the runs whose wake velocity changes. The line samples behind the csv
    are float32 copies of OpenFOAM's ASCII sets (writePrecision 6), so the new value is rounded to 6 significant
    digits: the value tools/openfoam_attributes.py reads from the archive (tools/wake_band.py already rounds)."""
    with open(path, newline='') as file:
        return {(r['run'], r['case']): (float(r['metadata']), float(f"{float(r['band_min']):.6g}"))
                for r in csv.DictReader(file) if float(r['band_min']) != float(r['line_min'])}


def patch_attributes(path, old, new, backup, dry_run=False):
    """:return: 'patched', 'done' (already the new value) or 'mismatch' (neither the old nor the new value)"""
    values = json.loads(path.read_text())
    current = values.get('wake_min_u_x')
    if current == new:
        return 'done'
    if current != old:
        return 'mismatch'
    if not dry_run:
        backup.mkdir(parents=True, exist_ok=True)
        if not (backup / path.name).exists():
            shutil.copy2(path, backup / path.name)
        values['wake_min_u_x'] = new
        text = json.dumps(values, indent=1, default=float)
        replace_atomically(path, lambda file: file.write(text.encode()))
    return 'patched'


def run(root, wake_csv, backup, dry_run=False):
    """:return: {run: counts}"""
    results = Path(root) / 'results'
    changes = wake_rows(wake_csv)
    counts = {}
    for run in RUNS:
        c = counts[run] = {'npz': 0, 'npz_rewritten': 0, 'wake_patched': 0, 'wake_done': 0, 'wake_mismatch': 0,
                           'wake_missing': 0}
        for path in sorted((results / run).glob('*.npz')):
            c['npz'] += 1
            c['npz_rewritten'] += patch_npz(path, dry_run)
        for (row_run, case), (old, new) in sorted(changes.items()):
            if row_run != run:
                continue
            path = results / run / f"{case}_attributes.json"
            if not path.exists():
                c['wake_missing'] += 1
                continue
            outcome = patch_attributes(path, old, new, Path(backup) / run, dry_run)
            c[f'wake_{outcome}'] += 1
            if outcome == 'mismatch':
                print(f"{run} {case}: wake_min_u_x is neither {old} nor {new}, left as it is", flush=True)
        print(f"{run}: {c}", flush=True)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--wake', required=True, help="wake_band.csv: run, case, line_min, metadata, band_min")
    parser.add_argument('--backup', help="folder for the original attribute files, default "
                                         "<data root>/audit/patch-results-<date>/attributes")
    parser.add_argument('--dry-run', action='store_true', help="count only, change nothing")
    args = parser.parse_args()
    root = data_root()
    backup = args.backup or root / 'audit' / f"patch-results-{date.today().isoformat()}" / 'attributes'
    counts = run(root, args.wake, backup, args.dry_run)
    if any(c['wake_mismatch'] or c['wake_missing'] for c in counts.values()):
        sys.exit("some attribute files were left as they are (see above)")


if __name__ == "__main__":
    main()
