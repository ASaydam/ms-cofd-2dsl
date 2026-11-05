"""
The wake velocity within the obstacle's y extent for every run of the dataset, from the line samples of the run
archives (tools/extract_line_samples.py), without solving or processing the runs again. The output is the input of
tools/patch_results.py:

    python -m tools.wake_band --lines <line-samples folder> --output wake_band.csv     data root: MSCOFD_DATA_ROOT

In the production, wake_min_u_x was the minimum of u_x over the whole wake line, wall to wall, so it was the no-slip
zero whenever no reverse flow reached the line. tools/openfoam_attributes.py now takes the minimum within the
obstacle's y extent. This script gives that value for the archived runs: the minimum over the points of the line with
gap_bottom <= y <= height - gap_top (the obstacle's bounding box in the manifest). The attribute code takes the extent
from the obstacle's surface faces instead, which can differ by a line point at either end. The minimum lies 3.1 mm or
more inside the extent in every run, so both give the same value.

The line samples are float32 copies of OpenFOAM's ASCII sets, which every case template writes with 6 significant
digits (writePrecision 6). line_min and band_min are therefore rounded to 6 significant digits, which recovers the
values tools/openfoam_attributes.py reads from the archive exactly.

Only the cases of the dataset are written (the dropped cases are not in the release). One row per run and case:
    run, case
    line_min       minimum over the whole line: the production value, so it must equal metadata
    metadata       wake_min_u_x in results/<run>/<case>_attributes.json
    band_min       minimum within the obstacle's y extent (the new value)
    band_y_mm      y of band_min (mm)
    band_points    number of line points within the extent
    at_edge        band_min is at the first or the last point within the extent
Prints the number of runs written, of excluded cases skipped, of runs whose value changes, of runs with band_min at
the edge of the extent and of runs whose line_min differs from the attributes file (both expected to be 0).
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hpc import ledger, topup  # noqa: E402

FIELDS = ['run', 'case', 'line_min', 'metadata', 'band_min', 'band_y_mm', 'band_points', 'at_edge']


def six_digits(value):
    """a float32 line sample as the double that OpenFOAM's ASCII set holds (writePrecision 6)"""
    return float(f"{float(value):.6g}")


def wake_row(run, case, wake, gap_bottom, gap_top, height, value):
    """
    one output row
    :param wake: the wake line of the run as extract_line_samples stores it, rows (y in m, p, u_x, u_y)
    :param value: wake_min_u_x of the run's attributes file
    """
    # y is a float32 copy too: rounded to 0.1 um it is the set's value again (the points are 0.05 mm apart), so a
    # point on the edge of the extent stays in it
    y, ux = np.round(wake[:, 0].astype(np.float64) * 1000, 4), wake[:, 2].astype(np.float64)
    band = np.flatnonzero((y >= gap_bottom) & (y <= height - gap_top))
    i = band[np.argmin(ux[band])]
    return {'run': run, 'case': case, 'line_min': six_digits(ux.min()), 'metadata': value,
            'band_min': six_digits(ux[i]), 'band_y_mm': round(float(y[i]), 4), 'band_points': len(band),
            'at_edge': bool(i in (band[0], band[-1]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lines', required=True, help="folder of extract_line_samples (shards and done.csv)")
    parser.add_argument('--output', required=True, help="the csv to write")
    args = parser.parse_args()

    root = ledger.data_root()
    _, dataset = topup.load_settings(root)
    runs = list(dataset['runs'])
    cases, dropped, _ = topup.dataset_cases(root, dataset, ledger.read_states(runs, root), topup.cases_file_of(root))
    members = set(cases) - set(dropped)
    manifest = {row['case_name']: row for row in ledger.read_manifest(root / 'geometry' / 'manifest.csv')}
    height = float(dataset['domain']['height'])

    lines = Path(args.lines)
    shards = defaultdict(list)
    with open(lines / 'done.csv', newline='') as file:
        for row in csv.DictReader(file):
            shards[row['shard']].append((row['run'], row['case']))
    rows, skipped = [], 0
    for k, (shard, shard_cases) in enumerate(sorted(shards.items())):
        with np.load(lines / shard) as data:
            for run, case in shard_cases:
                if case not in members:
                    skipped += 1
                    continue
                value = json.loads((root / 'results' / run / f"{case}_attributes.json").read_text())['wake_min_u_x']
                gaps = manifest[case]
                rows.append(wake_row(run, case, data[f"{case}/wakeLine"], float(gaps['gap_bottom']),
                                     float(gaps['gap_top']), height, value))
        print(f"{k + 1}/{len(shards)} {shard}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} runs written, {skipped} runs of excluded cases skipped, "
          f"{sum(r['band_min'] != r['line_min'] for r in rows)} change, "
          f"{sum(r['at_edge'] for r in rows)} at the edge of the extent, "
          f"{sum(r['line_min'] != r['metadata'] for r in rows)} with line_min != the attributes file")


if __name__ == "__main__":
    main()
