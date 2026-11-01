"""
How much an attribute changes when it is computed from the arrays instead of on the simulation mesh, per level
(docs/validation.md; Table 10 of the dataset paper):

    python validation/array_vs_openfoam.py <folder of flow_attributes_<run>_16x.csv> <metadata.parquet>

The csv files are those of tools/flow_attributes.py --coarse <run> --fine 16x --output <folder>, for the runs 0.25x,
base_fo, base_lu and 4x; every file also has the 16x values. The metadata of the dataset gives the obstacle heights.
Prints, per attribute and level, the median and 90th percentile over the cases of |array - OpenFOAM| / scale in %,
and the median signed value. The scales are those of the level errors (tools/uncertainty.py scale_of with the
obstacle height): the OpenFOAM value of the same run, except for the recirculation length and area, relative to at
least the obstacle height and its square, and the wake velocity, relative to the inlet velocity. So no case is left
out where the OpenFOAM value is 0.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.uncertainty import scale_of  # noqa: E402

ATTRIBUTES = ['recirculation_length_mm', 'recirculation_area_mm2', 'flow_split_upper', 'dp_inlet_outlet', 'dissipation',
              'mean_vorticity', 'peak_vorticity', 'wake_min_u_x']
LEVELS = [('0.25x', '2 mm'), ('base_fo', '1 mm FO'), ('base_lu', '1 mm'), ('4x', '0.5 mm'), ('16x', '0.25 mm')]


def read_columns(path):
    """the columns of a csv file, as lists of strings"""
    with open(path, newline='') as file:
        rows = list(csv.DictReader(file))
    return {name: [row[name] for row in rows] for name in rows[0]}


def floats(values):
    return np.array([float(v) if v not in ('', 'nan') else np.nan for v in values])


def array_vs_openfoam(folder, metadata):
    """
    :param metadata: metadata.parquet of the dataset (the obstacle heights)
    :return: {attribute: {run: (median |d|, p90 |d|, median d)}} with d = (array - OpenFOAM) / scale in %
    """
    import pyarrow.parquet as pq

    table = pq.read_table(metadata, columns=['case_name', 'bbox_height']).to_pydict()
    height_of = dict(zip(table['case_name'], table['bbox_height']))
    files = {run: Path(folder) / f"flow_attributes_{run}_16x.csv" for run, _ in LEVELS if run != '16x'}
    files['16x'] = files['4x']                       # the 16x values are the same in every file (tools/uncertainty.py)
    columns = {run: read_columns(path) for run, path in files.items()}
    results = {}
    for name in ATTRIBUTES:
        results[name] = {}
        for run, _ in LEVELS:
            data = columns[run]
            mesh, array = floats(data[f'{name}_{run}']), floats(data[f'{name}_{run}_array'])
            height = np.array([height_of[case] for case in data['case_name']], dtype=float)
            scale = scale_of(name, mesh, np.nan, height)
            with np.errstate(divide='ignore', invalid='ignore'):
                d = np.where(scale > 0, 100 * (array - mesh) / scale, np.nan)
            results[name][run] = (np.nanmedian(np.abs(d)), np.nanquantile(np.abs(d), 0.9), np.nanmedian(d))
    return results


if __name__ == "__main__":
    print("|array - OpenFOAM| / scale over the cases, %: median (90th percentile) [median signed]")
    print(f"  {'attribute':24s} " + ' '.join(f"{label:>24s}" for _, label in LEVELS))
    for name, by_run in array_vs_openfoam(sys.argv[1], sys.argv[2]).items():
        cells = [f"{m:.2f} ({p:.2f}) [{s:+.2f}]" for m, p, s in (by_run[run] for run, _ in LEVELS)]
        print(f"  {name:24s} " + ' '.join(f"{c:>24s}" for c in cells))
