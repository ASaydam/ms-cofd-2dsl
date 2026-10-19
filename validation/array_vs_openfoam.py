"""
How much an attribute changes when it is computed from the arrays instead of on the simulation mesh, per level
(docs/validation.md; Table 10 of the dataset paper):

    python validation/array_vs_openfoam.py <folder of flow_attributes_<run>_16x.csv>

The csv files are those of tools/flow_attributes.py --coarse <run> --fine 16x --output <folder>, for the runs 0.25x,
base_fo, base_lu and 4x; every file also has the 16x values. Prints, per attribute and level, the median and 90th
percentile over the cases of |array - OpenFOAM| / |OpenFOAM| in %, and the median signed value.
"""

import csv
import sys
from pathlib import Path

import numpy as np

ATTRIBUTES = ['recirculation_length_mm', 'recirculation_area_mm2', 'flow_split_upper', 'dp_inlet_outlet', 'dissipation',
              'mean_vorticity', 'peak_vorticity', 'wake_min_u_x']
LEVELS = [('0.25x', '2 mm'), ('base_fo', '1 mm FO'), ('base_lu', '1 mm'), ('4x', '0.5 mm'), ('16x', '0.25 mm')]


def column(path, name):
    with open(path, newline='') as file:
        return np.array([float(row[name]) if row[name] not in ('', 'nan') else np.nan for row in csv.DictReader(file)])


def array_vs_openfoam(folder):
    """:return: {attribute: {run: (median |d|, p90 |d|, median d)}} with d = (array - OpenFOAM) / |OpenFOAM| in %"""
    files = {run: Path(folder) / f"flow_attributes_{run}_16x.csv" for run, _ in LEVELS if run != '16x'}
    files['16x'] = files['4x']                       # the 16x values are the same in every file (tools/uncertainty.py)
    results = {}
    for name in ATTRIBUTES:
        results[name] = {}
        for run, _ in LEVELS:
            d = column(files[run], f'{name}_{run}_array_vs_openfoam_%')
            results[name][run] = (np.nanmedian(np.abs(d)), np.nanquantile(np.abs(d), 0.9), np.nanmedian(d))
    return results


if __name__ == "__main__":
    print("|array - OpenFOAM| / |OpenFOAM| over the cases, %: median (90th percentile) [median signed]")
    print(f"  {'attribute':24s} " + ' '.join(f"{label:>24s}" for _, label in LEVELS))
    for name, by_run in array_vs_openfoam(sys.argv[1]).items():
        cells = [f"{m:.2f} ({p:.2f}) [{s:+.2f}]" for m, p, s in (by_run[run] for run, _ in LEVELS)]
        print(f"  {name:24s} " + ' '.join(f"{c:>24s}" for c in cells))
