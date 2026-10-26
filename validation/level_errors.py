"""
Error of every cheaper level against the 0.25 mm label, per attribute, next to the uncertainty of the label
(docs/validation.md; Table 9 and Fig. 7 of the dataset paper):

    python validation/level_errors.py <dataset folder> <label_uncertainty.parquet>

Reads the OpenFOAM attributes of every run from metadata.parquet and the uncertainty U of the 0.25 mm label (source
"openfoam") from the table of tools/uncertainty.py. Prints the median and 90th percentile over the cases of
|level - 0.25 mm| / scale and of U / scale, in %.

Scales: the drag parts and lift relative to the drag; wake velocity relative to the inlet velocity; base pressure
relative to the dynamic pressure; recirculation length relative to at least the obstacle height and its area to at
least the square of it; every other attribute relative to the 0.25 mm value.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.uncertainty import INLET_VELOCITY  # noqa: E402

RUNS = [('0.25x', '2 mm'), ('base_fo', '1 mm FO'), ('base_lu', '1 mm'), ('4x', '0.5 mm')]
ATTRIBUTES = [('Fx', 'drag'), ('Fx_pressure', 'pressure drag'), ('Fx_viscous', 'friction drag'), ('Fy', 'lift'),
              ('recirculation_length_mm', 'recirculation length'), ('recirculation_area_mm2', 'recirculation area'),
              ('wake_min_u_x', 'wake velocity'), ('base_pressure', 'base pressure'), ('dissipation', 'dissipation'),
              ('mean_vorticity', 'mean vorticity'), ('dp_sections', 'dp sections'),
              ('flow_split_upper', 'flow split'), ('mean_wall_shear', 'mean wall shear')]


def floats(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def scale(name, metadata):
    """the scale of an attribute's errors, per case"""
    label = np.abs(floats(metadata[f'16x.{name}']))
    height = floats(metadata['bbox_height'])
    if name.startswith('Fy') or name in ('Fx_pressure', 'Fx_viscous'):
        return np.abs(floats(metadata['16x.Fx']))
    if name == 'wake_min_u_x':
        return np.full(len(label), INLET_VELOCITY)
    if name == 'base_pressure':
        return np.full(len(label), 0.5 * INLET_VELOCITY ** 2)
    if name == 'recirculation_length_mm':
        return np.maximum(label, height)
    if name == 'recirculation_area_mm2':
        return np.maximum(label, height ** 2)
    return label


def level_errors(dataset, uncertainty_path):
    """
    :return: [(attribute, {run: (median, p90) of the error in %}, (median, p90) of U in %)]
    """
    import pyarrow.parquet as pq

    columns = ['case_name', 'bbox_height'] + [f'{run}.{name}' for run, _ in RUNS + [('16x', '')]
                                              for name, _ in ATTRIBUTES]
    metadata = pq.read_table(Path(dataset) / 'metadata.parquet', columns=columns).to_pydict()
    table = pq.read_table(uncertainty_path).to_pydict()
    sources = table.get('source', ['openfoam'] * len(table['case_name']))
    u_of = {(c, a): u for c, a, u, s in zip(table['case_name'], table['attribute'], table['U'], sources)
            if s == 'openfoam'}
    results = []
    for name, _ in ATTRIBUTES:
        s = scale(name, metadata)
        label = floats(metadata[f'16x.{name}'])
        errors = {}
        for run, _ in RUNS:
            e = 100 * np.abs(floats(metadata[f'{run}.{name}']) - label) / s
            errors[run] = (np.nanmedian(e), np.nanquantile(e, 0.9))
        u = 100 * floats([u_of.get((c, name)) for c in metadata['case_name']]) / s
        results.append((name, errors, (np.nanmedian(u), np.nanquantile(u, 0.9))))
    return results


def exceedance(dataset, uncertainty_path):
    """
    per case instead of per median: the share of cases whose own level error exceeds their own uncertainty U
    (|level - 0.25 mm| > U, both absolute, so no scale is needed). Cases with a missing value or U are left out and
    counted; U = 0 (all four second-order levels equal) counts as exceeded wherever the level differs from the label.
    :return: [(attribute, {run: (share exceeding in %, cases compared)}, cases left out)]
    """
    import pyarrow.parquet as pq

    columns = ['case_name'] + [f'{run}.{name}' for run, _ in RUNS + [('16x', '')] for name, _ in ATTRIBUTES]
    metadata = pq.read_table(Path(dataset) / 'metadata.parquet', columns=columns).to_pydict()
    table = pq.read_table(uncertainty_path).to_pydict()
    sources = table.get('source', ['openfoam'] * len(table['case_name']))
    u_of = {(c, a): u for c, a, u, s in zip(table['case_name'], table['attribute'], table['U'], sources)
            if s == 'openfoam'}
    results = []
    for name, _ in ATTRIBUTES:
        label = floats(metadata[f'16x.{name}'])
        u = floats([u_of.get((c, name)) for c in metadata['case_name']])
        shares = {}
        for run, _ in RUNS:
            error = np.abs(floats(metadata[f'{run}.{name}']) - label)
            valid = np.isfinite(error) & np.isfinite(u)
            shares[run] = (100 * np.mean(error[valid] > u[valid]), int(valid.sum()))
        results.append((name, shares, int(np.sum(~np.isfinite(u) | ~np.isfinite(label)))))
    return results


if __name__ == "__main__":
    labels = dict(ATTRIBUTES)
    print("median |level - 0.25 mm| / scale over the cases, % (90th percentile); U: uncertainty of the 0.25 mm label")
    print(f"  {'attribute':22s} " + ' '.join(f"{label:>15s}" for _, label in RUNS) + f" {'U of the label':>15s}")
    for name, errors, u in level_errors(sys.argv[1], sys.argv[2]):
        cells = [f"{m:6.2f} ({p:6.2f})" for m, p in (errors[run] for run, _ in RUNS)]
        print(f"  {labels[name]:22s} " + ' '.join(f"{c:>15s}" for c in cells) + f" {u[0]:6.2f} ({u[1]:6.2f})")
    print("\nshare of cases whose own |level - 0.25 mm| exceeds their own U, % (cases compared)")
    print(f"  {'attribute':22s} " + ' '.join(f"{label:>15s}" for _, label in RUNS))
    for name, shares, _ in exceedance(sys.argv[1], sys.argv[2]):
        cells = [f"{s:6.1f} ({n})" for s, n in (shares[run] for run, _ in RUNS)]
        print(f"  {labels[name]:22s} " + ' '.join(f"{c:>15s}" for c in cells))
