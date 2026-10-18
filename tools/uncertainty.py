"""
Numerical uncertainty of every attribute and case from the four second-order levels (2, 1, 0.5 and 0.25 mm), with the
least-squares procedure of Eça & Hoekstra (2014), "A procedure for the estimation of the numerical uncertainty of CFD
calculations based on grid refinement studies", J. Comput. Phys. 262:104-130. The procedure is made for data that
scatter and are not in the asymptotic range, as the four independent unstructured meshes of the dataset are
(the three-level GCI of tools/mesh_study.py is undefined in a third or more of the cases, docs/validation.md).

    python -m tools.uncertainty <dataset folder> <output folder>
    python -m tools.uncertainty <dataset folder> <output folder> --arrays <folder of flow_attributes_<run>_16x.csv>

Reads mesh_study.parquet (the values of every level) and metadata.parquet (the cells of every level) of a dataset
folder (dataset/build_dataset.py), writes label_uncertainty.parquet / .csv to the output folder and prints a summary.
Its values are the OpenFOAM values on the simulation mesh (source "openfoam").

With --arrays, the values are instead the attributes computed from the arrays of every level on its own grid
(tools/flow_attributes.py --coarse <run> --fine 16x --output <folder>, for the runs 0.25x, base_lu and 4x), the
quantities a network's prediction is scored on. The resampling to the grid loses a different amount at each level, so
these converge differently from the OpenFOAM values. Writes label_uncertainty_arrays.parquet / .csv (source "array").

For every case and attribute (phi_1 = 0.25 mm, ..., phi_4 = 2 mm; h_i = sqrt(N_1 / N_i), so h_1 = 1):

    fits    phi = phi_0 + a h^p (p from the data), phi_0 + a h, phi_0 + a h^2, phi_0 + a h + b h^2; each least squares,
            unweighted and with weights proportional to 1 / h; the one of the two with the smaller standard
            deviation sigma is used
    method  power      0.5 <= p < 2.1: error estimate e = |phi_1 - phi_0| of the power fit, factor of safety 1.25
            low_order  0 < p < 0.5: e from the one of the fits with p = 1, 2 and 1 + 2 with the smallest sigma, 3
            high_order p >= 2.1: e from the one of the fits with p = 2 and 1 + 2 with the smallest sigma, 3
            anomalous  p <= 0 (no convergence): U = 3 max |phi_i - phi_j|
            equal      all levels equal: U = 0
    U       Fs e + sigma + |phi_1 - fit(h_1)| when sigma < dphi = (max phi - min phi) / 3, else
            3 sigma / dphi (e + sigma + |phi_1 - fit(h_1)|)
    U_%     U on the scale of the attribute: relative to |phi_1|; lift and the parts of the drag relative to the drag
            (as in validation/audit_dataset.py); wake velocity relative to the inlet velocity, front and base pressure
            to the dynamic pressure
    U1_%    the simple fallback 3 |phi_1 - phi_2| (factor of safety 3, first order), on the same scale

Needs numpy and pandas with pyarrow.
"""

import sys
from pathlib import Path

import numpy as np

LEVELS = ['16x', '4x', 'base', '0.25x']              # fine to coarse, all second order
CELLS = ['16x.cells', '4x.cells', 'base_lu.cells', '0.25x.cells']
RUNS = {'4x': '4x', 'base': 'base_lu', '0.25x': '0.25x'}                         # level of LEVELS: run of its csv
ARRAY_ATTRIBUTES = ['recirculation_length_mm', 'recirculation_area_mm2', 'flow_split_upper', 'dp_inlet_outlet',
                    'dissipation', 'peak_vorticity', 'mean_vorticity', 'wake_min_u_x']   # tools/flow_attributes.py
INLET_VELOCITY = 0.1                                 # m/s, the case templates (validation/audit_dataset.py reads it)
P_GRID = np.round(np.concatenate([np.arange(-3.0, -0.02, 0.01), np.arange(0.03, 8.0001, 0.01)]), 2)


def weights_of(h):
    """weights proportional to 1 / h, normalised to a sum of 1 (one row per case)"""
    w = 1 / h
    return w / w.sum(axis=1, keepdims=True)


def linear_fit(x, phi, w):
    """
    weighted least squares phi = c + a x of every row (x, phi, w: (n, m))
    :return: c, a, weighted sum of the squared residuals
    """
    sw, sx, sy = w.sum(1), (w * x).sum(1), (w * phi).sum(1)
    sxx, sxy = (w * x * x).sum(1), (w * x * phi).sum(1)
    det = sw * sxx - sx * sx
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(det != 0, (sw * sxy - sx * sy) / det, 0.0)
    c = (sy - a * sx) / sw
    residual = phi - (c[:, None] + a[:, None] * x)
    return c, a, (w * residual ** 2).sum(1)


def sigma_of(ssr, w, parameters):
    """standard deviation of a fit from its weighted sum of squared residuals (weights summing to 1)"""
    m = w.shape[1]
    return np.sqrt(np.maximum(ssr, 0) * m / (m - parameters))


def power_fit(h, phi, w):
    """phi = phi_0 + a h^p with p on P_GRID (step 0.01): phi_0, a, p, sigma of every row"""
    best = None
    for p in P_GRID:
        c, a, ssr = linear_fit(h ** p, phi, w)
        if best is None:
            best = [c, a, np.full(len(c), p), ssr]
            continue
        better = ssr < best[3]
        for k, value in enumerate((c, a, np.full(len(c), p), ssr)):
            best[k] = np.where(better, value, best[k])
    c, a, p, ssr = best
    return c, a, p, sigma_of(ssr, w, 3)


def fixed_fits(h, phi, w):
    """the fits with p = 1, p = 2 and the two terms h + h^2: {name: (phi_0, fit at h_1, sigma)}"""
    fits = {}
    for name, p in (('p1', 1.0), ('p2', 2.0)):
        c, a, ssr = linear_fit(h ** p, phi, w)
        fits[name] = (c, c + a * h[:, 0] ** p, sigma_of(ssr, w, 2))
    basis = np.stack([np.ones_like(h), h, h ** 2], axis=2)                       # (n, m, 3)
    normal = np.einsum('nm,nmi,nmj->nij', w, basis, basis)
    right = np.einsum('nm,nmi,nm->ni', w, basis, phi)
    coefficients = np.linalg.solve(normal + 1e-300 * np.eye(3), right[..., None])[..., 0]
    fitted = np.einsum('nmi,ni->nm', basis, coefficients)
    fits['p12'] = (coefficients[:, 0], fitted[:, 0], sigma_of((w * (phi - fitted) ** 2).sum(1), w, 3))
    return fits


def uncertainty(h, phi):
    """
    Eça & Hoekstra (2014) uncertainty of every row
    :param h: (n, 4) step sizes, fine to coarse (h[:, 0] = 1)
    :param phi: (n, 4) values, fine to coarse
    :return: dict of arrays: U, p, method, phi_0
    """
    n, m = phi.shape
    choices = []
    for w in (np.full_like(h, 1 / m), weights_of(h)):          # unweighted, weighted
        c, a, p, sigma = power_fit(h, phi, w)
        choices.append((sigma, c, c + a * h[:, 0] ** p, p, fixed_fits(h, phi, w)))
    weighted = choices[1][0] < choices[0][0]

    def pick(k):
        return np.where(weighted, choices[1][k], choices[0][k])

    sigma, phi0, fit1, p = pick(0), pick(1), pick(2), pick(3)
    fixed = {name: tuple(np.where(weighted, choices[1][4][name][k], choices[0][4][name][k]) for k in range(3))
             for name in ('p1', 'p2', 'p12')}

    def best_of(names):
        sigmas = np.stack([fixed[name][2] for name in names])
        index = sigmas.argmin(0)
        take = lambda k: np.stack([fixed[name][k] for name in names])[index, np.arange(n)]   # noqa: E731
        return take(0), take(1), take(2)

    low, high = best_of(['p1', 'p2', 'p12']), best_of(['p2', 'p12'])
    spread = phi.max(1) - phi.min(1)
    dphi = spread / (m - 1)
    method = np.where(spread == 0, 'equal', np.where(p <= 0, 'anomalous', np.where(
        p < 0.5, 'low_order', np.where(p < 2.1, 'power', 'high_order'))))
    for name, (c, f, s) in (('low_order', low), ('high_order', high)):
        mask = method == name
        phi0, fit1, sigma = np.where(mask, c, phi0), np.where(mask, f, fit1), np.where(mask, s, sigma)
    safety = np.where(method == 'power', 1.25, 3.0)
    error = np.abs(phi[:, 0] - phi0)
    misfit = np.abs(phi[:, 0] - fit1)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.where(sigma < dphi, safety * error + sigma + misfit, 3 * sigma / dphi * (error + sigma + misfit))
    u = np.where(method == 'anomalous', 3 * spread, u)
    u = np.where(method == 'equal', 0.0, u)
    return {'U': u, 'p': p, 'method': method, 'phi_0': phi0}


def scale_of(name, value, drag):
    """the scale of an attribute (validation/audit_dataset.py level_difference)"""
    if name.startswith('Fy') or name in ('Fx_pressure', 'Fx_viscous'):
        return np.abs(drag)
    if name == 'wake_min_u_x':
        return np.full(len(value), INLET_VELOCITY)
    if name in ('front_pressure', 'base_pressure'):
        return np.full(len(value), 0.5 * INLET_VELOCITY ** 2)
    return np.abs(value)


def array_study(folder):
    """
    the array attributes of every level in the layout of mesh_study.parquet (case_name, attribute, 16x, 4x, base,
    0.25x), from the csv files of tools/flow_attributes.py (each has the coarse run and 16x)
    """
    import pandas as pd

    tables = {level: pd.read_csv(Path(folder) / f"flow_attributes_{run}_16x.csv").set_index('case_name')
              for level, run in RUNS.items()}
    cases = tables['4x'].index
    frames = []
    for name in ARRAY_ATTRIBUTES:
        fine = [tables[level][f'{name}_16x_array'].reindex(cases) for level in RUNS]
        if any(not np.allclose(f, fine[0], equal_nan=True) for f in fine[1:]):
            raise ValueError(f"{name}: the 16x array values differ between the csv files")
        frame = pd.DataFrame({'case_name': cases, 'attribute': name, '16x': fine[0].to_numpy()})
        for level, run in RUNS.items():
            frame[level] = tables[level][f'{name}_{run}_array'].reindex(cases).to_numpy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Numerical uncertainty of every attribute and case")
    parser.add_argument('dataset')
    parser.add_argument('output')
    parser.add_argument('--arrays', help="folder of the flow_attributes_<run>_16x.csv files: the array attributes")
    args = parser.parse_args()
    folder, output = Path(args.dataset), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    study = array_study(args.arrays) if args.arrays else pd.read_parquet(folder / 'mesh_study.parquet')
    cells = pd.read_parquet(folder / 'metadata.parquet', columns=['case_name'] + CELLS).set_index('case_name')
    study = study.join(cells, on='case_name')
    if study[CELLS].isna().any().any():
        sys.exit("cases without the cells of every level in metadata.parquet")
    h = np.sqrt(study[CELLS[0]].to_numpy()[:, None] / study[CELLS].to_numpy())
    phi = study[LEVELS].to_numpy(dtype=float)
    result = uncertainty(h, phi)
    if (study.attribute == 'Fx').any():
        drag = study[['case_name']].join(study[study.attribute == 'Fx'].set_index('case_name')['16x'].rename('drag'),
                                         on='case_name')['drag'].to_numpy()
    else:                                       # array attributes: none is scaled by the drag
        drag = np.full(len(study), np.nan)

    table = study[['case_name', 'attribute'] + LEVELS].copy()
    table.insert(2, 'source', 'array' if args.arrays else 'openfoam')
    table['U'], table['p'], table['method'], table['phi_0'] = result['U'], result['p'], result['method'], result['phi_0']
    scale = np.empty(len(table))
    for name in table['attribute'].unique():
        mask = (table['attribute'] == name).to_numpy()
        scale[mask] = scale_of(name, phi[mask, 0], drag[mask])
    with np.errstate(divide='ignore', invalid='ignore'):
        table['U_%'] = 100 * table['U'] / scale
        table['U1_%'] = 100 * 3 * np.abs(phi[:, 0] - phi[:, 1]) / scale
    name = 'label_uncertainty_arrays' if args.arrays else 'label_uncertainty'
    table.to_parquet(output / f'{name}.parquet', index=False)
    table.to_csv(output / f'{name}.csv', index=False)

    summary = table.groupby('attribute').agg(U_median=('U_%', 'median'), U_p95=('U_%', lambda s: s.quantile(0.95)),
                                             U1_median=('U1_%', 'median'), U1_p95=('U1_%', lambda s: s.quantile(0.95)))
    methods = table.pivot_table(index='attribute', columns='method', values='case_name', aggfunc='count',
                                fill_value=0)
    print("uncertainty of the 0.25 mm value, % on the scale of the attribute (median, 95th percentile); U1 = 3 |phi_1 - "
          "phi_2| for comparison; cases per method")
    print(summary.join(methods).round(3).to_string())
    print(f"written to {output}")


if __name__ == "__main__":
    main()
