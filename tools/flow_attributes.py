"""
Compares flow attributes of a coarse and a fine run of every case (default base_lu and 16x), to see which quantities
the coarse mesh gets right and which it gets wrong.

    python -m tools.flow_attributes                          # results in the data root (MSCOFD_DATA_ROOT)
    python -m tools.flow_attributes --coarse base_fo --fine 16x

Attributes computed from the arrays (on each resolution's own grid, mm and m/s):

    recirculation_length_mm   furthest reverse flow (u_x < 0) connected to the obstacle behind it, from its
                              downstream end (a separate thin reverse-flow layer along a channel wall does not count)
    recirculation_area_mm2    fluid area with u_x < 0 connected to the obstacle
    flow_split_upper          share of the flow passing above the obstacle (at its centroid)
    dp_inlet_outlet           mean pressure on the inlet minus the outlet column
    dissipation               nu * integral of |grad u|^2 over the fluid (first 5 mm left out: inlet corners)
    peak_vorticity            largest |vorticity| (shear layers leaving the obstacle), first 5 mm left out
    mean_vorticity            |vorticity| averaged over the fluid area, first 5 mm left out
    wake_min_u_x              smallest u_x 20 mm behind the obstacle (wake velocity deficit)

The same attributes computed with OpenFOAM on the simulation mesh (results/<run>/<case>_attributes.json, see
tools/openfoam_attributes.py) are the reference values; they also give the forces on the obstacle (drag, pressure
and friction drag, lift), the peak wall shear and the front (stagnation) and base pressure. The summary shows

1. coarse vs fine: (coarse - fine) / |fine| in %, OpenFOAM values where available (lift relative to the drag, lift
   is close to zero for nearly symmetric obstacles), plus the field differences on the fine grid (coarse
   interpolated): relative L2 of u_x, p and vorticity,
2. arrays vs OpenFOAM: (array - OpenFOAM) / |OpenFOAM| in %, how much of an attribute is lost by computing it
   from the arrays (what a network predicts) instead of on the simulation mesh.

Writes results/flow_attributes_<coarse>_<fine>.csv and prints a summary.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import data_root, load_config, write_csv  # noqa: E402
from dataset.results import load_attributes, pairs  # noqa: E402

NU = 1e-4  # m^2/s
ARRAY_ATTRIBUTES = ['recirculation_length_mm', 'recirculation_area_mm2', 'flow_split_upper', 'dp_inlet_outlet',
                    'dissipation', 'peak_vorticity', 'mean_vorticity', 'wake_min_u_x']
OPENFOAM_ONLY = ['Fx', 'Fx_pressure', 'Fx_viscous', 'Fy', 'Fy_pressure', 'Fy_viscous', 'max_wall_shear',
                 'mean_wall_shear', 'front_pressure', 'base_pressure', 'dp_sections']


def spacing(nx, ny, domain):
    """grid spacing in mm (the grid points include both ends of the domain)"""
    return domain['length'] / (nx - 1), domain['height'] / (ny - 1)


def attributes(sample, domain):
    """flow attributes of one (6, nx, ny) sample, domain in mm (length, height)"""
    from scipy import ndimage

    ux, uy, p, cell_type = sample[0], sample[1], sample[2], sample[4]
    nx, ny = ux.shape
    hx, hy = spacing(nx, ny, domain)
    x = np.arange(nx)[:, None] * hx * np.ones((1, ny))
    y = np.ones((nx, 1)) * np.arange(ny)[None, :] * hy

    obstacle = np.zeros(ux.shape, bool)
    obstacle[1:-1, 1:-1] = cell_type[1:-1, 1:-1] == 1
    fluid = cell_type != 1
    x_end = x[obstacle].max()

    labels, _ = ndimage.label((ux < 0) & fluid)
    touching = labels[ndimage.binary_dilation(obstacle) & (labels > 0)]
    reverse = np.isin(labels, touching)
    behind = reverse & (x > x_end)
    result = {'recirculation_length_mm': float(x[behind].max() - x_end) if behind.any() else 0.0,
              'recirculation_area_mm2': float(reverse.sum() * hx * hy)}

    # share of the flow above the obstacle, in the column of its centroid
    i = int(round(x[obstacle].mean() / hx))
    top = y[i][obstacle[i]].max()
    column = fluid[i]
    result['flow_split_upper'] = float(ux[i][column & (y[i] > top)].sum() / ux[i][column].sum())
    result['dp_inlet_outlet'] = float(p[0, 1:-1].mean() - p[-1, 1:-1].mean())

    # gradients in 1/s (spacing in m), without the inlet corners
    dx, dy = hx / 1000, hy / 1000
    gradients = [np.gradient(u, dx, axis=0) for u in (ux, uy)] + [np.gradient(u, dy, axis=1) for u in (ux, uy)]
    region = fluid & (x > 5)
    result['dissipation'] = float(NU * sum((g[region] ** 2).sum() for g in gradients) * dx * dy)
    vorticity = gradients[1] - gradients[2]  # d(u_y)/dx - d(u_x)/dy
    result['peak_vorticity'] = float(np.abs(vorticity[region]).max())
    result['mean_vorticity'] = float(np.abs(vorticity[region]).mean())

    i_wake = min(int(round((x_end + 20) / hx)), nx - 1)
    result['wake_min_u_x'] = float(ux[i_wake][fluid[i_wake]].min())
    return result, vorticity


def upsample(field, nx, ny):
    """coarse field at the fine grid points (bilinear)"""
    bx, by = field.shape
    xi = np.arange(nx) * (bx - 1) / (nx - 1)
    yi = np.arange(ny) * (by - 1) / (ny - 1)
    x0, y0 = np.minimum(np.floor(xi).astype(int), bx - 2), np.minimum(np.floor(yi).astype(int), by - 2)
    wx, wy = (xi - x0)[:, None], (yi - y0)[None, :]
    return ((1 - wx) * (1 - wy) * field[np.ix_(x0, y0)] + wx * (1 - wy) * field[np.ix_(x0 + 1, y0)] +
            (1 - wx) * wy * field[np.ix_(x0, y0 + 1)] + wx * wy * field[np.ix_(x0 + 1, y0 + 1)])


def relative_difference(coarse, fine, scale=None):
    """(coarse - fine) / |scale| in %, scale = fine by default"""
    scale = abs(fine) if scale is None else abs(scale)
    return 100 * (coarse - fine) / scale if scale > 1e-30 else float('nan')


def summary(label, values):
    """prints mean, mean |.| and max |.| of a list of (percentage, case name) and the case of the largest one"""
    diffs = np.array([value for value, _ in values], dtype=float)
    valid = ~np.isnan(diffs)
    if not valid.any():
        return
    worst = values[int(np.nanargmax(np.abs(diffs)))][1]
    print(f"{label:34s} {np.mean(diffs[valid]):8.1f} {np.mean(np.abs(diffs[valid])):9.1f} "
          f"{np.max(np.abs(diffs[valid])):8.1f}  {worst}")


def main():
    parser = argparse.ArgumentParser(description="Compare flow attributes of a coarse and a fine run")
    parser.add_argument("--coarse", default="base_lu", help="coarse run (base_lu, base_fo or 4x)")
    parser.add_argument("--fine", default="16x", help="fine run (16x or 4x)")
    parser.add_argument("--output", help="folder of the csv file, default <data root>/results (a frozen data root is "
                                         "only read when an output folder outside it is given)")
    args = parser.parse_args()

    root, domain = data_root(), load_config()['domain']
    runs = (args.coarse, args.fine)

    def openfoam_value(case_name, run, name):
        value = load_attributes(root, run, case_name).get(name, '')
        return float(value) if value not in ('', None) else float('nan')

    rows = []
    for case_name, coarse, fine in pairs(root, args.coarse, args.fine):
        row = {'case_name': case_name, 'size': case_name.split('_')[0],
               'geometry': case_name.split('_', 1)[1].rsplit('_', 1)[0]}

        array_values = {}
        (array_values[args.coarse], vorticity_coarse), (array_values[args.fine], vorticity_fine) = (
            attributes(coarse, domain), attributes(fine, domain))

        for name in ARRAY_ATTRIBUTES + OPENFOAM_ONLY:
            values = {}
            for run in runs:
                reference = openfoam_value(case_name, run, name)
                if name in ARRAY_ATTRIBUTES:
                    array_value = array_values[run][name]
                    row[f'{name}_{run}_array'] = array_value
                    row[f'{name}_{run}_array_vs_openfoam_%'] = relative_difference(array_value, reference)
                    # OpenFOAM value when there is one, the array value otherwise
                    values[run] = reference if not np.isnan(reference) else array_value
                else:
                    values[run] = reference
                row[f'{name}_{run}'] = values[run]
            # lift is close to zero for nearly symmetric obstacles, it is compared relative to the drag
            scale = values[args.fine] if not name.startswith('Fy') else openfoam_value(case_name, args.fine, 'Fx')
            row[f'{name}_diff_%'] = relative_difference(values[args.coarse], values[args.fine], scale)

        # field differences on the fine grid, fluid points without the inlet corners
        nx, ny = fine.shape[1:]
        region = (fine[4] != 1) & (np.arange(nx)[:, None] * domain['length'] / (nx - 1) > 5)
        for name, coarse_field, fine_field in (('u_x', coarse[0], fine[0]), ('p', coarse[2], fine[2]),
                                               ('vorticity', vorticity_coarse, vorticity_fine)):
            difference = upsample(coarse_field, nx, ny) - fine_field
            row[f'field_rel_l2_{name}_%'] = 100 * float(np.linalg.norm(difference[region]) /
                                                        np.linalg.norm(fine_field[region]))
        rows.append(row)

    if not rows:
        sys.exit(f"no case has results of both {args.coarse} and {args.fine} in {root / 'results'}")
    output = (Path(args.output) if args.output else root / 'results') / f"flow_attributes_{args.coarse}_{args.fine}.csv"
    write_csv(rows, output, list(rows[0].keys()))

    print(f"{len(rows)} cases. 1) {args.coarse} vs {args.fine}, (coarse - fine) / |fine| in % "
          f"(OpenFOAM values where available, lift relative to drag)")
    print(f"{'attribute':34s} {'mean':>8s} {'mean |.|':>9s} {'max |.|':>8s}  worst case")
    for name in OPENFOAM_ONLY + ARRAY_ATTRIBUTES:
        summary(name, [(row[f'{name}_diff_%'], row['case_name']) for row in rows])
    for name in ('u_x', 'p', 'vorticity'):
        summary(f"field L2 {name} (arrays)", [(row[f'field_rel_l2_{name}_%'], row['case_name']) for row in rows])

    print("\n2) arrays vs OpenFOAM, (array - OpenFOAM) / |OpenFOAM| in %")
    print(f"{'attribute':34s} {'mean':>8s} {'mean |.|':>9s} {'max |.|':>8s}  worst case")
    for name in ARRAY_ATTRIBUTES:
        for run in runs:
            summary(f"{name} ({run})", [(row[f'{name}_{run}_array_vs_openfoam_%'], row['case_name']) for row in rows])
    print(f"Attributes written to {output}")


if __name__ == "__main__":
    main()
