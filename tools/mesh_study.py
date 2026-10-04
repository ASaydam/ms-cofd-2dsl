"""
Mesh convergence study of the flow attributes: three meshes of every case (base 1 mm, 4x 0.5 mm, 16x 0.25 mm),
evaluated with the procedure of Celik et al. (2008), "Procedure for estimation and reporting of uncertainty due to
discretization in CFD applications", J. Fluids Eng. 130(7):

    python -m tools.mesh_study pilot/openfoam_attributes.csv

The input is the openfoam_attributes.csv of tools/openfoam_attributes.py collected with --resolutions base 4x 16x.
For every case and attribute (phi_1 = 16x, phi_2 = 4x, phi_3 = base, cells N_i):

    r21, r32        refinement ratios, sqrt(N_1 / N_2) and sqrt(N_2 / N_3) (2D meshes, ~2 here)
    ratio           (phi_2 - phi_1) / (phi_3 - phi_2): 0..1 monotonic convergence, -1..0 oscillatory convergence,
                    |ratio| > 1 the differences grow with refinement (no convergence, e.g. at a sharp corner)
    order           observed order of convergence p (2 for a second order scheme, 1 for first order)
    extrapolated    Richardson extrapolation phi_ext = (r21^p phi_1 - phi_2) / (r21^p - 1), the estimate of the
                    mesh independent value (only for monotonic convergence)
    base_error_%, 4x_error_%, 16x_error_%   |phi - phi_ext| / |phi_ext| of every mesh
    gci_16x_%       grid convergence index of the 16x mesh, 1.25 |phi_1 - phi_2| / |phi_1| / (r21^p - 1)

Writes <input>_mesh_study.csv and prints a summary per attribute. Needs numpy.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import read_csv, write_csv

# lift is left out: it is close to zero for most obstacles, relative values mean nothing there
ATTRIBUTES = ['Fx', 'Fx_pressure', 'Fx_viscous', 'front_pressure', 'base_pressure', 'max_wall_shear',
              'peak_vorticity', 'recirculation_length_mm', 'recirculation_area_mm2', 'flow_split_upper',
              'wake_min_u_x', 'dissipation', 'dp_inlet_outlet']
MESHES = ['16x', '4x', 'base']  # fine, medium, coarse (phi_1, phi_2, phi_3)


def observed_order(phi1, phi2, phi3, r21, r32, iterations=100):
    """
    observed order of convergence p, solved by fixed-point iteration (Celik et al. 2008, eq. 3), nan when the
    differences are zero
    """
    e21, e32 = phi2 - phi1, phi3 - phi2
    if e21 == 0 or e32 == 0:
        return float('nan')
    s = math.copysign(1, e32 / e21)
    p = abs(math.log(abs(e32 / e21))) / math.log(r21)
    for _ in range(iterations):
        q = math.log((r21 ** p - s) / (r32 ** p - s))
        p = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
    return p


def study(phi1, phi2, phi3, n1, n2, n3):
    """
    mesh study of one attribute of one case (1 = fine, 3 = coarse), see the module docstring
    """
    r21, r32 = math.sqrt(n1 / n2), math.sqrt(n2 / n3)
    row = {'r21': r21, 'r32': r32}
    e21, e32 = phi2 - phi1, phi3 - phi2
    row['ratio'] = e21 / e32 if e32 != 0 else float('nan')

    if 0 < row['ratio'] < 1:
        row['convergence'] = 'monotonic'
    elif -1 < row['ratio'] < 0:
        row['convergence'] = 'oscillatory'
    elif e21 == 0 and e32 == 0:
        row['convergence'] = 'equal'
    else:
        row['convergence'] = 'none'

    p = observed_order(phi1, phi2, phi3, r21, r32)
    row['order'] = p
    if row['convergence'] == 'monotonic' and not math.isnan(p):
        extrapolated = (r21 ** p * phi1 - phi2) / (r21 ** p - 1)
        row['extrapolated'] = extrapolated
        for name, phi in zip(MESHES, (phi1, phi2, phi3)):
            row[f'{name}_error_%'] = 100 * abs(phi - extrapolated) / abs(extrapolated) if extrapolated else float('nan')
        row['gci_16x_%'] = 100 * 1.25 * abs(e21 / phi1) / (r21 ** p - 1) if phi1 else float('nan')
    return row


def main():
    parser = argparse.ArgumentParser(description="Mesh convergence study (base / 4x / 16x) of the flow attributes")
    parser.add_argument("attributes", help="openfoam_attributes.csv collected with --resolutions base 4x 16x")
    args = parser.parse_args()

    rows = {}
    for row in read_csv(args.attributes):
        if row['status'] == 'ok':
            rows.setdefault(row['case_name'], {})[row['resolution']] = row

    results = []
    for case_name, meshes in sorted(rows.items()):
        if not all(mesh in meshes for mesh in MESHES):
            print(f"{case_name}: skipped, not all of {MESHES} are ok")
            continue
        cells = [float(meshes[mesh]['cells']) for mesh in MESHES]
        for name in ATTRIBUTES:
            phi = [float(meshes[mesh][name]) for mesh in MESHES]
            result = {'case_name': case_name, 'attribute': name, 'base': phi[2], '4x': phi[1], '16x': phi[0]}
            result.update(study(*phi, *cells))
            results.append(result)

    output = Path(args.attributes).with_name(Path(args.attributes).stem + "_mesh_study.csv")
    fields = ['case_name', 'attribute', 'base', '4x', '16x', 'r21', 'r32', 'ratio', 'convergence', 'order',
              'extrapolated', 'base_error_%', '4x_error_%', '16x_error_%', 'gci_16x_%']
    write_csv(results, output, fields)

    print(f"{len(rows)} cases. Convergence: monotonic / oscillatory / none (differences grow with refinement);")
    print("order, errors (|phi - extrapolated| / |extrapolated|) and GCI: medians of the monotonic cases")
    print(f"{'attribute':26s} {'mono':>5s} {'osc':>4s} {'none':>5s} {'order':>6s} {'base %':>7s} {'4x %':>6s} "
          f"{'16x %':>6s} {'GCI 16x %':>9s}")
    for name in ATTRIBUTES:
        group = [row for row in results if row['attribute'] == name]
        counts = [sum(row['convergence'] == kind for row in group) for kind in ('monotonic', 'oscillatory', 'none')]
        monotonic = [row for row in group if row['convergence'] == 'monotonic']

        def median(key):
            values = [row[key] for row in monotonic if not math.isnan(row[key])]
            return np.median(values) if values else float('nan')

        print(f"{name:26s} {counts[0]:5d} {counts[1]:4d} {counts[2]:5d} {median('order'):6.2f} "
              f"{median('base_error_%'):7.1f} {median('4x_error_%'):6.1f} {median('16x_error_%'):6.1f} "
              f"{median('gci_16x_%'):9.1f}")
    print(f"Mesh study written to {output}")


if __name__ == "__main__":
    main()
