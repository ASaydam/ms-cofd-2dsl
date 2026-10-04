"""
Compares the solutions of two OpenFOAM versions on the same meshes (validation/solver_check.sh, docs/validation.md):

    python validation/compare_solvers.py <new cases> <reference cases> <new attributes.csv> <reference attributes.csv>

<new cases> and <reference cases> hold <resolution>/<case> folders. For every run of the new version:
(1) whether the meshes are identical (polyMesh owner and neighbour), (2) the iterations to convergence and the
relative difference of U and p cell by cell (RMS and maximum, relative to the RMS and maximum of the reference),
(3) the relative difference of every attribute (tools/openfoam_attributes.py; lift relative to the drag).
"""

import csv
import statistics
import sys
from pathlib import Path

import numpy as np


def read_list(path):
    """values of an ascii OpenFOAM list: the internal field of a field file, or a polyMesh list"""
    lines = Path(path).read_text().splitlines()
    field = [i for i, line in enumerate(lines) if line.startswith('internalField')]
    start = field[0] + 1 if field else \
        next(i for i, line in enumerate(lines) if line.strip().isdigit() and lines[i + 1].strip() == '(')
    count = int(lines[start])
    values = ' '.join(lines[start + 2:start + 2 + count]).replace('(', ' ').replace(')', ' ').split()
    array = np.array(values, dtype=float)
    return array.reshape(count, -1) if array.size != count else array


def latest(folder):
    return max((p for p in folder.iterdir() if p.is_dir() and p.name.replace('.', '').isdigit()),
               key=lambda p: float(p.name))


def rms(array):
    return float(np.sqrt(np.mean(array ** 2)))


def compare_fields(new_cases, reference_cases):
    print('(1)-(2) per run: mesh, iterations, difference of the fields relative to the reference (RMS, maximum)')
    print(f"  {'run':5s} {'case':26s} {'mesh':5s} {'iterations':>11s} {'u_x rms':>9s} {'u_y rms':>9s} {'p rms':>9s} "
          f"{'u_x max':>9s} {'p max':>9s}")
    for run_folder in sorted(p for p in new_cases.iterdir() if p.is_dir()):
        for new in sorted(p for p in run_folder.iterdir() if p.is_dir()):
            reference = reference_cases / run_folder.name / new.name
            same_mesh = all(np.array_equal(read_list(new / 'constant/polyMesh' / name),
                                           read_list(reference / 'constant/polyMesh' / name))
                            for name in ('owner', 'neighbour'))
            time_new, time_reference = latest(new), latest(reference)
            u, u_ref = read_list(time_new / 'U'), read_list(time_reference / 'U')
            p, p_ref = read_list(time_new / 'p'), read_list(time_reference / 'p')
            row = [rms(u[:, 0] - u_ref[:, 0]) / rms(u_ref[:, 0]), rms(u[:, 1] - u_ref[:, 1]) / rms(u_ref[:, 1]),
                   rms(p - p_ref) / rms(p_ref), float(np.abs(u[:, 0] - u_ref[:, 0]).max() / np.abs(u_ref[:, 0]).max()),
                   float(np.abs(p - p_ref).max() / np.abs(p_ref).max())]
            print(f"  {run_folder.name:5s} {new.name:26s} {'same' if same_mesh else 'DIFF':5s} "
                  f"{time_new.name + '/' + time_reference.name:>11s} " + ' '.join(f"{100 * v:8.4f}%" for v in row))


def compare_attributes(new_csv, reference_csv):
    def table(path):
        return {(r['case_name'], r['resolution']): r for r in csv.DictReader(open(path)) if r['status'] == 'ok'}

    new, reference = table(new_csv), table(reference_csv)
    keys = sorted(set(new) & set(reference))
    names = [name for name in csv.DictReader(open(new_csv)).fieldnames
             if name not in ('case_name', 'resolution', 'status', 'time', 'cells', 'faces')]
    print(f"\n(3) attributes, |new - reference| / |reference| in % over {len(keys)} runs (lift relative to the drag)")
    print(f"  {'attribute':26s} {'median':>9s} {'max':>9s}")
    for name in names:
        differences = []
        for key in keys:
            a, b = new[key].get(name, ''), reference[key].get(name, '')
            if a in ('', 'nan') or b in ('', 'nan'):
                continue
            scale = abs(float(reference[key]['Fx'])) if name.startswith('Fy') else abs(float(b))
            if scale > 1e-30:
                differences.append(abs(float(a) - float(b)) / scale * 100)
        if differences:
            print(f"  {name:26s} {statistics.median(differences):9.4f} {max(differences):9.4f}")


if __name__ == "__main__":
    compare_fields(Path(sys.argv[1]), Path(sys.argv[2]))
    compare_attributes(sys.argv[3], sys.argv[4])
