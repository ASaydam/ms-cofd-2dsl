import csv

import numpy as np
import pytest


def test_level_errors(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from validation.level_errors import ATTRIBUTES, RUNS, level_errors

    # two cases; every attribute 1.0 at 0.25 mm, the levels off by 1, 2, 3, 4 % of it; drag 2.0 (the scale of the drag
    # parts and lift), obstacle height 10 mm (the scale of the recirculation, larger than its value)
    rows = []
    for case in ('normal_circle_0001', 'normal_circle_0002'):
        row = {'case_name': case, 'bbox_height': 10.0}
        for name, _ in ATTRIBUTES:
            row[f'16x.{name}'] = 2.0 if name == 'Fx' else 1.0
            for k, (run, _) in enumerate(RUNS):
                row[f'{run}.{name}'] = row[f'16x.{name}'] * (1 + 0.01 * (k + 1))
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / 'metadata.parquet')
    uncertainty = [{'case_name': r['case_name'], 'attribute': name, 'source': source, 'U': u}
                   for r in rows for name, _ in ATTRIBUTES for source, u in (('openfoam', 0.02), ('array', 9.0))]
    pq.write_table(pa.Table.from_pylist(uncertainty), tmp_path / 'u.parquet')

    results = {name: (errors, u) for name, errors, u in level_errors(tmp_path, tmp_path / 'u.parquet')}
    errors, u = results['dissipation']                                     # relative to the label
    assert errors['0.25x'][0] == pytest.approx(1.0) and errors['4x'][0] == pytest.approx(4.0)
    assert u[0] == pytest.approx(2.0)                                      # the openfoam row, not the array row
    assert results['Fy'][0]['0.25x'][0] == pytest.approx(0.5)              # 0.01 of the drag of 2.0
    assert results['wake_min_u_x'][0]['0.25x'][0] == pytest.approx(10.0)   # 0.01 m/s of U_in = 0.1 m/s
    assert results['recirculation_length_mm'][0]['0.25x'][0] == pytest.approx(0.1)   # 0.01 mm of the 10 mm height


def test_exceedance(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from validation.level_errors import ATTRIBUTES, RUNS, exceedance

    # four cases with the label 1.0; every level off by 0.01, 0.02, 0.03, 0.04 (runs in order); U per case 0.015, 0.025,
    # 0 and missing: the first exceeds from the second run on, the second from the third, the third (U = 0) everywhere,
    # the fourth is left out
    rows, uncertainty = [], []
    for i, u in enumerate((0.015, 0.025, 0.0, None)):
        case = f'normal_circle_{i:04d}'
        row = {'case_name': case}
        for name, _ in ATTRIBUTES:
            row[f'16x.{name}'] = 1.0
            for k, (run, _) in enumerate(RUNS):
                row[f'{run}.{name}'] = 1.0 + 0.01 * (k + 1)
            uncertainty.append({'case_name': case, 'attribute': name, 'source': 'openfoam', 'U': u})
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / 'metadata.parquet')
    pq.write_table(pa.Table.from_pylist(uncertainty), tmp_path / 'u.parquet')

    results = {name: (shares, left_out) for name, shares, left_out in exceedance(tmp_path, tmp_path / 'u.parquet')}
    shares, left_out = results['Fx']
    assert left_out == 1
    assert shares['0.25x'] == (pytest.approx(100 / 3), 3)                  # only the U = 0 case
    assert shares['base_fo'][0] == pytest.approx(200 / 3)                 # 0.02 > 0.015
    assert shares['4x'][0] == pytest.approx(100.0)


def test_array_vs_openfoam(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from validation.array_vs_openfoam import ATTRIBUTES, array_vs_openfoam

    # three cases, obstacle height 10 mm; every OpenFOAM value 1.0 (the wake velocity 0.0, the wall zero of a case
    # without reverse flow), the arrays off by k + 1 % of 1.0 at level k (two cases low, one high), 0.5 % at 16x
    cases = [f'normal_circle_{i:04d}' for i in range(3)]
    pq.write_table(pa.Table.from_pylist([{'case_name': c, 'bbox_height': 10.0} for c in cases]),
                   tmp_path / 'metadata.parquet')
    for k, run in enumerate(('0.25x', 'base_fo', 'base_lu', '4x')):
        with open(tmp_path / f"flow_attributes_{run}_16x.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['case_name'] + [f'{a}_{r}{s}' for a in ATTRIBUTES for r in (run, '16x') for s in ('', '_array')])
            for case, sign in zip(cases, (-1, -1, 1)):
                values = []
                for a in ATTRIBUTES:
                    mesh = 0.0 if a == 'wake_min_u_x' else 1.0
                    values += [mesh, mesh + sign * 0.01 * (k + 1), mesh, mesh + 0.005]
                writer.writerow([case] + values)
    results = array_vs_openfoam(tmp_path, tmp_path / 'metadata.parquet')
    assert results['dissipation']['0.25x'][0] == pytest.approx(1.0)        # median |d|, relative to the value
    assert results['dissipation']['0.25x'][2] == pytest.approx(-1.0)       # median signed d
    assert results['dissipation']['4x'][0] == pytest.approx(4.0)
    assert results['dissipation']['16x'] == pytest.approx((0.5, 0.5, 0.5))
    assert results['recirculation_length_mm']['0.25x'][0] == pytest.approx(0.1)     # 0.01 mm of the 10 mm height
    assert results['wake_min_u_x']['base_lu'][0] == pytest.approx(30.0)    # 0.03 m/s of U_in = 0.1 m/s, mesh value 0
