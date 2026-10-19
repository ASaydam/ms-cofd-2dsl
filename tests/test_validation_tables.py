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


def test_array_vs_openfoam(tmp_path):
    from validation.array_vs_openfoam import ATTRIBUTES, array_vs_openfoam

    for k, run in enumerate(('0.25x', 'base_fo', 'base_lu', '4x')):
        with open(tmp_path / f"flow_attributes_{run}_16x.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            names = [f'{a}_{r}_array_vs_openfoam_%' for a in ATTRIBUTES for r in (run, '16x')]
            writer.writerow(['case_name'] + names)
            for i, value in enumerate((-(k + 1), -(k + 1), k + 1)):          # two cases low, one high
                writer.writerow([f'normal_circle_{i:04d}'] + [value if n.endswith(f'_{run}_array_vs_openfoam_%')
                                                             else 0.5 for n in names])
    results = array_vs_openfoam(tmp_path)
    assert results['dissipation']['0.25x'][0] == pytest.approx(1.0)        # median |d|
    assert results['dissipation']['0.25x'][2] == pytest.approx(-1.0)       # median signed d
    assert results['dissipation']['4x'][0] == pytest.approx(4.0)
    assert results['dissipation']['16x'] == pytest.approx((0.5, 0.5, 0.5))
    assert np.isfinite(results['wake_min_u_x']['base_lu'][1])
