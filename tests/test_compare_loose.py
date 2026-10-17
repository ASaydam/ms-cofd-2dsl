import csv
import json
import sys

import numpy as np
import pytest

from test_audit_dataset import fake_sample
from tools import mesh_study

ROW = {'case_name': 'normal_rectangle_0001', 'size': 'normal', 'geometry': 'rectangle', 'index': '1',
       'x1': '120', 'x2': '160', 'x3': '160', 'x4': '120', 'y1': '40', 'y2': '40', 'y3': '70', 'y4': '70',
       'bbox_width': '40', 'bbox_height': '30'}
GRIDS = {'base_lu': (91, 31), '16x': (181, 61)}


def test_weighted_quantile_and_field_change():
    from validation.compare_loose import attribute_changes, field_change, weighted_quantile

    assert weighted_quantile([1, 2, 3], [1, 1, 1], 0.5) == pytest.approx(2)
    assert weighted_quantile([1, 2, 3], [1, 1, 10], 0.5) > 2.5        # a heavy case pulls the median
    converged = fake_sample(ROW, *GRIDS['16x']).astype(float)
    loose = converged.copy()
    loose[0] += 0.01                                                  # u_x off by 10 % of the inlet velocity
    loose[0][converged[4] == 1] = 0
    change = field_change(loose, converged, 0.1)
    assert change['rms_u_x_of_scale_%'] == pytest.approx(10) and change['rms_u_y_of_scale_%'] == 0
    attributes = {name: 1.0 for name in mesh_study.ATTRIBUTES}
    changed = attribute_changes(dict(attributes, Fx=1.1, Fx_viscous=1.2), attributes, 0.1, 30)
    assert changed['Fx'] == pytest.approx(10) and changed['Fx_viscous'] == pytest.approx(20)   # of the drag (1.0)


def test_compare_loose(tmp_path, monkeypatch):
    pytest.importorskip("scipy")
    from validation import compare_loose

    production, audit = tmp_path / 'production', tmp_path / 'audit'
    (production / 'geometry').mkdir(parents=True)
    with open(production / 'geometry' / 'manifest.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(ROW))
        writer.writeheader()
        writer.writerow(ROW)
    audit.mkdir()
    with open(audit / 'selection.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['case', 'size', 'family', 'group', 'inclusion_probability'])
        writer.writeheader()
        writer.writerow({'case': ROW['case_name'], 'size': 'normal', 'family': 'rectangle', 'group': 'test',
                         'inclusion_probability': '0.25'})

    attributes = {name: 1.0 for name in mesh_study.ATTRIBUTES}
    runs = {(production, 'base_lu'): (1.0, 600), (production, '16x'): (1.0, 3500),
            (audit, '16x_loose'): (1.0, 8), (audit, '16x_loose_scratch'): (0.9, 366),
            (audit, 'base_lu_loose_scratch'): (1.0, 250)}
    for (root, run), (factor, iterations) in runs.items():
        folder = root / 'results' / run
        folder.mkdir(parents=True)
        sample = fake_sample(ROW, *GRIDS['16x' if run.startswith('16x') else 'base_lu'])
        sample[0] *= factor                                           # the loose run from rest: 10 % slower
        np.savez_compressed(folder / f"{ROW['case_name']}.npz", sample=sample, time=float(iterations))
        (folder / f"{ROW['case_name']}_attributes.json").write_text(json.dumps(
            dict(attributes, recirculation_length_mm=50.0 * factor)))
        (folder / f"{ROW['case_name']}_provenance.json").write_text(json.dumps({'iterations': str(iterations)}))

    output = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['compare_loose', str(audit), str(production), str(output)])
    compare_loose.main()

    rows = {r['run']: r for r in csv.DictReader(open(output / 'loose_runs.csv'))}
    assert float(rows['16x_loose']['rms_u_x_of_scale_%']) == 0
    assert float(rows['16x_loose_scratch']['recirculation_length_mm']) == pytest.approx(-10)
    assert rows['16x_loose_scratch']['iterations'] == '366.0' and rows['16x_loose']['iterations_converged'] == '3500.0'
    summary = json.loads((output / 'loose_summary.json').read_text())
    assert summary['runs']['test']['16x_loose_scratch']['recirculation_length_mm']['median'] == pytest.approx(-10)
    gap = summary['gap']['test']
    assert gap['loose_mapped']['bilinear']['u_x']['median_abs'] == pytest.approx(gap['converged']['bilinear']['u_x']['median_abs'])
    assert gap['loose_scratch']['bilinear']['u_x']['median_abs'] > 0
