import csv
import json
import sys

import numpy as np
import pytest

from extraction.build_arrays import build_sample
from tools import mesh_study
from test_compare_loose import ROW

CASE = ROW['case_name']


def fields_of(nx=360, ny=120):
    """resampled fields of a channel with a rectangular obstacle (as extraction/resample_vtk.py returns them)"""
    x = np.arange(nx)[:, None] * 360 / (nx - 1) * np.ones((1, ny))
    y = np.ones((nx, 1)) * np.arange(ny)[None, :] * 120 / (ny - 1)
    valid = ~((x >= 120) & (x <= 160) & (y >= 40) & (y <= 70))
    ux = np.where(valid, 0.6 * y * (120 - y) / 120 ** 2, 0)
    return {'u_x': ux, 'u_y': 0.01 * ux, 'p': np.where(valid, 1e-3 * (360 - x) / 360, 0), 'valid': valid.astype(float),
            'time': 900.0}


def write_run(root, run, fields, attributes, iterations='900'):
    folder = root / 'results' / run
    folder.mkdir(parents=True, exist_ok=True)
    sample = build_sample(fields['u_x'], fields['u_y'], fields['p'], fields['valid']).astype(np.float32)
    np.savez_compressed(folder / f"{CASE}.npz", sample=sample, time=fields['time'])
    (folder / f"{CASE}_attributes.json").write_text(json.dumps(attributes))
    (folder / f"{CASE}_provenance.json").write_text(json.dumps({'iterations': iterations}))


def production_root(root):
    (root / 'geometry' / 'geo-files' / 'base').mkdir(parents=True)
    (root / 'geometry' / 'geo-files' / '0.25x').mkdir(parents=True)
    for resolution in ('base', '0.25x'):
        (root / 'geometry' / 'geo-files' / resolution / f"{CASE}.geo").write_text(f"// {resolution}\n")
    with open(root / 'geometry' / 'manifest.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(ROW))
        writer.writeheader()
        writer.writerow(ROW)
    attributes = {name: 1.0 for name in mesh_study.ATTRIBUTES}
    write_run(root, 'base_lu', fields_of(), attributes)
    write_run(root, '0.25x', fields_of(180, 60), attributes)
    (root / 'archives').mkdir()
    with open(root / 'archives' / 'archive_index.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['run', 'case', 'file', 'bytes', 'md5', 'remote', 'uploaded'])
        writer.writeheader()
        writer.writerows([
            {'run': 'base_lu', 'case': CASE, 'md5': 'old', 'remote': f"r:a/base_lu/{CASE}.tar.zst", 'uploaded': '2026-10-05'},
            {'run': 'base_lu', 'case': CASE, 'md5': 'new', 'remote': f"r:a/base_lu/{CASE}.tar.zst", 'uploaded': '2026-10-06'},
            {'run': 'base_lu', 'case': CASE, 'md5': 'cap', 'remote': f"r:a/base_lu/superseded/{CASE}.cap.tar.zst",
             'uploaded': '2026-10-09'}])
    return attributes


def test_archive_row_and_differences(tmp_path):
    from validation.check_reproduction import archive_row, attribute_difference, field_difference

    production_root(tmp_path)
    assert archive_row(tmp_path, 'base_lu', CASE)['md5'] == 'new'          # newest live row, superseded left out
    assert archive_row(tmp_path, '4x', CASE) is None
    worst, name, missing = attribute_difference({'Fx': 2.0, 'Fy': 0.1, 'p': 1.0}, {'Fx': 2.0, 'Fy': 0.0, 'p': 1.0, 'q': 3.0})
    assert worst == pytest.approx(0.05) and name == 'Fy' and missing == ['q']   # lift relative to the drag
    fields = fields_of()
    sample = build_sample(fields['u_x'], fields['u_y'], fields['p'], fields['valid'])
    changed = sample.copy()
    changed[0][sample[4] == 4] *= 1.001
    row = field_difference(changed, sample)
    assert row['max_u_x'] == pytest.approx(0.001, rel=1e-3) and row['max_p'] == 0 and row['geometry_equal']


def test_setup_and_solved(tmp_path, monkeypatch):
    from validation import check_reproduction

    production, check = tmp_path / 'production', tmp_path / 'check'
    production.mkdir()
    attributes = production_root(production)
    with pytest.raises(SystemExit):
        check_reproduction.setup(production, production / 'results' / 'x', [CASE])  # inside the production root
    check_reproduction.setup(production, check, [CASE])
    assert (check / 'cases.txt').read_text().split() == [CASE]
    assert (check / 'geometry' / 'geo-files' / 'base' / f"{CASE}.geo").exists()
    assert 'validation-repro-check' in (check / 'dataset.yaml').read_text()

    write_run(check, 'base_lu', fields_of(), attributes)
    write_run(check, '0.25x', fields_of(180, 60), dict(attributes, Fx=1.0 + 1e-7))
    assert not check_reproduction.solved(check, production, tmp_path / 'out')
    rows = list(csv.DictReader(open(tmp_path / 'out' / 'reproduction_solved.csv')))
    assert len(rows) == 2 and all(float(r['max_u_x']) == 0 for r in rows)

    changed = fields_of()
    changed['u_x'] = changed['u_x'] * 1.01
    write_run(check, 'base_lu', changed, attributes)
    assert check_reproduction.solved(check, production, tmp_path / 'out')        # 1 % off: not reproduced


def test_archives_flow(tmp_path, monkeypatch):
    pytest.importorskip("scipy")
    from extraction import resample_vtk
    from tools import openfoam_attributes
    from validation import check_reproduction

    production = tmp_path / 'production'
    production.mkdir()
    attributes = production_root(production)
    monkeypatch.setattr(check_reproduction, 'restore', lambda row, folder: (folder / CASE).mkdir(parents=True))
    monkeypatch.setattr(check_reproduction.subprocess, 'run', lambda *a, **k: None)
    monkeypatch.setattr(openfoam_attributes, 'obstacle_geometry', lambda row, domain: ([0.1] * 4, 0.14, 0.18))
    monkeypatch.setattr(openfoam_attributes, 'collect_case', lambda work: dict(attributes))
    monkeypatch.setattr(resample_vtk, 'resample_case', lambda folder, nx, ny: fields_of(nx, ny))
    assert not check_reproduction.archives(production, tmp_path / 'work', tmp_path, ['base_lu:' + CASE])
    row = next(csv.DictReader(open(tmp_path / 'reproduction_archives.csv')))
    assert row['arrays_identical'] == 'True' and float(row['attributes_max_difference']) == 0
    assert not (tmp_path / 'work' / f"base_lu-{CASE}").exists()                  # the restored case is removed

    monkeypatch.setattr(resample_vtk, 'resample_case', lambda folder, nx, ny: dict(fields_of(nx, ny), time=901.0))
    assert check_reproduction.archives(production, tmp_path / 'work', tmp_path, ['base_lu:' + CASE])
