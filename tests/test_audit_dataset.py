import csv
import json
import sys

import numpy as np
import pytest
import yaml

from extraction.build_arrays import cell_type_mask, wall_distance
from hpc import ledger
from hpc.upload import INDEX_FIELDS
from tools import mesh_study
from utils import MANIFEST_FIELDS, load_config, write_csv

RUNS = ['0.25x', 'base_fo', 'base_lu', '4x', '16x']
GRIDS = {'0.25x': [46, 16], 'base': [91, 31], '4x': [181, 61], '16x': [181, 61]}
CELLS = {'0.25x': 300, 'base_fo': 1000, 'base_lu': 1000, '4x': 4000, '16x': 16000}


def inside(x, y, xs, ys):
    """points (x, y) inside the polygon (ray casting)"""
    result = np.zeros(x.shape, bool)
    for k in range(len(xs)):
        x1, y1, x2, y2 = xs[k], ys[k], xs[(k + 1) % len(xs)], ys[(k + 1) % len(xs)]
        if y1 != y2:
            result ^= ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
    return result


def fake_sample(row, nx, ny, inlet=0.1):
    """a sample of the channel flow without the obstacle's wake: Poiseuille profile, 0 inside the obstacle"""
    x = np.arange(nx)[:, None] * 360 / (nx - 1) * np.ones((1, ny))
    y = np.ones((nx, 1)) * np.arange(ny)[None, :] * 120 / (ny - 1)
    obstacle = inside(x, y, [float(row[f"x{k}"]) for k in range(1, 5)], [float(row[f"y{k}"]) for k in range(1, 5)])
    obstacle[[0, -1], :] = obstacle[:, [0, -1]] = False
    sample = np.zeros((6, nx, ny))
    sample[0] = np.where(obstacle, 0, 6 * inlet * y * (120 - y) / 120 ** 2)
    sample[2] = np.where(obstacle, 0, 1e-3 * (360 - x) / 360)
    sample[3], sample[4] = wall_distance(nx, ny), cell_type_mask(obstacle)
    sample[5] = np.where(obstacle, -1.0, 1.0)
    return sample.astype(np.float32)


def fake_root(root, monkeypatch):
    """
    a data root of 3 normal rectangles (two per class): 0002 without a steady solution, replaced by the reserve 0003;
    every run done with consistent results, archives and .geo files
    """
    from geometries.write_geo import create_geo

    root.mkdir()
    (root / 'dataset.yaml').write_text(yaml.safe_dump({'geometries': ['rectangle'], 'cases_per_geometry':
                                                       {'normal': 2}, 'extraction': GRIDS}))
    monkeypatch.setenv('MSCOFD_DATA_ROOT', str(root))
    config = load_config()
    rows = create_geo(config, ['normal'], ['rectangle'], config['default_resolutions'], end=3)
    write_csv(rows, root / 'geometry' / 'manifest.csv', MANIFEST_FIELDS)

    index = []
    for run in RUNS:
        settings = config['runs'][run]
        folder = root / 'results' / run
        folder.mkdir(parents=True)
        summary = []
        for row in rows:
            case = row['case_name']
            unsteady = case == 'normal_rectangle_0002'
            if unsteady and run != 'base_lu':
                continue
            status = 'end_time' if unsteady else 'converged'
            np.savez_compressed(folder / f"{case}.npz", sample=fake_sample(row, *GRIDS[settings['resolution']]),
                                time=900.0)
            attributes = {name: 1.0 for name in mesh_study.ATTRIBUTES}
            attributes.update({'outlet_reverse_flow': 0.0, 'cells': CELLS[run], 'faces': 10, 'time': '900'})
            (folder / f"{case}_attributes.json").write_text(json.dumps(attributes))
            (folder / f"{case}_provenance.json").write_text(json.dumps(
                {'run': run, 'case': case, 'templates': settings['templates'], 'status': status,
                 'iterations': '900', 'solver_runs': 1, 'execution_time_s': 60.0, 'residual_Ux': '5e-07',
                 'residual_Uy': '9.9e-07', 'residual_p': '1e-07', 'mesh_ok': True, 'cells': str(CELLS[run]),
                 'exported_time': 900.0, 'mesh_variant': '0', 'end_time_override': None, 'dropped_times': [],
                 'init_from': settings.get('init_from') or '', 'case_settings': {}, 'peak_memory_kb': 1000,
                 'code_revision': 'abc', 'container_digest': 'sha256:1', 'cpu_model': 'cpu'}))
            summary.append({'case': case, 'status': status, 'iterations': '900'})
            ledger.set_state(run, case, 'done', f"{status} 900 1", root=root)
            index.append({'run': run, 'case': case, 'file': f"{case}.tar.zst", 'bytes': 100,
                          'md5': f"md5-{run}-{case}", 'remote': f"remote:archives/{run}/{case}.tar.zst",
                          'uploaded': '2026-10-05T10:00:00+0300'})
        with open(folder / 'summary.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['case', 'status', 'iterations'])
            writer.writeheader()
            writer.writerows(summary)
    (root / 'archives').mkdir()
    with open(root / 'archives' / 'archive_index.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=INDEX_FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(index)
    return {row['case_name']: row for row in rows}, index


def run_audit(tmp_path, monkeypatch, *options):
    from validation import audit_dataset

    output = tmp_path / 'audit'
    monkeypatch.setattr(sys, 'argv', ['audit_dataset', str(output), *options])
    with pytest.raises(SystemExit) as stop:
        audit_dataset.main()
    findings = list(csv.DictReader(open(output / 'findings.csv', newline='')))
    return stop.value.code, findings, json.loads((output / 'summary.json').read_text())


def test_audit_of_a_consistent_root(tmp_path, monkeypatch):
    fake_root(tmp_path / 'root', monkeypatch)
    code, findings, summary = run_audit(tmp_path, monkeypatch)
    assert [f for f in findings if f['level'] != 'info'] == [] and code == 0
    assert summary['members']['dropped_cases'] == ['normal_rectangle_0002']
    assert summary['members']['reserve_cases'] == ['normal_rectangle_0003']
    assert summary['provenance']['16x']['iterations']['median'] == 900
    assert summary['arrays']['inlet_velocity'] == 0.1 and summary['arrays']['16x']['flux_deviation_%']['max'] < 1
    assert sum(summary['splits']['counts'].values()) == 2
    assert (tmp_path / 'audit' / 'report.md').exists() and (tmp_path / 'audit' / 'splits.csv').exists()


def test_audit_finds_the_faults(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    manifest, index = fake_root(root, monkeypatch)
    first, reserve = 'normal_rectangle_0001', 'normal_rectangle_0003'

    path = root / 'results' / '16x' / f"{first}_provenance.json"                  # not converged to 1e-6
    path.write_text(path.read_text().replace('9.9e-07', '2e-06'))
    path = root / 'results' / 'base_lu' / f"{reserve}.npz"                       # a value that is not finite
    with np.load(path) as data:
        sample = data['sample'].copy()
    sample[0, 20, 10] = np.nan
    np.savez_compressed(path, sample=sample, time=900.0)
    moved = dict(manifest[first])                                                # the obstacle of another case
    for k in range(1, 5):
        moved[f"x{k}"] = float(manifest[first][f"x{k}"]) + 20
    np.savez_compressed(root / 'results' / '4x' / f"{first}.npz", sample=fake_sample(moved, *GRIDS['4x']), time=900.0)
    for run, case, timings in (('16x', reserve, {'mapFields': 50.0}), ('4x', first, {})):  # before init_from existed
        path = root / 'results' / run / f"{case}_provenance.json"
        record = json.loads(path.read_text())
        del record['init_from']
        record['timings_s'] = timings
        path.write_text(json.dumps(record))
    path = root / 'results' / '0.25x' / f"{first}_attributes.json"               # negative drag
    path.write_text(path.read_text().replace('"Fx": 1.0', '"Fx": -1.0'))
    path = root / 'geometry' / 'geo-files' / '16x' / f"{first}.geo"              # changed by hand
    path.write_text(path.read_text() + '\n')
    with open(root / 'archives' / 'archive_index.csv', 'a', newline='') as file:  # solved again: two live rows
        csv.DictWriter(file, fieldnames=INDEX_FIELDS, lineterminator='\n').writerow(
            dict(index[0], md5='md5-old', uploaded='2026-10-04T10:00:00+0300'))

    from validation import audit_dataset

    def listing(folder):
        run = folder.rsplit('/', 1)[1]
        files = {f"{row['case']}.tar.zst": (100, row['md5']) for row in index if row['run'] == run}
        if run == '16x':
            files[f"{reserve}.tar.zst"] = (100, 'md5-other')                     # changed on the remote
        return files

    monkeypatch.setattr(audit_dataset, 'remote_listing', listing)
    code, findings, summary = run_audit(tmp_path, monkeypatch, '--remote')

    def found(level, check, run, case, text):
        return any(f['level'] == level and f['check'] == check and f['run'] == run and f['case'] == case and
                   text in f['message'] for f in findings)

    assert code == 1
    assert found('error', 'provenance', '16x', first, 'residuals above the criterion')
    assert found('error', 'arrays', 'base_lu', reserve, 'not finite')
    assert found('warning', 'arrays', '4x', first, 'centroid')
    assert found('error', 'attributes', '0.25x', first, 'Fx = -1.0')
    assert found('error', 'geometry', '', first, '16x .geo file differs')
    assert found('warning', 'archives', index[0]['run'], index[0]['case'], '2 live rows')
    assert found('error', 'archives', '16x', reserve, 'remote MD5 md5-other')
    assert not any(f['check'] == 'archives' and f['level'] == 'error' and f['run'] != '16x' for f in findings)
    assert summary['archives']['duplicate_live_rows'] == 1
    assert found('warning', 'provenance', '4x', first, 'init_from not recorded')    # no mapFields step: archive
    assert not any(f['check'] == 'provenance' and (f['run'], f['case']) == ('16x', reserve) for f in findings)
    assert summary['provenance']['16x']['init_from_not_recorded'] == 1


def test_levels_warn_only_for_many_outliers(monkeypatch):
    from validation import audit_dataset

    rows = {f"normal_circle_{i:04d}": {'size': 'normal', 'geometry': 'circle', 'bbox_height': '30'} for i in range(12)}
    values = {}
    for i, case in enumerate(rows):
        values[('16x', case)] = {name: 1.0 for name in mesh_study.ATTRIBUTES}
        values[('base_lu', case)] = {name: 1.0 + 0.001 * i for name in mesh_study.ATTRIBUTES}
    values[('base_lu', 'normal_circle_0003')].update(Fx=1.3, dissipation=1.3, mean_wall_shear=1.3)   # mixed up
    values[('base_lu', 'normal_circle_0007')].update(dissipation=1.3)                             # one attribute
    values[('base_lu', 'normal_circle_0009')].update(front_pressure=5.0)                           # not tested
    audit = audit_dataset.Audit()
    audit_dataset.check_levels(audit, ['base_lu', '16x'], list(rows), rows, values)
    warnings = [(f['case'], f['message']) for f in audit.findings if f['level'] == 'warning']
    assert len(warnings) == 1 and warnings[0][0] == 'normal_circle_0003' and '3 attributes' in warnings[0][1]
    assert any(f['case'] == 'normal_circle_0007' and f['level'] == 'info' for f in audit.findings)
    assert not any(f['case'] == 'normal_circle_0009' for f in audit.findings)
    assert audit.summary['levels']['median_abs_difference_%']['base_lu']['front_pressure'] > 0


def test_level_difference_scales():
    from validation.audit_dataset import level_difference

    assert level_difference('Fx_viscous', 0.11, 0.10, 2.0, 0.1, 30) == pytest.approx(0.005)    # of the drag
    assert level_difference('recirculation_length_mm', 3.0, 2.0, 1.0, 0.1, 30) == pytest.approx(1 / 30)
    assert level_difference('recirculation_length_mm', 60.0, 50.0, 1.0, 0.1, 30) == pytest.approx(0.2)
    assert level_difference('Fx', 1.1, 1.0, 1.0, 0.1, 30) == pytest.approx(0.1)
