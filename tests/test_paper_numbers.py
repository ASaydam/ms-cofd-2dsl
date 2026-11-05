import csv
import json

import pytest

from dataset.splits import assign_splits

RUNS = ['0.25x', 'base_fo', 'base_lu', '4x', '16x']
CPUS = ['Intel(R) Xeon(R) CPU E5-2640 0 @ 2.50GHz', 'Intel(R) Xeon(R) CPU E5-2670 v2 @ 2.50GHz']
CONFIG = {'splits': {'seed': 2026, 'ood_quantile': 0.9, 'ood': ['blockage', 'rotation', 'position'],
                     'fractions': {'train': 0.8, 'val': 0.1, 'test': 0.1}},
          'placement': {'shift_x': [-30, 10], 'shift_y': [-15, 15]},
          'cases_per_geometry': {'smaller': 5, 'normal': 5, 'larger': 5}}


def fake_metadata():
    """
    20 cases, 5 per size and family; normal_left_triangle_0006 is the reserve case of the dropped 0003. Every run
    converged; the 0.25 mm run has 16 times the cells, 6 times the iterations and 100 times the solver time of the
    1 mm run; two relaxed 2 mm runs (575 and 7,652 iterations); the 1 mm meshes of case 0 and the 0.25 mm meshes of
    cases 0 and 1 meshed again; the first-order recirculation shorter than the label in all cases but the last
    """
    rows = []
    for size in ('normal', 'larger'):
        for geometry in ('circle', 'left_triangle'):
            indices = [1, 2, 4, 5, 6] if (size, geometry) == ('normal', 'left_triangle') else [1, 2, 3, 4, 5]
            for index in indices:
                i = len(rows)
                rows.append({'case_name': f"{size}_{geometry}_{index:04d}", 'size': size, 'geometry': geometry,
                             'index': index, 'blockage_ratio': 0.1 + 0.01 * i, 'rotation': -40.0 + 4.1 * i,
                             'shift_x': -30.0 + 2.0 * i, 'shift_y': -15.0 + 1.5 * ((7 * i) % 20)})
    splits = assign_splits(rows, CONFIG['splits'], CONFIG['placement'])
    cells = {'0.25x': 250, 'base_fo': 1000, 'base_lu': 1000, '4x': 4000, '16x': 16000}
    iterations = {'0.25x': 50, 'base_fo': 80, 'base_lu': 100, '4x': 200, '16x': 600}
    times = {'0.25x': 1.0, 'base_fo': 8.0, 'base_lu': 10.0, '4x': 100.0, '16x': 1000.0}
    for i, row in enumerate(rows):
        row['split'] = splits[row['case_name']]
        for run in RUNS:
            relaxed = run == '0.25x' and i < 2
            remeshed = (run in ('base_fo', 'base_lu') and i == 0) or (run == '16x' and i < 2)
            row.update({f'{run}.status': 'converged',
                        f'{run}.iterations': [575, 7652][i] if relaxed else iterations[run],
                        f'{run}.residual_Ux': 5e-7, f'{run}.residual_Uy': 9.99999e-7 if i == 3 else 8e-7,
                        f'{run}.residual_p': 1e-7, f'{run}.mesh_variant': '1 Mesh.Smoothing = 5;' if remeshed else '0',
                        f'{run}.templates': 'second-order-relaxed' if relaxed else 'second-order',
                        f'{run}.init_from': 'base_lu' if relaxed or run in ('4x', '16x') else 'rest',
                        f'{run}.cells': cells[run], f'{run}.execution_time_s': times[run],
                        f'{run}.peak_memory_kb': 1024 ** 2 if run == '16x' and i == 5 else 1000,
                        f'{run}.cpu_model': CPUS[i % 2], f'{run}.container_digest': 'sha256:abc'})
        row['base_fo.recirculation_length_mm'] = 10.0 if i == 19 else 9.0
        row['16x.recirculation_length_mm'] = 10.0
        row['16x.outlet_reverse_flow'] = 0.0
    return rows


def fake_release(folder):
    """metadata.parquet and the other release files of fake_metadata (no arrays)"""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    rows = fake_metadata()
    pq.write_table(pa.Table.from_pylist(rows), folder / 'metadata.parquet')
    # mesh study: drag not monotonic in the first 5 cases; observed order of the monotonic friction drag 0.2 for
    # circles and 0.5 for left triangles; front pressure not converging in 12 cases
    study = []
    for i, row in enumerate(rows):
        for name in ('Fx', 'Fx_viscous', 'front_pressure'):
            kind = 'none' if (name == 'Fx' and i < 5) or (name == 'front_pressure' and i < 12) else 'monotonic'
            order = 0.2 if row['geometry'] == 'circle' else 0.5
            study.append({'case_name': row['case_name'], 'attribute': name, 'convergence': kind, 'order': order})
    pq.write_table(pa.Table.from_pylist(study), folder / 'mesh_study.parquet')
    # uncertainty: the drag anomalous in 8 cases; dissipation U_% 0.5 against U1_% 1.5 (array rows: anomalous, left
    # out); recirculation length zero at 0.25 mm in 3 cases, 2 of them equal at all levels (U = 0)
    table = []
    for i, row in enumerate(rows):
        case = row['case_name']
        table += [{'case_name': case, 'attribute': 'Fx', 'source': 'openfoam', '16x': 1.0, 'U': 0.1,
                   'method': 'anomalous' if i < 8 else 'power', 'U_%': 10.0, 'U1_%': 1.0},
                  {'case_name': case, 'attribute': 'dissipation', 'source': 'openfoam', '16x': 1.0, 'U': 0.005,
                   'method': 'power', 'U_%': 0.5, 'U1_%': 1.5},
                  {'case_name': case, 'attribute': 'dissipation', 'source': 'array', '16x': 1.0, 'U': 0.5,
                   'method': 'anomalous', 'U_%': 50.0, 'U1_%': 1.5},
                  {'case_name': case, 'attribute': 'recirculation_length_mm', 'source': 'openfoam',
                   '16x': 0.0 if i < 3 else 10.0, 'U': 0.2 if i == 2 else 0.0 if i < 2 else 0.3,
                   'method': 'equal' if i < 2 else 'high_order', 'U_%': 3.0, 'U1_%': 3.0}]
    pq.write_table(pa.Table.from_pylist(table), folder / 'label_uncertainty.parquet')
    with open(folder / 'dropped_cases.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['case_name', 'size', 'geometry', 'index', 'blockage_ratio', 'iterations', 'residual_Ux',
                         'residual_Uy', 'residual_p', 'reason'])
        writer.writerow(['normal_left_triangle_0003', 'normal', 'left_triangle', '3', '0.38', '10000', '1e-4', '2e-2',
                         '3e-3', 'did not converge'])
    splits = {}
    for row in rows:
        splits[row['split']] = splits.get(row['split'], 0) + 1
    (folder / 'dataset_info.json').write_text(json.dumps(
        {'cases': len(rows), 'splits': splits, 'dropped_cases': 1, 'reserve_cases': 1,
         'runs': {run: {'grid': [9, 3]} for run in RUNS}, 'code_revisions': ['a', 'b'], 'container_digests': ['x']}))
    names = sorted(p.name for p in folder.iterdir())
    (folder / 'checksums.md5').write_text(''.join(f"0123  {name}\n" for name in names))
    return rows


def numbers_of(folder):
    from validation.paper_numbers import paper_numbers

    return {label: value for _, lines in paper_numbers(folder, CONFIG) for label, value in lines}


def test_paper_numbers(tmp_path):
    rows = fake_release(tmp_path)
    numbers = numbers_of(tmp_path)

    # Table 2, Geometry, Meshing, Flow solution
    assert numbers['16x (0.25 mm)'].startswith('cells 16,000 (16,000); iterations to 1e-6 600 (600); '
                                                'solver time on one core 1,000 s = 16.7 min')
    assert numbers['the blockage ratio ranges from ... to ... (median)'] == '0.1000 to 0.2900 (0.1950)'
    assert numbers['reserve cases in the dataset, one per excluded case'] == '1: 1 normal left_triangle'
    assert numbers['cases were excluded and replaced'] == '1: 1 normal left_triangle'
    assert numbers['and largest residuals of ... at the iteration cap'] == '2.00e-02 to 2.00e-02'
    assert numbers['needed for ... meshes (mesh_variant not 0)'] == \
        '0 at 2 mm, 1 at 1 mm first order, 1 at 1 mm second order, 0 at 0.5 mm, 2 at 0.25 mm'
    assert numbers['the 1 mm runs (each meshed on its own): the same cases'].startswith('yes')
    assert numbers['the largest final residual in the dataset (u_x, u_y, p)'] == '9.99999e-07'
    assert numbers['the longest runs needed ... iterations at 2 mm'] == f"7,652 ({rows[1]['case_name']})"
    assert numbers['2 mm runs started again from their 1 mm solution (relaxed template), started from'] == \
        '2: base_lu 2'
    assert numbers['these relaxed 2 mm runs converged after ... iterations'] == '575 to 7,652'

    # Attributes, Label uncertainty: the array rows are left out
    assert numbers['the front pressure did not converge with the mesh (convergence none)'] == '12 of 20 (60.0 %)'
    assert numbers['including ... of the drag values'] == '8 of 20 (40.0 %)'
    assert numbers['but ... of the dissipation values'] == '0 of 20 (0.0 %)'
    assert numbers['the fallback rule (fitted order p <= 0) applies to ... of the attribute-case pairs'] == \
        '8 of 60 (13.3 %)'
    assert numbers['where all four values are equal (U = 0): pairs'] == '2: recirculation_length_mm 2'
    assert numbers['pairs whose 0.25 mm value is zero (percentage undefined)'] == '3: recirculation_length_mm 3'
    assert numbers['of these, U = 0 / U > 0'] == '2 / 1'

    # Splits: two cases above the 90th percentile of every measure; the split column as dataset/splits.py
    assert numbers['blockage: above the 90th percentile (threshold), tail'] == '0.27, 2 cases'
    assert numbers['cases of the other splits in any tail'] == '0'
    assert numbers['split column equal to dataset/splits.py on the metadata'] == '20 of 20 (100.0 %)'

    # Computation and provenance: 100 times the time = 16 x 6 x (100 / 96) per iteration and cell
    assert numbers['the runs took ... solver core-hours'].startswith(f"{20 * 1119 / 3600:,.1f} (")
    assert numbers['... of them at 0.25 mm'] == f"{100 * 1000 / 1119:.1f} %"
    assert numbers['a 0.25 mm run costs about ... times as much as a 1 mm run (ratio of the median solver times)'] \
        .startswith('100.0')
    assert numbers['it has ... times the cells (ratio of the medians)'] == '16.00'
    assert numbers['and ... times the time per iteration and cell, E5-2640'].startswith('1.04')
    assert numbers['peak solver memory was at most'] == '1,048,576 KiB = 1,024 MiB = 1.00 GiB (1.07 GB)'
    assert numbers['image digests of the runs'] == '1: sha256:abc 100'

    # Data Records
    assert numbers['metadata.parquet has one row per case and ... columns'] == f"{len(rows[0])}"
    assert numbers['cases per size and family'] == '5 to 5 in the 4 classes'
    assert numbers['label_uncertainty.parquet: rows (attributes) per source'] == 'array 20 (1), openfoam 60 (3)'
    assert numbers['dataset_info.json: cases, dropped cases, reserve cases'] == '20, 1, 1'
    assert numbers['listed but not in the folder / in the folder but not listed'] == '0 / 0'

    # Technical Validation
    assert numbers['the three-level GCI is undefined (not monotonic) for the drag'] == '5 of 20 (25.0 %)'
    assert numbers['observed order of the friction drag (median of the monotonic cases; per family)'] == \
        '0.35; 0.20 to 0.50 (circle 0.20, left_triangle 0.50)'
    assert numbers['dissipation: median U against the simple estimate 3|phi_0.25 - phi_0.5|, %'] == '0.50 against 1.50'
    assert numbers['the first-order 1 mm recirculation length is shorter than the label'] == '19 of 20 (95.0 %)'


def test_paper_numbers_without_the_other_files(tmp_path):
    fake_release(tmp_path)
    for name in ('mesh_study.parquet', 'label_uncertainty.parquet', 'dropped_cases.csv', 'dataset_info.json',
                 'checksums.md5'):
        (tmp_path / name).unlink()
    numbers = numbers_of(tmp_path)
    for name in ('mesh_study.parquet', 'label_uncertainty.parquet', 'dropped_cases.csv', 'dataset_info.json',
                 'checksums.md5'):
        assert numbers[name] == 'not in the release folder, its numbers are left out'
    # from the metadata alone
    assert numbers['reserve cases in the dataset, one per excluded case'] == '1: 1 normal left_triangle'
    assert 'the three-level GCI is undefined (not monotonic) for the drag' not in numbers
