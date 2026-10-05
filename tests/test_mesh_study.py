import math

import pytest

from tools.mesh_study import observed_order, study


def value(h, order, exact=1.0, constant=0.3):
    """attribute of a mesh with spacing h and a discretization error of the given order"""
    return exact + constant * h ** order


def test_second_order_values_are_extrapolated_to_the_exact_value():
    phi = [value(h, 2) for h in (0.25, 0.5, 1.0)]  # 16x, 4x, base
    cells = [16.0, 4.0, 1.0]
    result = study(*phi, *cells)
    assert result['convergence'] == 'monotonic'
    assert math.isclose(result['order'], 2.0, rel_tol=1e-6)
    assert math.isclose(result['extrapolated'], 1.0, rel_tol=1e-9)
    assert result['16x_error_%'] < result['4x_error_%'] < result['base_error_%']


def test_observed_order_with_unequal_refinement_ratios():
    r21, r32 = 2.1, 1.9
    h1 = 0.25
    phi1, phi2, phi3 = value(h1, 1.5), value(h1 * r21, 1.5), value(h1 * r21 * r32, 1.5)
    assert math.isclose(observed_order(phi1, phi2, phi3, r21, r32), 1.5, rel_tol=1e-6)


def test_growing_differences_are_not_converged():
    # a peak value at a sharp corner: the differences grow with refinement
    result = study(4.0, 2.0, 1.0, 16.0, 4.0, 1.0)
    assert result['convergence'] == 'none'
    assert 'extrapolated' not in result


def test_equal_differences_give_no_order():
    # equal steps between the levels: the first estimate of p is 0, where the fixed-point iteration divides by zero
    result = study(1.5, 1.25, 1.0, 16000, 4000, 1000)
    assert result['convergence'] == 'none' and math.isnan(result['order'])


def test_run_summary_table_and_extra_levels(tmp_path):
    """per-run results -> attribute table -> mesh study with the 2 mm level as an extra level"""
    import csv
    import json
    import subprocess
    import sys

    from tools import run_summary

    cells = {'0.25x': 1000, 'base_lu': 4000, '4x': 16000, '16x': 64000}
    value = {'0.25x': 1.16, 'base_lu': 1.04, '4x': 1.01, '16x': 1.0025}   # second order towards 1
    for run, n in cells.items():
        folder = tmp_path / 'results' / run
        folder.mkdir(parents=True)
        (folder / 'c1_provenance.json').write_text(json.dumps(
            {'status': 'converged', 'iterations': '100', 'clock_time_s': 60.0, 'peak_memory_kb': 2048, 'cells': n}))
        (folder / 'c1_attributes.json').write_text(json.dumps(
            {'cells': n, 'Fx': value[run], 'outlet_reverse_flow': 0.0}))
    sys.argv = ['run_summary', str(tmp_path)]
    run_summary.main()
    table = tmp_path / 'results' / 'openfoam_attributes.csv'
    rows = {r['resolution']: r for r in csv.DictReader(open(table))}
    assert set(rows) == {'0.25x', 'base', '4x', '16x'} and rows['base']['Fx'] == '1.04'

    pytest.importorskip("numpy")
    subprocess.run([sys.executable, '-m', 'tools.mesh_study', str(table), '--extra', '0.25x'], check=True,
                   capture_output=True)
    study = {r['attribute']: r for r in csv.DictReader(open(tmp_path / 'results' / 'openfoam_attributes_mesh_study.csv'))}
    assert abs(float(study['Fx']['order']) - 2) < 0.01
    assert abs(float(study['Fx']['0.25x_error_%']) - 16) < 0.1
