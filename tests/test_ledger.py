from datetime import datetime

import pytest

from hpc import ledger

RUNS = ['base_fo', 'base_lu', '4x', '16x']


def manifest(n=2):
    return [{'case_name': f"{size}_{geometry}_{i:04d}", 'size': size, 'geometry': geometry, 'index': str(i)}
            for size in ('normal', 'larger') for geometry in ('circle', 'diamond') for i in range(1, n + 1)]


def test_case_order_interleaves_sizes_and_geometries():
    order = ledger.case_order(manifest())
    assert order[:4] == ['larger_circle_0001', 'larger_diamond_0001', 'normal_circle_0001', 'normal_diamond_0001']
    assert all(name.endswith('_0002') for name in order[4:])


def test_fine_runs_wait_for_base_lu():
    cases = ledger.case_order(manifest(1))
    ready = ledger.ready_tasks(cases, RUNS, {})
    assert {run for run, _ in ready} == {'base_lu', 'base_fo'}
    assert ready[0][0] == 'base_lu'

    states = {('base_lu', cases[0]): ('done', '')}
    ready = ledger.ready_tasks(cases, RUNS, states)
    assert ('16x', cases[0]) in ready and ('4x', cases[0]) in ready
    assert ('16x', cases[1]) not in ready and ('base_lu', cases[0]) not in ready
    # 16x before 4x before base_fo
    order = [run for run, case in ready if case == cases[0]]
    assert order == ['16x', '4x', 'base_fo']


def test_set_state_keeps_one_marker(tmp_path):
    ledger.set_state('16x', 'larger_circle_0001', 'submitted', '123 attempt=1', root=tmp_path)
    ledger.set_state('16x', 'larger_circle_0001', 'running', '123 be01', root=tmp_path)
    states = ledger.read_states(['16x'], tmp_path)
    assert states == {('16x', 'larger_circle_0001'): ('running', '123 be01')}
    assert sorted(p.name for p in (tmp_path / 'state' / '16x').iterdir()) == ['larger_circle_0001.running']


def test_qos_window_by_date():
    pytest.importorskip("fcntl")  # topup.py runs on the cluster (Linux)
    from hpc import topup

    slurm, _ = topup.load_settings()
    inside = datetime.fromisoformat("2026-10-10T12:00:00+03:00")
    after = datetime.fromisoformat("2026-10-19T00:00:01+03:00")
    assert topup.current_window(slurm, inside)['qos'] == 'asaydam_oct26'
    assert topup.current_window(slurm, after)['qos'] == 'users'


def test_data_root_overrides(tmp_path):
    pytest.importorskip("fcntl")
    from hpc import topup

    slurm, _ = topup.load_settings(tmp_path)
    assert slurm['footprint']['pause_used_gib'] == 600
    assert topup.cases_file_of(tmp_path) is None

    (tmp_path / 'slurm.yaml').write_text("footprint: {pause_used_gib: 900, pause_backlog_gib: 50}\n")
    (tmp_path / 'cases.txt').write_text("normal_circle_0001\n")
    slurm, _ = topup.load_settings(tmp_path)
    assert slurm['footprint'] == {'pause_used_gib': 900, 'pause_backlog_gib': 50}
    assert slurm['windows'][0]['qos'] == 'asaydam_oct26'
    assert topup.cases_file_of(tmp_path) == tmp_path / 'cases.txt'
    assert topup.cases_file_of(tmp_path, 'other.txt').name == 'other.txt'


def test_last_call_stamp(tmp_path):
    pytest.importorskip("fcntl")
    from hpc import topup

    stamp = tmp_path / 'topup.last'
    assert topup.last_call(stamp) == 0.0
    stamp.write_text("")  # stamp of the earlier version (touched, empty)
    assert topup.last_call(stamp) == 0.0
    stamp.write_text("1791100000.250\n")
    assert topup.last_call(stamp) == 1791100000.25
