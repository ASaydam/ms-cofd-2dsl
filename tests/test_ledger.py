import os
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

    states = {('base_lu', cases[0]): ('done', 'converged 900 123')}
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
    from hpc import topup

    slurm, _ = topup.load_settings()
    inside = datetime.fromisoformat("2026-10-10T12:00:00+03:00")
    after = datetime.fromisoformat("2026-10-19T00:00:01+03:00")
    assert topup.current_window(slurm, inside)['qos'] == 'asaydam_oct26'
    assert topup.current_window(slurm, after)['qos'] == 'users'


def test_data_root_overrides(tmp_path):
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
    from hpc import topup

    stamp = tmp_path / 'topup.last'
    assert topup.last_call(stamp) == 0.0
    stamp.write_text("")  # stamp of the earlier version (touched, empty)
    assert topup.last_call(stamp) == 0.0
    stamp.write_text("1791100000.250\n")
    assert topup.last_call(stamp) == 1791100000.25


def test_select_tasks_lists_a_task_once():
    from hpc import topup

    cases = ledger.case_order(manifest(1))
    first, second = cases[0], cases[1]
    states = {('base_lu', first): ('done', 'converged 900 1'), ('base_lu', second): ('done', 'converged 800 2'),
              ('16x', first): ('failed', 'gmsh failed'),         # failed, submissions left
              ('4x', first): ('running', '120 be01'),           # its job is gone: interrupted
              ('16x', second): ('submitted', '121 attempt=1')}  # its job is in the queue
    jobs = {('16x', second): ('121', 'PENDING', 'users')}
    retry, ready, exhausted = topup.select_tasks(states, jobs, cases, RUNS, lambda run, case: 1, 3)
    assert sorted(retry) == [('16x', first), ('4x', first)]
    assert not set(retry) & set(ready)
    assert ('16x', second) not in ready and ('4x', second) in ready
    assert exhausted == []

    retry, ready, exhausted = topup.select_tasks(states, jobs, cases, RUNS, lambda run, case: 3, 3)
    assert retry == [] and exhausted == [('4x', first, '120 be01')]

    # held tasks are neither submitted again nor started, nor marked failed
    held = {('16x', first), ('4x', first), ('4x', second)}
    retry, ready, exhausted = topup.select_tasks(states, jobs, cases, RUNS, lambda run, case: 1, 3, held=held)
    assert retry == [] and not held & set(ready) and ('base_fo', second) in ready
    retry, ready, exhausted = topup.select_tasks(states, jobs, cases, RUNS, lambda run, case: 3, 3, held=held)
    assert exhausted == []


def test_task_lists_of_the_data_root(tmp_path):
    (tmp_path / 'hold.txt').write_text("# 0.25 mm runs of cases whose 2 mm run did not converge\n"
                                       "16x normal_left_triangle_0271\n\n16x larger_circle_0279  # flat\n")
    assert ledger.held_tasks(tmp_path) == {('16x', 'normal_left_triangle_0271'), ('16x', 'larger_circle_0279')}
    assert ledger.held_tasks(tmp_path / 'missing') == set()

    (tmp_path / 'case_settings.txt').write_text("0.25x normal_left_triangle_0271 templates second-order-relaxed\n"
                                                "0.25x normal_left_triangle_0271 init_from base_lu\n"
                                                "0.25x larger_circle_0279 fvSchemes linear\n")
    assert ledger.case_settings('0.25x', 'normal_left_triangle_0271', tmp_path) == \
        {'templates': 'second-order-relaxed', 'init_from': 'base_lu'}
    assert ledger.case_settings('16x', 'normal_left_triangle_0271', tmp_path) == {}
    with pytest.raises(ValueError):  # only solver settings and initial fields, never the discretisation
        ledger.case_settings('0.25x', 'larger_circle_0279', tmp_path)


def test_directory_lock(tmp_path):
    path = tmp_path / 'topup.lockdir'
    with ledger.DirectoryLock(path):
        assert (path / 'holder').exists()
        with pytest.raises(FileExistsError):
            path.mkdir()  # what a second holder would try
    assert not path.exists()

    path.mkdir()  # left by a holder that was killed
    old = 1_000_000_000
    os.utime(path, (old, old))
    with ledger.DirectoryLock(path, stale_s=600):
        assert (path / 'holder').exists()
    assert not path.exists() and not list(tmp_path.glob('*.stale.*'))


def test_excluded_nodes(tmp_path):
    from hpc import topup

    slurm, _ = topup.load_settings(tmp_path)
    assert topup.excluded_nodes(tmp_path, slurm) == ['dy02', 'dy03']
    (tmp_path / 'state').mkdir()
    (tmp_path / 'state' / 'bad_nodes').write_text("be09 2026-10-05T03:00:00+0300 FATAL: ...\n"
                                                    "da01 2026-10-05T08:53:33+0300 error\ndy02 2026-10-05\n")
    now = datetime.fromisoformat("2026-10-05T10:00:00+03:00")
    assert topup.excluded_nodes(tmp_path, slurm, now) == ['da01', 'dy02', 'dy03']  # be09: 7 h ago, tried again


def test_dependencies_from_the_run_definitions():
    runs = {'base_lu': {'resolution': 'base'}, '16x': {'resolution': '16x', 'init_from': 'base_lu'},
            '16x_scratch': {'resolution': '16x'}}
    dependencies = ledger.dependencies_of(runs)
    assert dependencies == {'16x': 'base_lu'}
    cases = ledger.case_order(manifest(1))
    ready = ledger.ready_tasks(cases, list(runs), {}, dependencies)
    assert ('16x_scratch', cases[0]) in ready and ('16x', cases[0]) not in ready
    assert ready[-1][0] == '16x_scratch'  # runs that are not in RUN_PRIORITY come last


def test_data_root_dataset_overlay(tmp_path, monkeypatch):
    import utils

    monkeypatch.setenv('MSCOFD_DATA_ROOT', str(tmp_path))
    assert 'init_from' in utils.load_config()['runs']['16x']
    (tmp_path / 'dataset.yaml').write_text("runs:\n  16x_scratch: {resolution: 16x, templates: second-order}\n")
    config = utils.load_config()
    assert list(config['runs']) == ['16x_scratch']
    assert config['paths']['manifest'] == tmp_path / 'geometry' / 'manifest.csv'


def test_no_fine_runs_without_a_steady_solution():
    cases = ledger.case_order(manifest(1))
    states = {('base_lu', cases[0]): ('done', 'end_time 10000 123'), ('base_lu', cases[1]): ('done', 'converged 800 124')}
    ready = ledger.ready_tasks(cases, RUNS, states)
    assert ('16x', cases[0]) not in ready and ('4x', cases[0]) not in ready
    assert ('16x', cases[1]) in ready and ('base_fo', cases[0]) in ready
    assert ledger.unsteady_cases(states) == [cases[0]]


def test_gate_and_discarded_cases():
    cases = ledger.case_order(manifest(1))
    states = {('base_lu', cases[0]): ('done', 'converged 900 1'), ('base_lu', cases[1]): ('done', 'end_time 10000 2')}
    discarded = ledger.unsteady_cases(states, ['base_lu'])
    assert discarded == [cases[1]]
    ready = ledger.ready_tasks(cases, RUNS, states, gate='base_lu', discarded=discarded)
    assert ('base_fo', cases[0]) in ready and ('16x', cases[0]) in ready
    assert not [task for task in ready if task[1] == cases[1]]                    # nothing of a discarded case
    assert ('base_fo', cases[2]) not in ready and ('base_lu', cases[2]) in ready  # waits for its base_lu


def test_reserve_cases_replace_discarded_ones():
    rows = [{'case_name': f"normal_circle_{i:04d}", 'size': 'normal', 'geometry': 'circle', 'index': str(i)}
            for i in range(1, 7)]                                                  # 4 cases + 2 reserves
    per_class = {'normal': 4}
    assert ledger.active_cases(rows, [], per_class) == [f"normal_circle_{i:04d}" for i in range(1, 5)]
    active = ledger.active_cases(rows, ['normal_circle_0002'], per_class)
    assert active == [f"normal_circle_{i:04d}" for i in range(1, 6)]               # 0005 replaces 0002
    active = ledger.active_cases(rows, ['normal_circle_0002', 'normal_circle_0005'], per_class)
    assert active[-1] == 'normal_circle_0006' and len(active) == 6                  # a discarded reserve as well


def test_run_priority_from_the_settings():
    cases = ledger.case_order(manifest(1))
    states = {('base_lu', cases[0]): ('done', 'converged 900 1')}
    ready = ledger.ready_tasks(cases, RUNS, states, priority=['base_lu', '0.25x', 'base_fo', '16x', '4x'])
    order = [run for run, case in ready if case == cases[0]]
    assert order == ['base_fo', '16x', '4x']
