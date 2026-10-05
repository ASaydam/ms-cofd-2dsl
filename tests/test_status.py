import math
import os
from datetime import datetime

from hpc import ledger, publish_status, status, topup


def fake_root(tmp_path):
    (tmp_path / 'geometry').mkdir()
    names = [f"normal_circle_{i:04d}" for i in (1, 2, 3)]
    (tmp_path / 'geometry' / 'manifest.csv').write_text(
        "case_name,size,geometry,index\n" + ''.join(f"{n},normal,circle,{i}\n" for i, n in enumerate(names, 1)))
    ledger.set_state('base_lu', names[0], 'done', 'converged 900 1', root=tmp_path)
    ledger.set_state('base_lu', names[1], 'done', 'end_time 10000 2', root=tmp_path)
    ledger.set_state('base_lu', names[2], 'running', '3 be01', root=tmp_path)
    ledger.set_state('16x', names[0], 'failed', 'RuntimeError: gmsh failed', root=tmp_path)
    (tmp_path / 'results' / 'base_lu').mkdir(parents=True)
    (tmp_path / 'results' / 'base_lu' / 'summary.csv').write_text(
        "case,status,execution_time_s,finished\n"
        f"{names[0]},converged,180,2026-10-05T09:30:00+0300\n{names[1]},end_time,2160,2026-10-05T02:00:00+0300\n")
    return names


def test_collect_and_history_row(tmp_path):
    names = fake_root(tmp_path)
    slurm, dataset = topup.load_settings(tmp_path)
    jobs = {('base_lu', names[2]): ('3', 'RUNNING', 'asaydam_oct26')}
    now = datetime.fromisoformat("2026-10-05T10:00:00+03:00")
    result = status.collect(tmp_path, slurm, dataset, jobs=jobs, nodes={'be01': 1}, used=100.0, backlog=0.0, now=now,
                            running=[])

    runs = {r['run']: r for r in result['runs']}
    assert result['unsteady'] == [names[1]] and result['cases'] == 2           # the unsteady case is dropped
    assert runs['base_lu']['done'] == 1 and runs['base_lu']['running'] == 1 and runs['base_lu']['target'] == 2
    assert runs['16x']['target'] == 2 and runs['16x']['failed'] == 1 and result['reserves'] == 0
    assert runs['base_lu']['done_1h'] == 1 and runs['base_lu']['done_24h'] == 2
    assert runs['base_lu']['core_hours'] == 0.65 and runs['base_lu']['end_time'] == 1
    assert result['failures'][0]['case'] == names[0] and result['interrupted'] == 0

    row = publish_status.history_row(result)
    assert row['done_base_lu'] == 1 and row['unsteady'] == 1 and row['running'] == 1


def test_parse_time():
    assert status.parse_time("2026-10-05T09:30:00+0300") == datetime.fromisoformat("2026-10-05T09:30:00+03:00")
    assert status.parse_time("") is None


def test_seconds():
    assert status.seconds("1-02:03:04") == 93784 and status.seconds("2:03:04") == 7384
    assert status.seconds("5:07") == 307 and status.seconds("UNLIMITED") is None


def write_log(folder, iterations, start=1, residual=lambda i: 10 ** (-2 - i / 1000), seconds=2.0, end=False):
    """a simpleFoam log: the largest residual (Uy) falls by one decade per 1000 iterations, 2 s per iteration"""
    lines = []
    for i in range(start, start + iterations):
        lines += [f"Time = {i}", "",
                  f"smoothSolver:  Solving for Ux, Initial residual = {residual(i) / 3:.6g}, Final residual = 1e-9, No Iterations 9",
                  f"smoothSolver:  Solving for Uy, Initial residual = {residual(i):.6g}, Final residual = 1e-9, No Iterations 9",
                  f"GAMG:  Solving for p, Initial residual = {residual(i) / 10:.6g}, Final residual = 1e-9, No Iterations 8",
                  f"GAMG:  Solving for p, Initial residual = {residual(i) / 100:.6g}, Final residual = 1e-9, No Iterations 1",
                  "time step continuity errors : sum local = 1e-09, global = 1e-11, cumulative = 1e-07",
                  f"ExecutionTime = {seconds * (i - start + 1):.1f} s  ClockTime = {seconds * (i - start + 1):.0f} s", ""]
    lines += ["SIMPLE solution converged in 0 iterations", "", "End", ""] if end else []
    (folder / 'log.simpleFoam').write_text("\n".join(lines) + "\n")


def test_job_progress(tmp_path):
    folder = tmp_path / 'case'
    now = datetime.fromisoformat("2026-10-05T10:00:00+03:00")
    assert status.job_progress(folder, mapped=True, now=now)['stage'] == 'starting'
    folder.mkdir()
    (folder / 'log.gmsh').write_text('')
    assert status.job_progress(folder, mapped=True, now=now)['stage'] == 'meshing'
    (folder / 'mesh_variant').write_text('0\n')
    assert status.job_progress(folder, mapped=True, now=now)['stage'] == 'mapping'
    assert status.job_progress(folder, mapped=False, now=now)['stage'] == 'starting'

    (folder / 'mapped').write_text('seed\n')
    (folder / 'system').mkdir()
    (folder / 'system' / 'controlDict').write_text("endTime         15000;  // cap\nwriteInterval   100;\n")
    write_log(folder, 2000)
    os.utime(folder / 'log.simpleFoam', (now.timestamp() - 30, now.timestamp() - 30))
    progress = status.job_progress(folder, mapped=True, now=now)
    assert progress['stage'] == 'solving' and progress['iteration'] == 2000 and progress['cap'] == 15000
    assert math.isclose(progress['residual'], 1e-4, rel_tol=1e-4) and progress['residuals'][1] == progress['residual']
    assert progress['s_per_iteration'] == 2.0 and progress['log_age_s'] == 30
    assert math.isclose(progress['decades_per_1000'], -1.0, rel_tol=1e-3)
    assert abs(progress['eta_iterations'] - 2000) <= 2 and abs(progress['eta_s'] - 4000) <= 4

    # a continued run: the execution time starts again, the trend comes from the new run only
    write_log(folder, 300, start=2001, residual=lambda i: 1e-5, seconds=3.0)
    progress = status.job_progress(folder, mapped=True, now=now)
    assert progress['iteration'] == 2300 and progress['s_per_iteration'] == 3.0
    assert progress['decades_per_1000'] == 0 and 'eta_s' not in progress           # flat: no estimate

    write_log(folder, 50, end=True)
    assert status.job_progress(folder, mapped=True, now=now)['stage'] == 'finishing'


def test_collect_running_jobs(tmp_path):
    names = fake_root(tmp_path)
    slurm, dataset = topup.load_settings(tmp_path)
    running = [{'job': '3', 'run': 'base_lu', 'case': names[2], 'node': 'be01', 'elapsed_s': 60, 'limit_s': 7200},
               {'job': '9', 'run': '16x', 'case': names[1], 'node': 'be02', 'elapsed_s': 60, 'limit_s': 43200}]
    now = datetime.fromisoformat("2026-10-05T10:00:00+03:00")
    result = status.collect(tmp_path, slurm, dataset, jobs={}, used=100.0, backlog=0.0, now=now, running=running)
    assert [j['job'] for j in result['jobs']] == ['3']                  # job 9 is not in this ledger (a pilot)
    assert result['jobs'][0]['pace'] == 'starting' and result['nodes'] == {'be01': 1, 'be02': 1}
    assert [f['case'] for f in result['finished']] == [names[0], names[1]]          # newest first
    assert result['finished'][0]['hours'] == 0.05 and result['finished'][1]['status'] == 'end_time'


def test_pace():
    job = {'stage': 'solving', 'log_age_s': 10, 'iteration': 1000, 'cap': 15000, 'decades_per_1000': 0.2}
    assert status.pace(job) == 'warming up'                                  # rising early after mapFields
    assert status.pace({**job, 'iteration': 4000}) == 'flat'
    assert status.pace({**job, 'iteration': 4000, 'decades_per_1000': -0.5, 'eta_iterations': 2000}) == 'ok'
    assert status.pace({**job, 'iteration': 4000, 'decades_per_1000': -0.1, 'eta_iterations': 14000}) == 'cap'
    assert status.pace({**job, 'log_age_s': 3600}) == 'frozen'
    assert status.pace({'stage': 'meshing'}) == 'meshing'
    assert status.pace({'stage': 'solving', 'log_age_s': 0}) == 'ok'          # no trend yet
