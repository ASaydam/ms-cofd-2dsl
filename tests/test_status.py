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
    result = status.collect(tmp_path, slurm, dataset, jobs=jobs, nodes={'be01': 1}, used=100.0, backlog=0.0, now=now)

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
