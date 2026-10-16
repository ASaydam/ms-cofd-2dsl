import pytest

pytest.importorskip("vtk")
from hpc.run_task import partial_time  # noqa: E402

CLOSED = "FoamFile {}\ninternalField uniform 0;\n\n\n// ************************************************************************* //\n"


def write_time(case, name, fields):
    (case / name).mkdir(parents=True)
    for field, text in fields.items():
        (case / name / field).write_text(text)


def test_partial_time_finds_a_time_step_cut_off_while_written(tmp_path):
    write_time(tmp_path, '0', {'U': CLOSED, 'p': CLOSED})
    assert partial_time(tmp_path) is None  # nothing written yet
    write_time(tmp_path, '17800', {'U': CLOSED, 'p': CLOSED, 'phi': CLOSED})
    assert partial_time(tmp_path) is None
    write_time(tmp_path, '17900', {'U': CLOSED, 'phi': "FoamFile {}\ninternalField nonuniform List<scalar>\n1e-07\n"})
    assert partial_time(tmp_path) == '17900'  # p missing, phi cut off
    (tmp_path / '17900' / 'p').write_text(CLOSED)
    assert partial_time(tmp_path) == '17900'  # phi cut off
    (tmp_path / '17900' / 'phi').write_text(CLOSED)
    assert partial_time(tmp_path) is None


def test_finished_record_only_for_complete_results(tmp_path):
    import json
    from hpc.run_task import finished_record

    path = tmp_path / 'case_provenance.json'
    assert finished_record(tmp_path, 'case') is None  # not run yet
    record = {'status': 'converged', 'iterations': '798', 'slurm_job_id': '1', 'timings_s': {'archive': 1.0}}
    path.write_text(json.dumps(record))
    assert finished_record(tmp_path, 'case') is None  # stopped before the upload
    record['timings_s']['upload'] = 2.0
    path.write_text(json.dumps(record))
    assert finished_record(tmp_path, 'case', 50000)['iterations'] == '798'
    record.update(status='end_time', iterations='10000')
    path.write_text(json.dumps(record))
    assert finished_record(tmp_path, 'case') is not None  # at its cap, not raised
    assert finished_record(tmp_path, 'case', 10000) is not None
    assert finished_record(tmp_path, 'case', 30000) is None  # cap raised since
