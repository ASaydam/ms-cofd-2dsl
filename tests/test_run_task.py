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
