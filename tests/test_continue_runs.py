import io
import tarfile

import pytest

from hpc import continue_runs


def test_set_cap_replaces_the_line_of_the_task(tmp_path):
    (tmp_path / 'iterations.txt').write_text("base_lu a 25000\n16x b 25000\n")
    continue_runs.set_cap(tmp_path, '16x', 'b', 45000)
    continue_runs.set_cap(tmp_path, '16x', 'c', 45000)
    assert (tmp_path / 'iterations.txt').read_text() == "base_lu a 25000\n16x b 45000\n16x c 45000\n"


def test_restore_only_the_case_folder(tmp_path):
    zstandard = pytest.importorskip("zstandard")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as tar:
        for name, text in (('case/system/controlDict', b'endTime 15000;'), ('case/15000/U', b'u'),
                           ('case_attributes/out.csv', b'x')):
            info = tarfile.TarInfo(name)
            info.size = len(text)
            tar.addfile(info, io.BytesIO(text))
    archive = tmp_path / 'case.tar.zst'
    archive.write_bytes(zstandard.ZstdCompressor().compress(buffer.getvalue()))
    work = tmp_path / 'work'
    work.mkdir()
    assert continue_runs.restore(archive, work, 'case') == 2
    assert (work / 'case' / '15000' / 'U').read_bytes() == b'u' and not (work / 'case_attributes').exists()
