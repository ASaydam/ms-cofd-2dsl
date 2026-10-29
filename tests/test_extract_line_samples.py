import csv
import hashlib
import io
import tarfile

import numpy as np
import pytest


def line_text(n, offset):
    """a raw OpenFOAM set: y, p, Ux, Uy, Uz"""
    y = np.linspace(0, 0.12, n)
    return ''.join(f"{v} {offset} {offset + v} {-v} 0\n" for v in y).encode()


def make_archive(path, case, sets, time_name='364'):
    """a .tar.zst like run_task.py writes: the case folder first, then the attribute work folder"""
    zstandard = pytest.importorskip("zstandard")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as tar:
        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        add(f"{case}/system/controlDict", b"x" * 5000)
        for name, text in sets.items():
            add(f"{case}_attributes/postProcessing/lines/{time_name}/{name}_p_U.xy", text)
        add(f"{case}_attributes/postProcessing/obstacleShear/{time_name}/wallShearStress.dat", b"y" * 100)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zstandard.ZstdCompressor(level=3).compress(buffer.getvalue()))
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_extract_line_samples(tmp_path):
    from tools.extract_line_samples import SETS, read_archive, run
    archives, output = tmp_path / 'archives', tmp_path / 'out'
    sets = {name: line_text(2401 if 'Section' in name or name == 'wakeLine' else 1000, k)
            for k, name in enumerate(SETS)}
    rows = []
    for run_name, case in (('16x', 'normal_circle_0001'), ('16x', 'normal_circle_0002'), ('base_lu', 'normal_circle_0001')):
        md5 = make_archive(archives / run_name / f"{case}.tar.zst", case, sets)
        rows.append({'run': run_name, 'case': case, 'file': f"{case}.tar.zst", 'bytes': 0, 'md5': md5,
                     'remote': f"remote:archives/{run_name}/{case}.tar.zst", 'uploaded': ''})
    rows[1]['md5'] = '0' * 32                                                   # a mismatch is recorded, not fatal
    rows.append(dict(rows[0], file='normal_circle_0001.cap10000.tar.zst',           # superseded attempts are not read
                     remote='remote:archives/16x/superseded/normal_circle_0001.cap10000.tar.zst'))
    index = tmp_path / 'archive_index.csv'
    with open(index, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    time_name, found, md5, size = read_archive(archives / '16x' / 'normal_circle_0001.tar.zst')
    assert time_name == '364' and md5 == rows[0]['md5'] and size > 0
    assert found['wakeLine'].shape == (2401, 4) and found['centroidBelow'].shape == (1000, 4)
    assert found['wakeLine'][-1, 0] == pytest.approx(0.12) and found['inletSection'][0, 2] == pytest.approx(3.0)

    staging = tmp_path / 'staging'
    run(archives, index, output, workers=2, shard_size=1, limit=2, staging=staging)
    run(archives, index, output, workers=2, shard_size=2, staging=staging)    # resumes with the third archive only
    with open(output / 'done.csv', newline='') as f:
        done = list(csv.DictReader(f))
    assert [(r['run'], r['case']) for r in done] == [('16x', 'normal_circle_0001'), ('16x', 'normal_circle_0002'),
                                                     ('base_lu', 'normal_circle_0001')]
    assert [r['md5_ok'] for r in done] == ['True', 'False', 'True']
    shard = np.load(output / done[2]['shard'])
    assert str(shard['normal_circle_0001/time']) == '364'
    assert np.array_equal(shard['normal_circle_0001/wakeLine'], found['wakeLine'])
    assert not (output / 'errors.csv').exists()


def test_drive_lost(tmp_path, monkeypatch):
    """a dropped mount stops the run; the finished archives of the shard are kept in staging and copied next time"""
    import errno
    import tools.extract_line_samples as module
    sets = {name: line_text(2401, k) for k, name in enumerate(module.SETS)}
    archives, output, staging = tmp_path / 'archives', tmp_path / 'out', tmp_path / 'staging'
    rows = []
    for case in ('normal_circle_0001', 'normal_circle_0002', 'normal_circle_0003'):
        md5 = make_archive(archives / '16x' / f"{case}.tar.zst", case, sets)
        rows.append({'run': '16x', 'case': case, 'file': f"{case}.tar.zst", 'bytes': 100, 'md5': md5,
                     'remote': f"remote:archives/16x/{case}.tar.zst", 'uploaded': ''})
    index = tmp_path / 'archive_index.csv'
    with open(index, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    read = module.read_archive
    def flaky(path):
        if path.name == 'normal_circle_0002.tar.zst':
            raise OSError(errno.ENOTCONN, 'Transport endpoint is not connected')
        return read(path)
    monkeypatch.setattr(module, 'read_archive', flaky)
    with pytest.raises(module.DriveLost):
        module.run(archives, index, output, workers=1, staging=staging)
    assert not (output / 'done.csv').exists() and len(list(staging.glob('*.csv'))) == 1

    monkeypatch.setattr(module, 'read_archive', read)
    module.run(archives, index, output, workers=1, staging=staging)
    with open(output / 'done.csv', newline='') as f:
        done = sorted(r['case'] for r in csv.DictReader(f))
    assert done == ['normal_circle_0001', 'normal_circle_0002', 'normal_circle_0003']
    assert not list(staging.iterdir()) and not (output / 'errors.csv').exists()


def test_shards_of():
    from tools.extract_line_samples import shards_of
    rows = [{'run': run, 'case': str(k), 'bytes': size} for k, (run, size) in
            enumerate([('4x', 5), ('4x', 5), ('4x', 5), ('16x', 30), ('16x', 1)])]
    assert [len(s) for s in shards_of(rows, shard_size=10, shard_bytes=10)] == [2, 1, 1, 1]
