import csv
import errno
import hashlib
import json
import sys
import types
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.upload_raw_archives as raw
import tools.upload_release as upload

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO = 'someone/ms-cofd-2dsl'
FILES = {'0.25x/train.h5': 300, '0.25x/test.h5': 120, '16x/train.h5': 900, '16x/test.h5': 400, '4x/test.h5': 200,
         'DATACARD.md': 50, 'metadata.parquet': 80, 'dataset_info.json': 30}
ORDER = ['DATACARD.md', 'dataset_info.json', 'metadata.parquet', '0.25x/test.h5', '0.25x/train.h5', '4x/test.h5',
         '16x/test.h5', '16x/train.h5']                       # top level first, then the runs from the smallest grid


class FakeHub:
    """huggingface_hub with one repository in memory: HfApi and CommitOperationAdd"""

    def __init__(self):
        self.files, self.commits, self.events, self.raise_next = {}, [], [], []
        self.exists, self.private, self.token = False, None, None
        hub = self

        class HfApi:
            def __init__(self, token=None):
                hub.token = token

            def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False):
                assert repo_id == REPO and repo_type == 'dataset' and exist_ok
                if not hub.exists:
                    hub.exists, hub.private = True, private
                    hub.events.append(('create_repo', private))

            def repo_info(self, repo_id, repo_type=None):
                assert hub.exists and repo_type == 'dataset'
                return SimpleNamespace(private=hub.private)

            def update_repo_settings(self, repo_id, private=None, repo_type=None):
                hub.private = private
                hub.events.append(('update_repo_settings', private))

            def list_repo_tree(self, repo_id, repo_type=None, recursive=False):
                assert repo_id == REPO and repo_type == 'dataset' and recursive
                return [hub.entry(path) for path in sorted(hub.files)] + [SimpleNamespace(path='16x', tree_id='0')]

            def create_commit(self, repo_id, operations, commit_message, repo_type=None):
                assert repo_id == REPO and repo_type == 'dataset'
                if hub.raise_next:
                    raise hub.raise_next.pop(0)
                for operation in operations:
                    hub.files[operation.path_in_repo] = operation.data()
                hub.commits.append((commit_message, [o.path_in_repo for o in operations]))
                hub.events.append(('create_commit', tuple(o.path_in_repo for o in operations)))
                return SimpleNamespace(oid=f"{len(hub.commits):040x}")

        class CommitOperationAdd:
            def __init__(self, path_in_repo, path_or_fileobj):
                self.path_in_repo, self.source = path_in_repo, path_or_fileobj
                data = self.data()
                self.upload_info = SimpleNamespace(sha256=hashlib.sha256(data).digest(), size=len(data))

            def data(self):
                return self.source if isinstance(self.source, bytes) else Path(self.source).read_bytes()

        self.module = types.ModuleType('huggingface_hub')
        self.module.HfApi, self.module.CommitOperationAdd = HfApi, CommitOperationAdd

    def entry(self, path):
        """a file of the listing: data files with their LFS size and SHA-256, the card and checksums with a blob id"""
        data = self.files[path]
        if path not in ('README.md', 'checksums.md5', 'DATACARD.md', 'dataset_info.json'):
            lfs = SimpleNamespace(size=len(data), sha256=hashlib.sha256(data).hexdigest(), pointer_size=134)
            return SimpleNamespace(path=path, size=len(data), blob_id='f' * 40, lfs=lfs)
        return SimpleNamespace(path=path, size=len(data), lfs=None,
                               blob_id=hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest())


@pytest.fixture
def hub(monkeypatch):
    fake = FakeHub()
    monkeypatch.setitem(sys.modules, 'huggingface_hub', fake.module)
    monkeypatch.setattr(raw.time, 'sleep', lambda seconds: None)
    return fake


def make_release(tmp_path):
    """a small release folder with checksums.md5 in md5sum's format, and a card"""
    release = tmp_path / 'release'
    lines = []
    for path, size in sorted(FILES.items()):
        data = (path.encode() + b'\n') * size
        (release / path).parent.mkdir(parents=True, exist_ok=True)
        (release / path).write_bytes(data)
        lines.append(f"{hashlib.md5(data).hexdigest()}  {path}\n")
    (release / 'checksums.md5').write_text(''.join(lines))
    card = tmp_path / 'README.md'
    card.write_bytes(b"---\nlicense: cc-by-4.0\n---\n\n# MS-COFD-2DSL\n")
    return release, card, hashlib.md5((release / 'checksums.md5').read_bytes()).hexdigest()


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def snapshot(folder):
    return {p.relative_to(folder): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(folder.rglob('*'))}


def files_in(folder):
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file())


def test_read_checksums(tmp_path):
    release, _, md5 = make_release(tmp_path)
    data, rows = upload.read_checksums(release, md5)
    assert data == (release / 'checksums.md5').read_bytes()
    assert [r['path'] for r in rows] == ORDER
    assert {r['path']: r['bytes'] for r in rows} == {p: (release / p).stat().st_size for p in FILES}
    assert {r['path']: r['run'] for r in rows}['16x/test.h5'] == '16x' and rows[0]['run'] == ''
    with pytest.raises(ValueError, match='not this release'):
        upload.read_checksums(release, '0' * 32)
    (release / '4x' / 'test.h5').unlink()
    with pytest.raises(ValueError, match='not in'):
        upload.read_checksums(release)
    (release / 'checksums.md5').write_text(f"{'1' * 32}  ../outside.h5\n")
    with pytest.raises(ValueError, match='unexpected path'):
        upload.read_checksums(release)


def test_upload(tmp_path, hub):
    release, card, md5 = make_release(tmp_path)
    output, staging, before = tmp_path / 'out', tmp_path / 'stage', snapshot(release)
    problems = upload.run(release, REPO, output, card, md5, batch=2, batch_bytes=1e9, workers=2, limit=3,
                          token='hf_secret', staging=staging)
    assert hub.token == 'hf_secret'
    # created private, the card and checksums first
    assert hub.events[:2] == [('create_repo', True), ('create_commit', ('README.md', 'checksums.md5'))]
    assert hub.commits[0][0] == "Dataset card, licence and checksums"
    assert hub.private is True and {p['problem'] for p in problems} == {'missing'}
    assert [paths for _, paths in hub.commits[1:]] == [['DATACARD.md', 'dataset_info.json'], ['metadata.parquet']]
    assert hub.commits[1][0] == "Add 2 top-level files" and hub.commits[2][0] == "Add metadata.parquet"

    problems = upload.run(release, REPO, output, card, md5, batch=2, batch_bytes=1e9, workers=2, staging=staging)
    # the whole release is in the repository, and it stays private (made public by hand)
    assert problems == [] and hub.private is True and 'update_repo_settings' not in {e for e, _ in hub.events}
    assert hub.commits[3] == ("Add 2 files of 0.25x/", ['0.25x/test.h5', '0.25x/train.h5'])
    assert set(hub.files) == {'README.md', 'checksums.md5'} | set(FILES)
    for path in FILES:
        assert hub.files[path] == (release / path).read_bytes()
    assert hub.files['checksums.md5'] == (release / 'checksums.md5').read_bytes()
    done = read_csv(output / 'done.csv')
    commits = {f"{k:040x}": paths for k, (_, paths) in enumerate(hub.commits, start=1)}
    assert sorted(r['path'] for r in done) == sorted(FILES)
    for r in done:
        assert r['md5_ok'] == 'True' and r['sha256'] == hashlib.sha256(hub.files[r['path']]).hexdigest()
        assert r['path'] in commits[r['commit']]
    assert read_csv(output / 'verify.csv') == [] and not (output / 'errors.csv').exists()

    count = len(hub.commits)
    upload.run(release, REPO, output, card, md5, batch=2, batch_bytes=1e9, workers=2, staging=staging)  # nothing to do
    assert len(hub.commits) == count
    card.write_bytes(card.read_bytes() + b"\nMore text.\n")
    upload.run(release, REPO, output, card, md5, batch=2, batch_bytes=1e9, workers=2, staging=staging)  # edited card
    assert hub.commits[-1] == ("Update README.md", ['README.md']) and hub.files['README.md'] == card.read_bytes()

    assert upload.run(release, REPO, output, card, md5, verify_only=True) == []
    assert snapshot(release) == before                       # nothing in the release folder changed
    assert files_in(staging) == []
    assert not any(b'hf_secret' in p.read_bytes() for p in output.iterdir())


def test_large_file_alone(tmp_path, hub):
    release, card, md5 = make_release(tmp_path)
    upload.run(release, REPO, tmp_path / 'out', card, md5, batch=20, batch_bytes=2000, workers=2,
               staging=tmp_path / 'stage')
    big = (release / '16x' / 'train.h5').stat().st_size
    assert big > 2000 and ("Add 16x/train.h5", ['16x/train.h5']) in hub.commits


def test_md5_mismatch(tmp_path, hub):
    """a file that differs from checksums.md5 is not uploaded, and the repository stays private"""
    release, card, md5 = make_release(tmp_path)
    path = release / '0.25x' / 'test.h5'
    path.write_bytes(path.read_bytes()[:-1] + b'x')
    output = tmp_path / 'out'
    problems = upload.run(release, REPO, output, card, md5, batch=2, workers=2, staging=tmp_path / 'stage')
    assert '0.25x/test.h5' not in hub.files and hub.private is True
    assert [r['path'] for r in read_csv(output / 'errors.csv')] == ['0.25x/test.h5']
    assert 'does not match checksums.md5' in read_csv(output / 'errors.csv')[0]['error']
    assert problems == [{'path': '0.25x/test.h5', 'problem': 'missing', 'detail': ''}]


def test_wrong_release(tmp_path, hub):
    release, card, _ = make_release(tmp_path)
    with pytest.raises(ValueError, match='not this release'):
        upload.run(release, REPO, tmp_path / 'out', card, md5='0' * 32, staging=tmp_path / 'stage')
    assert hub.events == []


def test_staging_space(tmp_path, hub, monkeypatch):
    """a batch that does not fit into the free space stops the run before it is copied"""
    release, card, md5 = make_release(tmp_path)
    usage = namedtuple('usage', 'total used free')
    monkeypatch.setattr(upload.shutil, 'disk_usage', lambda path: usage(10 ** 12, 0, upload.SPACE_MARGIN + 500))
    with pytest.raises(ValueError, match='GB free'):
        upload.run(release, REPO, tmp_path / 'out', card, md5, batch=20, batch_bytes=1e9, workers=2,
                   staging=tmp_path / 'stage')
    assert hub.commits[0][1] == ['README.md', 'checksums.md5'] and len(hub.commits) == 1
    assert files_in(tmp_path / 'stage') == []


def test_found_in_repository(tmp_path, hub):
    """a file committed without a row in done.csv is recorded without a second upload"""
    release, card, md5 = make_release(tmp_path)
    hub.exists, hub.private = True, True
    for path in ('16x/test.h5', 'DATACARD.md'):                # an LFS file and a small file stored without LFS
        hub.files[path] = (release / path).read_bytes()
    output = tmp_path / 'out'
    assert upload.run(release, REPO, output, card, md5, batch=2, workers=2, staging=tmp_path / 'stage') == []
    for path in ('16x/test.h5', 'DATACARD.md'):
        assert not any(path in paths for _, paths in hub.commits)
        row = next(r for r in read_csv(output / 'done.csv') if r['path'] == path)
        assert row['commit'] == '' and row['md5_ok'] == 'True'
    assert hub.private is True


def test_drive_lost(tmp_path, hub, monkeypatch):
    """a dropped mount stops the run before the batch is committed; the batch is copied again next time"""
    release, card, md5 = make_release(tmp_path)
    output, staging = tmp_path / 'out', tmp_path / 'stage'
    original = raw.HashingReader

    class Dropping(original):
        def read(self, size=-1):
            if self.bytes and Path(self.raw.name).parts[-2:] == ('16x', 'train.h5'):
                raise OSError(errno.ENOTCONN, 'Transport endpoint is not connected')
            return super().read(size)
    monkeypatch.setattr(raw, 'CHUNK', 64)
    monkeypatch.setattr(raw, 'HashingReader', Dropping)
    with pytest.raises(upload.DriveLost):
        upload.run(release, REPO, output, card, md5, batch=10, workers=2, staging=staging)
    assert '16x/train.h5' not in hub.files and files_in(staging) == [] and hub.private is True
    monkeypatch.setattr(raw, 'HashingReader', original)
    assert upload.run(release, REPO, output, card, md5, batch=10, workers=2, staging=staging) == []
    assert len(read_csv(output / 'done.csv')) == len(FILES) and hub.private is True


def test_compare():
    """the verify comparison on a faked listing"""
    def stored(path, size, sha256):
        return SimpleNamespace(path=path, size=size, blob_id='f' * 40,
                               lfs=SimpleNamespace(size=size, sha256=sha256, pointer_size=134))
    card, checksums = b"# card\n", b"m  a.h5\n"
    rows = [{'path': f"16x/{k}.h5", 'run': '16x', 'md5': 'm', 'bytes': 100} for k in range(1, 6)]
    done = {f"16x/{k}.h5": {'bytes': '100', 'sha256': 'a' * 64} for k in (1, 3, 4)}
    tree = {'16x/1.h5': stored('16x/1.h5', 100, 'a' * 64), '16x/3.h5': stored('16x/3.h5', 100, 'b' * 64),
            '16x/4.h5': stored('16x/4.h5', 99, 'a' * 64), '16x/5.h5': stored('16x/5.h5', 100, 'c' * 64),
            '16x/6.h5': stored('16x/6.h5', 100, 'd' * 64),
            '.gitattributes': SimpleNamespace(path='.gitattributes', size=10, blob_id='0' * 40, lfs=None),
            'README.md': SimpleNamespace(path='README.md', size=len(card), lfs=None,
                                         blob_id=hashlib.sha1(b'blob %d\0' % len(card) + card).hexdigest())}
    problems = upload.compare(tree, rows, done, {'README.md': card, 'checksums.md5': checksums})
    assert {(p['path'], p['problem']) for p in problems} == {
        ('checksums.md5', 'missing'), ('16x/2.h5', 'missing'), ('16x/3.h5', 'mismatch'), ('16x/4.h5', 'mismatch'),
        ('16x/5.h5', 'unchecked'), ('16x/6.h5', 'extra')}

    # small files the Hub stores without LFS: compared by their git blob id with the release file
    release = {'DATACARD.md': b"# card\n", 'dataset_info.json': b"{}\n"}
    small = {path: SimpleNamespace(path=path, size=len(data), lfs=None,
                                   blob_id=hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest())
             for path, data in release.items()}
    rows = [{'path': path, 'run': '', 'md5': 'm', 'bytes': len(data)} for path, data in release.items()]
    done = {path: {'bytes': str(len(data)), 'sha256': 'a' * 64} for path, data in release.items()}
    assert upload.compare(small, rows, done, {}, content=release.get) == []
    assert [p['problem'] for p in upload.compare(small, rows, done, {})] == ['mismatch', 'mismatch']   # no content
    changed = dict(release, **{'dataset_info.json': b"[]\n"})
    assert upload.compare(small, rows, done, {}, content=changed.get) == [
        {'path': 'dataset_info.json', 'problem': 'mismatch', 'detail': "git blob differs from the release file"}]


def test_notebook_copies():
    """the scripts written by the Colab notebook are those of the repository"""
    notebook = json.loads((REPO_ROOT / 'tools' / 'colab' / 'upload_release.ipynb').read_text(encoding='utf-8'))
    written = {}
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if source.startswith('%%writefile '):
            first, _, body = source.partition('\n')
            written[first.split()[1]] = body
    assert sorted(written) == ['tools/extract_line_samples.py', 'tools/upload_raw_archives.py',
                               'tools/upload_release.py']
    for name in written:
        assert written[name].rstrip('\n') == (REPO_ROOT / name).read_text(encoding='utf-8').rstrip('\n'), name


def test_card_front_matter():
    """the card has the licence and a config for every table it names, and points to files of the release"""
    text = (REPO_ROOT / 'docs' / 'hf' / 'release_README.md').read_text(encoding='utf-8')
    front = text.split('---\n')[1]
    assert 'license: cc-by-4.0' in front
    for name in ('metadata.parquet', 'label_uncertainty.parquet', 'mesh_study.parquet', 'surface.parquet',
                 'dropped_cases.csv'):
        assert f"data_files: {name}" in front and name in text.split('---\n', 2)[2]
