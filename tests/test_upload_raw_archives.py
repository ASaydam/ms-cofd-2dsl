import csv
import errno
import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.upload_raw_archives as upload

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO = 'someone/ms-cofd-2dsl-raw'
RUNS = ('0.25x', '16x', '4x', 'base_fo', 'base_lu')                  # sorted as the index rows are
CASES = ('normal_circle_0001', 'normal_circle_0002', 'normal_circle_0003')
DROPPED = 'normal_circle_0004'               # its base_lu archive is live, but the case is not in the release


class FakeHub:
    """huggingface_hub with one repository in memory: HfApi and CommitOperationAdd"""

    def __init__(self, settings=True):
        self.files, self.commits, self.events, self.raise_next = {}, [], [], []
        self.exists, self.private, self.token, self.on_commit = False, None, None, None
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

            def update_repo_visibility(self, repo_id, private, repo_type=None):
                hub.private = private
                hub.events.append(('update_repo_visibility', private))

            def list_repo_tree(self, repo_id, repo_type=None, recursive=False):
                assert repo_id == REPO and repo_type == 'dataset' and recursive
                return [hub.entry(path) for path in sorted(hub.files)] + [SimpleNamespace(path='16x', tree_id='0')]

            def create_commit(self, repo_id, operations, commit_message, repo_type=None):
                assert repo_id == REPO and repo_type == 'dataset'
                if hub.raise_next:
                    raise hub.raise_next.pop(0)
                if hub.on_commit:
                    hub.on_commit(operations)
                for operation in operations:
                    hub.files[operation.path_in_repo] = operation.data()
                hub.commits.append((commit_message, [o.path_in_repo for o in operations]))
                hub.events.append(('create_commit', tuple(o.path_in_repo for o in operations)))
                return SimpleNamespace(oid=f"{len(hub.commits):040x}")

        if not settings:                             # a huggingface_hub without update_repo_settings
            del HfApi.update_repo_settings

        class CommitOperationAdd:
            def __init__(self, path_in_repo, path_or_fileobj):
                self.path_in_repo, self.source = path_in_repo, path_or_fileobj
                data = self.data()                       # huggingface_hub reads the file for its SHA-256 as well
                self.upload_info = SimpleNamespace(sha256=hashlib.sha256(data).digest(), size=len(data))

            def data(self):
                return self.source if isinstance(self.source, bytes) else Path(self.source).read_bytes()

        self.module = types.ModuleType('huggingface_hub')
        self.module.HfApi, self.module.CommitOperationAdd = HfApi, CommitOperationAdd

    def entry(self, path):
        """a file of the listing: archives with their LFS size and SHA-256, small text files with their git blob id"""
        data = self.files[path]
        if path.endswith('.tar.zst'):
            lfs = SimpleNamespace(size=len(data), sha256=hashlib.sha256(data).hexdigest(), pointer_size=134)
            return SimpleNamespace(path=path, size=len(data), blob_id='f' * 40, lfs=lfs)
        return SimpleNamespace(path=path, size=len(data), lfs=None,
                               blob_id=hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest())


@pytest.fixture
def hub(monkeypatch):
    fake = FakeHub()
    monkeypatch.setitem(sys.modules, 'huggingface_hub', fake.module)
    monkeypatch.setattr(upload.time, 'sleep', lambda seconds: None)
    return fake


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def make_data(tmp_path):
    """archives of 3 cases x 5 runs and the base_lu run of a dropped case, the production's index, the case list"""
    archives, rows = tmp_path / 'archives', []
    for run in RUNS:
        for k, case in enumerate(CASES + (DROPPED,)):
            if case == DROPPED and run != 'base_lu':
                continue
            data = f"{run} {case}\n".encode() * (40 + 10 * k)
            path = archives / run / f"{case}.tar.zst"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            rows.append({'run': run, 'case': case, 'file': path.name, 'bytes': len(data),
                         'md5': hashlib.md5(data).hexdigest(), 'remote': f"remote:archives/{run}/{path.name}",
                         'uploaded': '2026-10-10T12:00:00+0300'})
    first = rows[0]
    rows.insert(0, dict(first, md5='1' * 32))                 # indexed twice: the later row is the current one
    rows.append(dict(first, file='normal_circle_0001.cap10000.tar.zst',                     # superseded attempts
                     remote='remote:archives/0.25x/superseded/normal_circle_0001.cap10000.tar.zst'))
    rows.append(dict(first, md5='2' * 32, remote='remote:archives/0.25x/superseded/normal_circle_0001.tar.zst'))
    index = tmp_path / 'archive_index.csv'
    write_csv(index, rows)
    cases = tmp_path / 'metadata.csv'
    write_csv(cases, [{'dataset_index': k, 'case_name': case, 'split': 'train'} for k, case in enumerate(CASES)])
    card = tmp_path / 'README.md'
    card.write_bytes(b"---\nlicense: cc-by-4.0\n---\n\n> **Upload in progress.**\n\n# Archives\n")
    return archives, index, cases, card


def snapshot(folder):
    return {p.relative_to(folder): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(folder.rglob('*'))}


def files_in(folder):
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file())


def expected_paths():
    return {f"{run}/{case}.tar.zst" for run in RUNS for case in CASES}


def test_case_names(tmp_path):
    _, _, cases, _ = make_data(tmp_path)
    assert upload.case_names(cases) == set(CASES)
    pq = pytest.importorskip('pyarrow.parquet')
    import pyarrow as pa
    pq.write_table(pa.table({'case_name': list(CASES), 'size': ['normal'] * 3}), tmp_path / 'metadata.parquet')
    assert upload.case_names(tmp_path / 'metadata.parquet') == set(CASES)


def test_select_rows(tmp_path):
    archives, index, cases, _ = make_data(tmp_path)
    rows, others = upload.select_rows(index, upload.case_names(cases))
    assert [(r['run'], r['case']) for r in rows] == [(run, case) for run in RUNS for case in CASES]
    assert others == [('base_lu', DROPPED)]                   # the dropped case is left out
    for r in rows:                                            # the current row of the archive indexed twice, live only
        assert r['md5'] == hashlib.md5((archives / r['run'] / r['file']).read_bytes()).hexdigest()
    with pytest.raises(ValueError):                           # dropped_cases.csv given by mistake: base_lu only
        upload.select_rows(index, {DROPPED})
    with pytest.raises(ValueError):                           # a case without archives
        upload.select_rows(index, set(CASES) | {'normal_circle_0005'})


def test_index_csv(tmp_path):
    _, index, cases, _ = make_data(tmp_path)
    rows, _ = upload.select_rows(index, upload.case_names(cases))
    lines = upload.index_csv(rows).decode().splitlines()
    assert lines[0] == 'run,case,file,bytes,md5' and len(lines) == 1 + 15
    assert lines[1].split(',') == [rows[0][k] for k in ('run', 'case', 'file', 'bytes', 'md5')]
    assert not any('remote' in line or 'superseded' in line or DROPPED in line for line in lines)


def test_upload_and_resume(tmp_path, hub):
    archives, index, cases, card = make_data(tmp_path)
    output, staging, before = tmp_path / 'out', tmp_path / 'stage', snapshot(archives)
    written = upload.run(archives, index, cases, REPO, output, card, batch=4, workers=2, limit=4, token='hf_secret',
                         staging=staging)
    assert hub.token == 'hf_secret' and len(written) == 4
    # created private, the card and index first, public before any archive
    assert hub.events[:3] == [('create_repo', True), ('create_commit', ('README.md', 'index.csv')),
                              ('update_repo_settings', False)]
    assert [name for name, _ in hub.events[3:]] == ['create_commit'] * 2     # batches of one run: 0.25x (3), 16x (1)
    assert hub.commits[0][0] == "Dataset card, licence and index"

    upload.run(archives, index, cases, REPO, output, card, batch=4, workers=2, staging=staging)        # resumes
    done = read_csv(output / 'done.csv')
    assert sorted((r['run'], r['case']) for r in done) == sorted((run, case) for run in RUNS for case in CASES)
    assert set(hub.files) == {'README.md', 'index.csv'} | expected_paths()
    assert hub.files['README.md'] == card.read_bytes()
    assert hub.files['index.csv'] == upload.index_csv(upload.select_rows(index, set(CASES))[0])
    commits = {f"{k:040x}": paths for k, (_, paths) in enumerate(hub.commits, start=1)}
    for r in done:
        path = f"{r['run']}/{r['case']}.tar.zst"
        assert r['md5_ok'] == 'True' and r['sha256'] == hashlib.sha256(hub.files[path]).hexdigest()
        assert int(r['bytes']) == len(hub.files[path]) and path in commits[r['commit']]
    assert not (output / 'errors.csv').exists()

    count = len(hub.events)
    upload.run(archives, index, cases, REPO, output, card, batch=4, workers=2, staging=staging)        # nothing to do
    assert len(hub.events) == count
    card.write_bytes(card.read_bytes().replace(b"> **Upload in progress.**\n\n", b""))
    upload.run(archives, index, cases, REPO, output, card, batch=4, workers=2, staging=staging)        # edited card
    assert hub.commits[-1] == ("Update README.md", ['README.md']) and hub.files['README.md'] == card.read_bytes()

    assert upload.run(None, index, cases, REPO, output, card, verify_only=True) == []
    assert read_csv(output / 'verify.csv') == []
    assert snapshot(archives) == before                       # nothing in the archives folder changed
    assert files_in(staging) == []
    assert not any(b'hf_secret' in p.read_bytes() for p in output.iterdir())


def test_staging_holds_one_batch(tmp_path, hub):
    """the commit uploads the staged copies; they are removed after it, so only one batch is ever on local disk"""
    archives, index, cases, card = make_data(tmp_path)
    staging, seen = tmp_path / 'stage', []

    def look(operations):
        sources = [Path(o.source) for o in operations if not isinstance(o.source, bytes)]
        seen.append((files_in(staging), sorted(o.path_in_repo for o in operations), sources))
    hub.on_commit = look
    upload.run(archives, index, cases, REPO, tmp_path / 'out', card, batch=2, workers=2, staging=staging)
    archive_commits = seen[1:]                                # the first commit is the card and index (bytes)
    assert seen[0][2] == [] and len(archive_commits) == 10    # 5 runs, 3 cases, at most 2 archives per commit
    for staged, paths, sources in archive_commits:
        assert staged == paths                                # this batch only: the earlier batches are gone
        assert all(source.parent.parent == staging for source in sources)
    assert files_in(staging) == [] and list(staging.iterdir()) == []


def test_partial_copy_removed(tmp_path, monkeypatch):
    """a copy that stops (the Drive mount drops) leaves no partial file in staging, and is redone"""
    source = tmp_path / 'archives' / '16x' / 'normal_circle_0001.tar.zst'
    source.parent.mkdir(parents=True)
    source.write_bytes(bytes(range(256)) * 4)
    target = tmp_path / 'stage' / '16x' / 'normal_circle_0001.tar.zst'
    partial = target.with_name(target.name + '.part')
    original = upload.HashingReader

    class Dropping(original):
        def read(self, size=-1):
            if self.bytes >= 300:
                assert partial.exists()                       # a partial copy is on disk when the mount drops
                raise OSError(errno.ENOTCONN, 'Transport endpoint is not connected')
            return super().read(size)
    monkeypatch.setattr(upload, 'CHUNK', 100)
    monkeypatch.setattr(upload, 'HashingReader', Dropping)
    with pytest.raises(OSError):
        upload.stage_archive(source, target)
    assert not target.exists() and not partial.exists()

    monkeypatch.setattr(upload, 'HashingReader', original)
    md5, sha256, size = upload.stage_archive(source, target)
    data = source.read_bytes()
    assert target.read_bytes() == data and size == len(data) and not partial.exists()
    assert (md5, sha256) == (hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest())


def test_staging_apart_from_archives(tmp_path, hub):
    archives, index, cases, card = make_data(tmp_path)
    for staging in (archives, archives / '16x', tmp_path):
        with pytest.raises(ValueError):
            upload.run(archives, index, cases, REPO, tmp_path / 'out', card, staging=staging)
    assert hub.events == []


def test_existing_repository(tmp_path, monkeypatch):
    """a public repository is left public; a private one with its card is made public before any archive"""
    archives, index, cases, card = make_data(tmp_path)
    for private, settings in ((False, True), (True, True), (True, False)):
        hub = FakeHub(settings=settings)
        monkeypatch.setitem(sys.modules, 'huggingface_hub', hub.module)
        hub.exists, hub.private = True, private
        hub.files['README.md'] = card.read_bytes()
        hub.files['index.csv'] = upload.index_csv(upload.select_rows(index, set(CASES))[0])
        upload.run(archives, index, cases, REPO, tmp_path / f"out_{private}_{settings}", card, batch=10, workers=2,
                   limit=3, staging=tmp_path / 'stage')
        names = [name for name, _ in hub.events]
        if private:
            method = 'update_repo_settings' if settings else 'update_repo_visibility'
            assert hub.events[0] == (method, False) and names[1:] == ['create_commit']
        else:
            assert names == ['create_commit']
        assert hub.private is False


def test_md5_mismatch(tmp_path, hub):
    """an archive whose MD5 does not match the index is not uploaded; it is an error row, tried again next time"""
    archives, index, cases, card = make_data(tmp_path)
    rows = read_csv(index)
    bad = next(r for r in rows if r['run'] == '4x' and r['case'] == CASES[1])
    bad['md5'] = '0' * 32
    write_csv(index, rows)
    output, staging = tmp_path / 'out', tmp_path / 'stage'
    upload.run(archives, index, cases, REPO, output, card, batch=10, workers=2, staging=staging)
    upload.run(archives, index, cases, REPO, output, card, batch=10, workers=2, staging=staging)
    errors = read_csv(output / 'errors.csv')
    assert [(r['run'], r['case']) for r in errors] == [('4x', CASES[1])] * 2 and 'MD5' in errors[0]['error']
    assert '4x/normal_circle_0002.tar.zst' not in hub.files and len(read_csv(output / 'done.csv')) == 14
    assert files_in(staging) == []
    problems = upload.run(None, index, cases, REPO, output, card, verify_only=True)
    assert [(p['path'], p['problem']) for p in problems] == [('4x/normal_circle_0002.tar.zst', 'missing')]


def test_found_in_repository(tmp_path, hub):
    """archives of a commit whose rows did not reach done.csv are read and recorded, not uploaded again"""
    archives, index, cases, card = make_data(tmp_path)
    output, staging = tmp_path / 'out', tmp_path / 'stage'
    upload.run(archives, index, cases, REPO, output, card, batch=10, workers=2, staging=staging)
    done = read_csv(output / 'done.csv')
    write_csv(output / 'done.csv', [r for r in done if r['run'] != 'base_lu'])
    hub.files['base_lu/normal_circle_0003.tar.zst'] = b'not the archive'
    assert {(p['case'], p['problem']) for p in upload.run(None, index, cases, REPO, output, card, verify_only=True)} \
        == {(CASES[0], 'unchecked'), (CASES[1], 'unchecked'), (CASES[2], 'mismatch')}

    count = len(hub.commits)
    written = upload.run(archives, index, cases, REPO, output, card, batch=10, workers=2, staging=staging)
    assert hub.commits[count:] == [("Add base_lu: normal_circle_0003 to normal_circle_0003 (1 archive)",
                                    ['base_lu/normal_circle_0003.tar.zst'])]
    assert [(r['case'], bool(r['commit'])) for r in written] == [(CASES[0], False), (CASES[1], False), (CASES[2], True)]
    assert upload.run(None, index, cases, REPO, output, card, verify_only=True) == []
    assert files_in(staging) == []


def test_drive_lost(tmp_path, hub, monkeypatch):
    """a dropped mount stops the run before the batch is committed; the batch is copied again next time"""
    archives, index, cases, card = make_data(tmp_path)
    output, staging = tmp_path / 'out', tmp_path / 'stage'
    original = upload.HashingReader

    class Dropping(original):
        def read(self, size=-1):
            if self.bytes and Path(self.raw.name).parts[-2:] == ('16x', 'normal_circle_0002.tar.zst'):
                raise OSError(errno.ENOTCONN, 'Transport endpoint is not connected')
            return super().read(size)
    monkeypatch.setattr(upload, 'CHUNK', 64)
    monkeypatch.setattr(upload, 'HashingReader', Dropping)
    with pytest.raises(upload.DriveLost):
        upload.run(archives, index, cases, REPO, output, card, batch=10, workers=2, staging=staging)
    assert {r['run'] for r in read_csv(output / 'done.csv')} == {'0.25x'}
    assert not any(path.startswith('16x/') for path in hub.files) and files_in(staging) == []

    monkeypatch.setattr(upload, 'HashingReader', original)
    hub.raise_next = [RuntimeError("upload failed: Transport endpoint is not connected")]   # lost during the upload
    with pytest.raises(upload.DriveLost):
        upload.run(archives, index, cases, REPO, output, card, batch=10, workers=2, staging=staging)
    assert files_in(staging) == []
    upload.run(archives, index, cases, REPO, output, card, batch=10, workers=2, staging=staging)
    assert len(read_csv(output / 'done.csv')) == 15 and not (output / 'errors.csv').exists()


def test_commit_tried_again(tmp_path, hub):
    archives, index, cases, card = make_data(tmp_path)
    hub.exists, hub.private = True, False
    hub.files['README.md'] = card.read_bytes()
    hub.files['index.csv'] = upload.index_csv(upload.select_rows(index, set(CASES))[0])
    hub.raise_next = [RuntimeError("504 Server Error: Gateway Time-out")]
    upload.run(archives, index, cases, REPO, tmp_path / 'out', card, batch=10, workers=2, limit=3,
               staging=tmp_path / 'stage')
    assert [paths for _, paths in hub.commits] == [[f"0.25x/{case}.tar.zst" for case in CASES]]


def test_compare():
    """the verify comparison on a faked listing"""
    def archive(path, size, sha256):
        return SimpleNamespace(path=path, size=size, blob_id='f' * 40,
                               lfs=SimpleNamespace(size=size, sha256=sha256, pointer_size=134))
    card, index = b"# card\n", b"run,case,file,bytes,md5\n"
    rows = [{'run': '16x', 'case': f"normal_circle_000{k}", 'file': f"normal_circle_000{k}.tar.zst", 'bytes': '100',
             'md5': 'm'} for k in range(1, 7)]
    done = {('16x', f"normal_circle_000{k}"): {'bytes': '100', 'sha256': 'a' * 64} for k in (1, 3, 4, 5)}
    tree = {'16x/normal_circle_0001.tar.zst': archive('16x/normal_circle_0001.tar.zst', 100, 'a' * 64),
            '16x/normal_circle_0003.tar.zst': archive('16x/normal_circle_0003.tar.zst', 100, 'b' * 64),
            '16x/normal_circle_0004.tar.zst': archive('16x/normal_circle_0004.tar.zst', 99, 'a' * 64),
            '16x/normal_circle_0005.tar.zst': SimpleNamespace(path='16x/normal_circle_0005.tar.zst', size=100,
                                                              blob_id='f' * 40, lfs=None),
            '16x/normal_circle_0006.tar.zst': archive('16x/normal_circle_0006.tar.zst', 100, 'c' * 64),
            '16x/normal_circle_0007.tar.zst': archive('16x/normal_circle_0007.tar.zst', 100, 'd' * 64),
            '.gitattributes': SimpleNamespace(path='.gitattributes', size=10, blob_id='0' * 40, lfs=None),
            'README.md': SimpleNamespace(path='README.md', size=len(card), lfs=None,
                                         blob_id=hashlib.sha1(b'blob %d\0' % len(card) + card).hexdigest() + 'x')}
    problems = upload.compare(tree, rows, done, {'README.md': card, 'index.csv': index})
    found = {(p['path'], p['problem']) for p in problems}
    assert found == {('README.md', 'mismatch'), ('index.csv', 'missing'),
                     ('16x/normal_circle_0002.tar.zst', 'missing'),
                     ('16x/normal_circle_0003.tar.zst', 'mismatch'),           # SHA-256 differs from done.csv
                     ('16x/normal_circle_0004.tar.zst', 'mismatch'),           # size differs from the index
                     ('16x/normal_circle_0005.tar.zst', 'mismatch'),           # no LFS / xet SHA-256
                     ('16x/normal_circle_0006.tar.zst', 'unchecked'),          # not in done.csv
                     ('16x/normal_circle_0007.tar.zst', 'extra')}
    assert next(p for p in problems if p['path'].endswith('0003.tar.zst'))['detail'] == \
        f"SHA-256 {'b' * 64}, done.csv {'a' * 64}"


def test_notebook_copies():
    """the scripts written by the Colab notebook are those of the repository"""
    notebook = json.loads((REPO_ROOT / 'tools' / 'colab' / 'upload_raw_archives.ipynb').read_text(encoding='utf-8'))
    written = {}
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if source.startswith('%%writefile '):
            first, _, body = source.partition('\n')
            written[first.split()[1]] = body
    for name in ('tools/extract_line_samples.py', 'tools/upload_raw_archives.py'):
        assert written[name].rstrip('\n') == (REPO_ROOT / name).read_text(encoding='utf-8').rstrip('\n'), name
