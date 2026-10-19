import csv
import json
import sys

import numpy as np
import pytest

from dataset.splits import assign_splits
from hpc import ledger

RUNS = ['0.25x', 'base_fo', 'base_lu', '4x', '16x']
PLACEMENT = {'shift_x': [-30, 10], 'shift_y': [-15, 15]}


def manifest_rows(n=40):
    rng = np.random.default_rng(0)
    rows = []
    for size in ('normal', 'larger'):
        for i in range(1, n + 1):
            rows.append({'case_name': f"{size}_circle_{i:04d}", 'size': size, 'geometry': 'circle', 'index': str(i),
                         'blockage_ratio': f"{rng.uniform(0.1, 0.4):.4f}", 'rotation': f"{rng.uniform(-45, 45):.4f}",
                         'shift_x': f"{rng.uniform(-30, 10):.4f}", 'shift_y': f"{rng.uniform(-15, 15):.4f}"})
    return rows


def test_splits_are_fixed_and_complete():
    rows = manifest_rows()
    settings = {'seed': 2026, 'ood_quantile': 0.9, 'ood': ['blockage', 'rotation', 'position'],
                'fractions': {'train': 0.8, 'val': 0.1, 'test': 0.1}}
    split = assign_splits(rows, settings, PLACEMENT)
    assert split == assign_splits(list(reversed(rows)), settings, PLACEMENT)   # independent of the order
    assert set(split) == {row['case_name'] for row in rows}
    top = max(rows, key=lambda row: float(row['blockage_ratio']))
    assert split[top['case_name']] == 'ood_blockage'
    counts = {name: list(split.values()).count(name) for name in set(split.values())}
    assert 6 <= counts['ood_blockage'] <= 9 and counts['train'] > 3 * counts['test']


def fake_root(root, n=3):
    """a data root with n normal circles, every run done; normal_circle_0002 without a steady solution"""
    (root / 'geometry').mkdir(parents=True)
    rows = [{'case_name': f"normal_circle_{i:04d}", 'size': 'normal', 'geometry': 'circle', 'index': str(i),
             'blockage_ratio': '0.3', 'rotation': str(10 * i), 'shift_x': '-10', 'shift_y': str(i)}
            for i in range(1, n + 1)]
    with open(root / 'geometry' / 'manifest.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shapes = {'0.25x': (6, 9, 3), 'base_fo': (6, 18, 6), 'base_lu': (6, 18, 6), '4x': (6, 36, 12), '16x': (6, 72, 24)}
    for run in RUNS:
        folder = root / 'results' / run
        folder.mkdir(parents=True)
        for row in rows:
            case = row['case_name']
            status = 'end_time' if (run == 'base_lu' and case == 'normal_circle_0002') else 'converged'
            np.savez_compressed(folder / f"{case}.npz", sample=np.full(shapes[run], float(row['index']), np.float32),
                                time=100.0)
            (folder / f"{case}_attributes.json").write_text(json.dumps(
                {'cells': {'base_lu': 1000, '4x': 4000, '16x': 16000}.get(run, 500), 'Fx': 1.0 + 0.01 * RUNS.index(run)}))
            (folder / f"{case}_provenance.json").write_text(json.dumps(
                {'status': status, 'iterations': '900', 'residual_Ux': '1e-07', 'timings_s': {'simpleFoam': 60.0}}))
            (folder / f"{case}_surface.csv").write_text("x_mm,y_mm,p,wall_shear_x,wall_shear_y\n1,2,3,4,5\n6,7,8,9,10\n")
            ledger.set_state(run, case, 'done', f"{status} 900 1", root=root)


def test_build_dataset(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    pq = pytest.importorskip("pyarrow.parquet")
    from dataset import build_dataset

    root = tmp_path / 'root'
    fake_root(root)
    monkeypatch.setenv('MSCOFD_DATA_ROOT', str(root))
    monkeypatch.setattr(sys, 'argv', ['build_dataset', str(tmp_path / 'out')])
    build_dataset.main()

    out = tmp_path / 'out'
    info = json.loads((out / 'dataset_info.json').read_text())
    assert info['cases'] == 2 and info['dropped_cases'] == 1                       # 0002 dropped
    names = None
    for run in RUNS:
        files = sorted((out / run).glob('*.h5'))
        cases = []
        for path in files:
            with h5py.File(path) as h5:
                cases += [c.decode() if isinstance(c, bytes) else c for c in h5['case_name'][:]]
                assert h5['fields'].chunks[0] == 1 and list(h5.attrs['channels'])[0] == 'u_x'
        names = names or cases
        assert cases == names and 'normal_circle_0002' not in cases               # same rows in every level
    metadata = pq.read_table(out / 'metadata.parquet')
    assert metadata.num_rows == 2 and 'base_lu.Fx' in metadata.column_names
    # starting fields not in the provenance: the run's init_from of config/dataset.yaml, else rest; never empty
    assert set(metadata.column('16x.init_from').to_pylist()) == {'base_lu'}
    assert set(metadata.column('0.25x.init_from').to_pylist()) == {'rest'}
    assert pq.read_table(out / 'surface.parquet').num_rows == 2 * 5 * 2
    dropped = list(csv.DictReader(open(out / 'dropped_cases.csv')))
    assert [row['case_name'] for row in dropped] == ['normal_circle_0002']
    assert len((out / 'checksums.md5').read_text().splitlines()) == len([p for p in out.rglob('*') if p.is_file()]) - 1

    # the loader reads the same rows in every level, with the parameters of the case
    from dataset.loader import MSCOFD
    data = MSCOFD(out, split=list(info['splits']), levels=['base_lu', '16x'], channels=['u_x', 'p'],
                  parameters=['rotation'])
    assert len(data) == 2 and data.case_names == metadata.column('case_name').to_pylist()
    for i in range(len(data)):
        sample = data[i]
        index = int(sample['case_name'][-4:])
        assert tuple(sample['base_lu'].shape) == (2, 18, 6) and tuple(sample['16x'].shape) == (2, 72, 24)
        assert float(sample['16x'][0, 0, 0]) == index and float(sample['parameters'][0]) == 10 * index
    normalised = MSCOFD(out, split=list(info['splits']), levels=['16x'], normalise=True)
    stacked = np.stack([np.asarray(normalised[i]['16x']) for i in range(len(normalised))])
    assert abs(stacked[:, 0].mean()) < 1e-5            # one split, so its statistics are those of all cases


def test_metadata_only(tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("h5py")
    from dataset import build_dataset

    root, release, out = tmp_path / 'root', tmp_path / 'release', tmp_path / 'out'
    fake_root(root)
    monkeypatch.setenv('MSCOFD_DATA_ROOT', str(root))
    monkeypatch.setattr(sys, 'argv', ['build_dataset', str(release)])
    build_dataset.main()

    # a start recorded in the provenance wins over the configured one
    provenance = root / 'results' / '0.25x' / 'normal_circle_0001_provenance.json'
    provenance.write_text(json.dumps({**json.loads(provenance.read_text()), 'init_from': 'base_lu'}))
    monkeypatch.setattr(sys, 'argv', ['build_dataset', str(out), '--metadata-only', str(release)])
    build_dataset.main()

    assert sorted(p.name for p in out.iterdir()) == ['checksums.md5', 'dataset_info.json', 'metadata.csv',
                                                     'metadata.parquet']
    starts = dict(zip(*pq.read_table(out / 'metadata.parquet', columns=['case_name', '0.25x.init_from'])
                      .to_pydict().values()))
    assert starts == {'normal_circle_0001': 'base_lu', 'normal_circle_0003': 'rest'}
    old = dict(line.split()[::-1] for line in (release / 'checksums.md5').read_text().splitlines())
    new = dict(line.split()[::-1] for line in (out / 'checksums.md5').read_text().splitlines())
    assert new.keys() == old.keys()
    assert all(new[name] == old[name] for name in old if name.endswith('.h5') or name.startswith('surface'))
    assert new['metadata.parquet'] == build_dataset.md5_of(out / 'metadata.parquet') != old['metadata.parquet']
    info = json.loads((out / 'dataset_info.json').read_text())
    assert info['created'] == json.loads((release / 'dataset_info.json').read_text())['created']
    assert 'metadata_rebuilt' in info

    # a release with other cases or splits is refused
    (root / 'results' / '16x' / 'normal_circle_0003.npz').unlink()
    monkeypatch.setattr(sys, 'argv', ['build_dataset', str(tmp_path / 'out2'), '--metadata-only', str(release),
                                      '--allow-missing'])
    with pytest.raises(SystemExit):
        build_dataset.main()
