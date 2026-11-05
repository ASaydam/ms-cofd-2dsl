import csv
import json

import numpy as np

from extraction.build_arrays import wall_distance


def write_run(root, run, case, wake, nx=12, ny=5):
    """a result as hpc/run_task.py writes it, with channel 3 as min(j, ny - j)"""
    folder = root / 'results' / run
    folder.mkdir(parents=True, exist_ok=True)
    sample = np.random.default_rng(0).random((6, nx, ny)).astype(np.float32)
    rows = np.arange(ny)
    sample[3] = np.tile(np.minimum(ny - rows, rows), (nx, 1))
    np.savez_compressed(folder / f"{case}.npz", sample=sample, time=364.0)
    (folder / f"{case}_attributes.json").write_text(json.dumps({'Fx': 1.5, 'wake_min_u_x': wake}, indent=1))
    return sample


def test_patch_results(tmp_path):
    from tools.patch_results import run

    root, backup = tmp_path / 'root', tmp_path / 'backup'
    samples = {('16x', 'normal_circle_0001'): write_run(root, '16x', 'normal_circle_0001', 0.0),
               ('16x', 'normal_circle_0002'): write_run(root, '16x', 'normal_circle_0002', -0.02),
               ('4x', 'normal_circle_0001'): write_run(root, '4x', 'normal_circle_0001', 0.5)}
    wake = tmp_path / 'wake_band.csv'
    with open(wake, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['run', 'case', 'line_min', 'metadata', 'band_min'])
        writer.writerow(['16x', 'normal_circle_0001', 0.0, 0.0, 0.0125])        # wall zero: changes
        writer.writerow(['16x', 'normal_circle_0002', -0.02, -0.02, -0.02])     # reverse flow: stays
        writer.writerow(['4x', 'normal_circle_0001', 0.0, 0.0, 0.011])          # file holds another value

    dry = run(root, wake, backup, dry_run=True)
    assert dry['16x']['npz_rewritten'] == 2 and dry['16x']['wake_patched'] == 1
    assert not backup.exists()
    with np.load(root / 'results' / '16x' / 'normal_circle_0001.npz') as data:
        assert data['sample'][3, 0, -1] == 1                                    # nothing changed

    counts = run(root, wake, backup)
    assert counts['16x'] == {'npz': 2, 'npz_rewritten': 2, 'wake_patched': 1, 'wake_done': 0, 'wake_mismatch': 0,
                             'wake_missing': 0}
    assert counts['4x']['wake_mismatch'] == 1
    for (run_name, case), old in samples.items():
        with np.load(root / 'results' / run_name / f"{case}.npz") as data:
            sample = data['sample']
            assert data['time'] == 364.0 and sample.dtype == np.float32
        assert np.array_equal(sample[3], wall_distance(12, 5)) and sample[3, 0, 0] == sample[3, 0, -1] == 0
        assert np.array_equal(np.delete(sample, 3, axis=0), np.delete(old, 3, axis=0))   # other channels unchanged
    attributes = json.loads((root / 'results' / '16x' / 'normal_circle_0001_attributes.json').read_text())
    assert attributes == {'Fx': 1.5, 'wake_min_u_x': 0.0125}
    assert json.loads((backup / '16x' / 'normal_circle_0001_attributes.json').read_text())['wake_min_u_x'] == 0.0
    assert json.loads((root / 'results' / '4x' / 'normal_circle_0001_attributes.json').read_text())['wake_min_u_x'] == 0.5
    assert not list((root / 'results').rglob('*.tmp'))

    again = run(root, wake, backup)                                             # a second run changes nothing
    assert again['16x']['npz_rewritten'] == 0 and again['16x']['wake_done'] == 1 and again['16x']['wake_patched'] == 0


def test_patch_results_rounds_to_the_line_samples(tmp_path):
    """the csv holds float32 copies of OpenFOAM's sets (writePrecision 6); the patched value is the set's value"""
    from tools.patch_results import run

    root, backup = tmp_path / 'root', tmp_path / 'backup'
    write_run(root, '16x', 'normal_circle_0001', 0.0)
    wake = tmp_path / 'wake_band.csv'
    float32 = float(np.float32(0.0290714))
    with open(wake, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['run', 'case', 'line_min', 'metadata', 'band_min'])
        writer.writerow(['16x', 'normal_circle_0001', 0.0, 0.0, float32])
    run(root, wake, backup)
    attributes = json.loads((root / 'results' / '16x' / 'normal_circle_0001_attributes.json').read_text())
    assert float32 != 0.0290714 and attributes['wake_min_u_x'] == 0.0290714
