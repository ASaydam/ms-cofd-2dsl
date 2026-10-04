"""
Field differences between the base and the 16x solutions: the base fields are upsampled to the 16x grid with
several interpolation methods and compared with the 16x fields point by point:

    python -m tools.upsample_mse pilot/ms-cofd.h5
    python -m tools.upsample_mse pilot/ms-cofd.h5 --released mscofd2000_v4.h5   # also the released fields

Both grids span the whole channel (0-260 mm, 0-120 mm) with points on its edges, so the 16x point i lies at base
index i * (260 - 1) / (1040 - 1) ("align corners"). Methods:

    nearest          value of the nearest base point
    bilinear         linear in x and y (4 base points)
    bicubic          cubic convolution in x and y (16 base points, Keys 1981 with a = -0.5 as in MATLAB and
                     Pillow: exact for linear and quadratic fields; torch and OpenCV use a = -0.75, which is not)
    bicubic_filled   bicubic after the obstacle points of the base grid got the value of the nearest fluid point;
                     the arrays hold 0 inside the obstacle, which the other methods mix into the fluid next to it
                     (harmless for the velocity, which is 0 at the wall, not for the pressure)

For every case, channel (u_x, u_y, p) and method, over the fluid points of the 16x grid:

    mse                 mean of the squared difference per point
    rel_rmse_%          square root of the mse relative to the RMS of the 16x field
    near_obstacle_%     share of the squared difference within 2 mm of the obstacle

Writes <dataset>_upsample_mse.csv and prints the medians over the cases. Needs h5py and scipy.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import write_csv

CHANNELS = {'u_x': 0, 'u_y': 1, 'p': 2}
CELL_TYPE = 4
METHODS = ['nearest', 'bilinear', 'bicubic', 'bicubic_filled']
FINE_SPACING = 0.25  # mm, 16x grid (1040 x 480 points over 260 x 120 mm, rounded)


def keys_kernel(s, a=-0.5):
    """cubic convolution kernel (Keys 1981)"""
    s = np.abs(s)
    return np.where(s <= 1, (a + 2) * s ** 3 - (a + 3) * s ** 2 + 1,
                    np.where(s < 2, a * s ** 3 - 5 * a * s ** 2 + 8 * a * s - 4 * a, 0.0))


def weights(n_coarse, n_fine, method):
    """
    (n_fine, n_coarse) matrix that interpolates a line of coarse points to the fine points, both lines with points
    on their two ends; the edge values are repeated beyond the ends (bicubic)
    """
    t = np.arange(n_fine) * (n_coarse - 1) / (n_fine - 1)
    rows = np.arange(n_fine)
    w = np.zeros((n_fine, n_coarse))
    if method == 'nearest':
        w[rows, np.floor(t + 0.5).astype(int)] = 1
    elif method == 'bilinear':
        i0 = np.minimum(np.floor(t).astype(int), n_coarse - 2)
        w[rows, i0] = 1 - (t - i0)
        w[rows, i0 + 1] = t - i0
    elif method.startswith('bicubic'):
        i0 = np.floor(t).astype(int)
        for k in (-1, 0, 1, 2):
            np.add.at(w, (rows, np.clip(i0 + k, 0, n_coarse - 1)), keys_kernel(t - (i0 + k)))
    else:
        raise ValueError(f"unknown method {method}")
    return w


def obstacle_mask(cell_type):
    """obstacle points: wall type (1) except the two channel walls (first and last row)"""
    mask = cell_type == 1
    mask[:, 0] = mask[:, -1] = False
    return mask


def fill_obstacle(field, cell_type):
    """the obstacle points get the value of the nearest fluid point"""
    from scipy.ndimage import distance_transform_edt

    obstacle = obstacle_mask(cell_type)
    if not obstacle.any():
        return field
    nearest = distance_transform_edt(obstacle, return_distances=False, return_indices=True)
    return field[tuple(nearest)]


def upsample(field, nx, ny, method, cell_type=None):
    """base field (bx, by) at the 16x grid points (nx, ny)"""
    if method == 'bicubic_filled':
        field = fill_obstacle(field, cell_type)
    bx, by = field.shape
    return weights(bx, nx, method) @ field @ weights(by, ny, method).T


def compare(base, fine):
    """rows of mse, relative rmse and near-obstacle share for every channel and method of one case"""
    from scipy.ndimage import distance_transform_edt

    nx, ny = fine.shape[1:]
    fluid = fine[CELL_TYPE] != 1
    near = fluid & (distance_transform_edt(~obstacle_mask(fine[CELL_TYPE])) * FINE_SPACING <= 2.0)
    rows = []
    for method in METHODS:
        for name, channel in CHANNELS.items():
            difference = upsample(base[channel], nx, ny, method, base[CELL_TYPE]) - fine[channel]
            squared = difference[fluid] ** 2
            mse = float(squared.mean())
            rms = float(np.sqrt((fine[channel][fluid] ** 2).mean()))
            rows.append({'method': method, 'channel': name, 'mse': mse, 'rel_rmse_%': 100 * np.sqrt(mse) / rms,
                         'near_obstacle_%': 100 * float((difference[near] ** 2).sum() / squared.sum())})
    return rows


def case_rows(file, names=None):
    """(case name, base sample, 16x sample) of a dataset file, only the given case names when there are some"""
    all_names = [n.decode() if isinstance(n, bytes) else n for n in file['metadata/case_name'][:]]
    for index, name in enumerate(all_names):
        if names is None or name in names:
            yield name, file['base_tlast_6ch'][index].astype(float), file['16x_tlast_6ch'][index].astype(float)


def main():
    parser = argparse.ArgumentParser(description="MSE between the 16x fields and the upsampled base fields")
    parser.add_argument("dataset", help="dataset file with base_tlast_6ch, 16x_tlast_6ch and metadata/case_name")
    parser.add_argument("--released", help="released dataset (mscofd2000_v4.h5): the same cases compared there too")
    args = parser.parse_args()

    # imported here, the interpolation functions are used without h5py as well
    import h5py

    results = []
    sources = [('new', args.dataset)] + ([('released', args.released)] if args.released else [])
    names = None
    for source, path in sources:
        with h5py.File(path, 'r') as file:
            for case_name, base, fine in case_rows(file, names):
                for row in compare(base, fine):
                    results.append({'source': source, 'case_name': case_name, **row})
                print(f"{source:8s} {case_name}")
        names = {row['case_name'] for row in results}

    output = Path(args.dataset).with_name(Path(args.dataset).stem + "_upsample_mse.csv")
    write_csv(results, output, ['source', 'case_name', 'method', 'channel', 'mse', 'rel_rmse_%', 'near_obstacle_%'])

    print(f"\nmedian over the cases: mse (relative rmse %, share within 2 mm of the obstacle %)")
    for source, _ in sources:
        print(f"{source} fields")
        print(f"{'method':16s} " + " ".join(f"{name:>30s}" for name in CHANNELS))
        for method in METHODS:
            cells = []
            for name in CHANNELS:
                group = [r for r in results if r['source'] == source and r['method'] == method and r['channel'] == name]
                cells.append(f"{np.median([r['mse'] for r in group]):10.3e} ({np.median([r['rel_rmse_%'] for r in group]):5.2f}, "
                             f"{np.median([r['near_obstacle_%'] for r in group]):4.1f})")
            print(f"{method:16s} " + " ".join(f"{cell:>30s}" for cell in cells))
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
