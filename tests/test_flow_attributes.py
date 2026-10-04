import math

import numpy as np

from tools.flow_attributes import attributes, relative_difference, upsample
from tools.openfoam_attributes import obstacle_geometry

NX, NY = 261, 121  # 1 mm spacing


def sample_with_wake():
    """obstacle at x 100-120 mm, y 50-70 mm, reverse flow up to x = 140 mm behind it"""
    sample = np.zeros((6, NX, NY))
    x = np.arange(NX)[:, None] * np.ones((1, NY))
    y = np.ones((NX, 1)) * np.arange(NY)[None, :]
    obstacle = (x >= 100) & (x <= 120) & (y >= 50) & (y <= 70)
    sample[0] = np.where(obstacle, 0, 0.1)
    sample[0][(x > 120) & (x <= 140) & (y > 55) & (y < 65)] = -0.01
    sample[2] = 0.01 * (1 - x / 260)
    sample[4] = np.where(obstacle, 1, 4)
    sample[4][:, 0] = sample[4][:, -1] = 1
    return sample


def test_attributes_of_a_simple_wake():
    result, vorticity = attributes(sample_with_wake())
    assert math.isclose(result['recirculation_length_mm'], 20 * 260 / 260, rel_tol=0.02)
    assert result['recirculation_area_mm2'] > 0
    assert 0.4 < result['flow_split_upper'] < 0.6
    assert result['dp_inlet_outlet'] > 0
    assert result['peak_vorticity'] > 0 and vorticity.shape == (NX, NY)


def test_relative_difference_and_upsample():
    assert math.isclose(relative_difference(1.1, 1.0), 10.0)
    assert math.isclose(relative_difference(0.2, 0.1, scale=2.0), 5.0)
    assert math.isnan(relative_difference(1.0, 0.0))
    coarse = np.arange(12, dtype=float).reshape(4, 3)
    fine = upsample(coarse, 7, 5)
    assert fine.shape == (7, 5) and fine[0, 0] == coarse[0, 0] and fine[-1, -1] == coarse[-1, -1]


def test_obstacle_geometry_stays_clear_of_the_channel_walls():
    row = {'x1': '100', 'x2': '160', 'x3': '160', 'x4': '100', 'y1': '5', 'y2': '5', 'y3': '115', 'y4': '115',
           'centroid_x': '130'}
    (xmin, ymin, xmax, ymax), x_centroid, x_wake = obstacle_geometry(row)
    assert (xmin, xmax) == (0.09, 0.17)
    assert ymin == 0.001 and ymax == 0.119
    assert x_centroid == 0.13 and x_wake == 0.18
