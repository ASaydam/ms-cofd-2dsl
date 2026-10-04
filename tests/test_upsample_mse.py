import numpy as np
import pytest

from tools.flow_attributes import upsample as bilinear_upsample
from tools.upsample_mse import METHODS, fill_obstacle, upsample, weights


def test_weights_sum_to_one_and_keep_the_corner_points():
    for method in METHODS:
        w = weights(26, 104, method)
        assert np.allclose(w.sum(axis=1), 1)
        assert w[0, 0] == 1 and w[-1, -1] == 1


def test_linear_and_quadratic_fields_are_upsampled_exactly():
    def field(x, y, order):
        return 0.3 * x - 0.2 * y + 1 + (0.01 * x ** 2 - 0.02 * x * y if order == 2 else 0)

    x, y = np.arange(26)[:, None] * 10.0, np.arange(12)[None, :] * 10.0
    fine_x, fine_y = np.arange(104)[:, None] * 250 / 103, np.arange(48)[None, :] * 110 / 47
    assert np.allclose(upsample(field(x, y, 1), 104, 48, 'bilinear'), field(fine_x, fine_y, 1))
    # bicubic repeats the edge values beyond the grid: exact one coarse cell away from the edges
    inner = (slice(5, -5), slice(5, -5))
    for order in (1, 2):
        assert np.allclose(upsample(field(x, y, order), 104, 48, 'bicubic')[inner], field(fine_x, fine_y, order)[inner])


def test_bilinear_matches_the_upsampling_of_flow_attributes():
    field = np.random.default_rng(1).random((26, 12))
    assert np.allclose(upsample(field, 104, 48, 'bilinear'), bilinear_upsample(field, 104, 48))


def test_obstacle_points_get_the_nearest_fluid_value():
    pytest.importorskip("scipy")
    field = np.arange(50, dtype=float).reshape(10, 5)
    cell_type = np.full((10, 5), 4)
    cell_type[:, 0] = cell_type[:, -1] = 1   # channel walls stay as they are
    cell_type[4:6, 2] = 1                    # obstacle
    filled = fill_obstacle(field, cell_type)
    assert filled[4, 2] in (field[3, 2], field[4, 1], field[4, 3])
    assert filled[0, 0] == field[0, 0]
