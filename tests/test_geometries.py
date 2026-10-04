import math
from itertools import product

import numpy as np
import pytest

from geometries.sampling import (case_rng, check_geometry, get_coordinates, obstacle_outline, polygon_area,
                                 rotate_and_shift)
from utils import describe_obstacle

SIZES = ['normal', 'smaller', 'larger']


def sample(config, size, geometry, index, seed=None):
    rng = case_rng(config['seed'] if seed is None else seed, size, geometry, index)
    return get_coordinates(geometry, size, config, rng)


def test_same_seed_gives_same_obstacle(config):
    for size, geometry in product(SIZES, config['geometries']):
        first, second = sample(config, size, geometry, 7), sample(config, size, geometry, 7)
        assert np.array_equal(first['coordinates'], second['coordinates'])
        assert first['rotation'] == second['rotation'] and first['shift_x'] == second['shift_x']
        if geometry == 'circle':
            assert np.array_equal(first['arcs'], second['arcs'])


def test_cases_and_seeds_give_different_obstacles(config):
    for size, geometry in product(SIZES, config['geometries']):
        reference = sample(config, size, geometry, 1)['coordinates']
        assert not np.array_equal(reference, sample(config, size, geometry, 2)['coordinates'])
        assert not np.array_equal(reference, sample(config, size, geometry, 1, seed=12345)['coordinates'])


@pytest.mark.parametrize("size", SIZES)
def test_triangles_point_the_right_way(config, size):
    """rotated by at most 45 degrees, a left triangle still points upstream and a right triangle downstream"""
    for i in range(1, 201):
        left = sample(config, size, 'left_triangle', i)['coordinates']
        right = sample(config, size, 'right_triangle', i)['coordinates']

        # pointy edge: first point of a left triangle, second point of a right triangle
        left_direction = left[0] - 0.5 * (left[1] + left[2])
        right_direction = right[1] - 0.5 * (right[0] + right[2])
        assert left_direction[0] < 0 and right_direction[0] > 0


def test_sampled_values_stay_in_their_ranges(config):
    sizes, placement = config['sizes'], config['placement']
    for size, geometry in product(SIZES, config['geometries']):
        nominal = sizes['nominal'][geometry]
        for i in range(1, 101):
            obstacle = sample(config, size, geometry, i)
            assert abs(obstacle['width'] / (nominal['width'] * sizes['scale'][size]) - 1) <= sizes['width_jitter']
            assert abs(obstacle['height'] / (nominal['height'] * sizes['scale'][size]) - 1) <= sizes['height_jitter']
            assert placement['rotation'][0] <= obstacle['rotation'] <= placement['rotation'][1]
            assert placement['shift_x'][0] <= obstacle['shift_x'] <= placement['shift_x'][1]
            assert placement['shift_y'][0] <= obstacle['shift_y'] <= placement['shift_y'][1]


def test_every_obstacle_keeps_the_minimum_gaps(config):
    placement = config['placement']
    for size, geometry in product(SIZES, config['geometries']):
        for i in range(1, 101):
            obstacle = sample(config, size, geometry, i)
            gaps = describe_obstacle(obstacle['coordinates'], obstacle['arcs'], config['domain'])
            assert min(gaps['gap_bottom'], gaps['gap_top']) >= placement['min_wall_gap']
            assert gaps['gap_inlet'] >= placement['min_inlet_gap']
            assert gaps['gap_outlet'] >= placement['min_outlet_gap']
            assert obstacle['attempts'] >= 1


def test_rotate_and_shift():
    square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    rotated = rotate_and_shift(square, 90, [10, 0], pivot=np.array([1.0, 1.0]))
    assert np.allclose(rotated, [[12, 0], [12, 2], [10, 2], [10, 0]])
    # rotation keeps the area and the counter-clockwise order
    turned = rotate_and_shift(square, 33, [0, 0], pivot=np.array([1.0, 1.0]))
    assert math.isclose(polygon_area(turned), polygon_area(square))


def test_check_geometry(config):
    domain, placement = config['domain'], config['placement']
    square = np.array([[100, 50], [150, 50], [150, 70], [100, 70]], dtype=float)
    assert check_geometry(square, domain, placement)
    assert not check_geometry(square[::-1], domain, placement)                  # clockwise
    assert not check_geometry(square[[0, 2, 1, 3]], domain, placement)          # crosses itself
    assert not check_geometry(square + [0, 47], domain, placement)              # 3 mm from the top wall
    assert not check_geometry(square - [45, 0], domain, placement)              # 55 mm from the inlet
    assert not check_geometry(square + [70, 0], domain, placement)              # 140 mm from the outlet


def test_circle_outline_has_arc_points(config):
    obstacle = sample(config, 'normal', 'circle', 3)
    assert obstacle['arcs'].shape == (4, 2)
    assert len(obstacle_outline(obstacle['coordinates'], obstacle['arcs'])) == 8
