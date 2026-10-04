from itertools import product

import numpy as np
import pytest

from conftest import RELEASED_NORMAL_GEO, RELEASED_EXTENDED_GEO
from geometries.sampling import case_rng, check_geometry, get_coordinates, obstacle_outline
from tools.build_manifest import read_obstacle

DOMAIN = {'length': 260, 'height': 120}


def sample(config, size, geometry, index, seed=None):
    rng = case_rng(config['seed'] if seed is None else seed, size, geometry, index)
    return get_coordinates(geometry, config['sizes'][size], config['domain'], rng)


def test_same_seed_gives_same_obstacle(config):
    for size, geometry in product(config['sizes'], config['geometries']):
        first, second = sample(config, size, geometry, 7), sample(config, size, geometry, 7)
        assert np.array_equal(first['coordinates'], second['coordinates'])
        if geometry == 'circle':
            assert np.array_equal(first['arcs'], second['arcs'])


def test_cases_and_seeds_give_different_obstacles(config):
    for size, geometry in product(config['sizes'], config['geometries']):
        reference = sample(config, size, geometry, 1)['coordinates']
        assert not np.array_equal(reference, sample(config, size, geometry, 2)['coordinates'])
        assert not np.array_equal(reference, sample(config, size, geometry, 1, seed=12345)['coordinates'])


@pytest.mark.parametrize("size", ["normal", "smaller", "larger"])
def test_triangles_point_the_right_way(config, size):
    for i in range(1, 301):
        left = sample(config, size, 'left_triangle', i)['coordinates']
        right = sample(config, size, 'right_triangle', i)['coordinates']

        # left triangle: first point is the pointy edge, on the left and between the other two in y
        assert left[0, 0] == left[:, 0].min() and left[0, 1] == np.median(left[:, 1])
        # right triangle: second point is the pointy edge, on the right and between the other two in y
        assert right[1, 0] == right[:, 0].max() and right[1, 1] == np.median(right[:, 1])


def test_sampled_sizes_stay_in_their_ranges(config):
    for size, geometry in product(config['sizes'], config['geometries']):
        parameters = config['sizes'][size]
        for i in range(1, 101):
            obstacle = sample(config, size, geometry, i)
            if parameters['scheme'] == 'legacy':
                assert parameters[geometry]['width'][0] <= obstacle['width'] <= parameters[geometry]['width'][1]
                assert parameters[geometry]['height'][0] <= obstacle['height'] <= parameters[geometry]['height'][1]
            else:
                nominal = parameters[geometry]
                assert abs(obstacle['width'] / nominal['width'] - 1) <= parameters['width_jitter']
                assert abs(obstacle['height'] / nominal['height'] - 1) <= parameters['height_jitter']


def test_check_geometry():
    square = np.array([[100, 50], [150, 50], [150, 70], [100, 70]], dtype=float)
    assert check_geometry(square, DOMAIN)
    assert not check_geometry(square[::-1], DOMAIN)                      # clockwise
    assert not check_geometry(square[[0, 2, 1, 3]], DOMAIN)              # crosses itself
    assert not check_geometry(square + [0, 60], DOMAIN)                  # outside of the domain


def test_legacy_circle_arcs(config):
    """arc mid-points follow the blockMeshDict generator: 5 mm bulge top / bottom, width / 20 left / right"""
    for i in range(1, 101):
        obstacle = sample(config, 'normal', 'circle', i)
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = obstacle['coordinates']
        bottom, right, top, left = obstacle['arcs']
        width, height = obstacle['width'], obstacle['height']

        assert bottom[0] == top[0] == int(x4) + width / 2
        assert abs(bottom[1] - y2) == 5 and abs(top[1] - y3) == 5
        assert abs(right[0] - x3) == pytest.approx(width / 20) and right[1] == y3 - height / 2
        assert abs(left[0] - x1) == pytest.approx(width / 20) and left[1] == y4 - height / 2
        # the left arc reuses the random sign of the top arc
        assert np.sign(left[0] - x1) == np.sign(top[1] - y3)


@pytest.mark.skipif(not RELEASED_NORMAL_GEO.exists(), reason="released normal .geo files not found")
def test_released_normal_circles_follow_the_legacy_arcs():
    """every released normal circle can be explained by the legacy arc rule with integer sizes 20 ... 30"""
    for number in range(801, 1001):
        coordinates, arcs = read_obstacle(RELEASED_NORMAL_GEO / f"{number}.geo")
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = coordinates
        bottom, right, top, left = arcs

        width = 2 * (bottom[0] - int(x4))
        height = 2 * (y3 - right[1])
        assert width == int(width) and 20 <= width <= 30
        assert height == int(height) and 20 <= height <= 30
        assert top[0] == bottom[0] and left[1] == y4 - height / 2
        assert abs(bottom[1] - y2) == 5 and abs(top[1] - y3) == 5
        assert abs(right[0] - x3) == pytest.approx(width / 20) and abs(left[0] - x1) == pytest.approx(width / 20)
        assert np.sign(left[0] - x1) == np.sign(top[1] - y3)


@pytest.mark.skipif(not RELEASED_EXTENDED_GEO.exists(), reason="released smaller / larger .geo files not found")
def test_released_geometries_pass_check_geometry():
    files = list(RELEASED_EXTENDED_GEO.glob("*.geo"))
    if RELEASED_NORMAL_GEO.exists():
        files += list(RELEASED_NORMAL_GEO.glob("*.geo"))

    for geo_file in files:
        assert check_geometry(obstacle_outline(*read_obstacle(geo_file)), DOMAIN), geo_file.name
