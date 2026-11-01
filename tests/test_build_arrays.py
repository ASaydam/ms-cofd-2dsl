import numpy as np
import pytest

# the obstacle distance needs scipy (k-d tree), the tests run in the container
pytest.importorskip("scipy")

from extraction.build_arrays import (FLUID, INLET, OBSTACLE, OUTLET, WALL, build_sample, cell_type_mask,  # noqa: E402
                                     signed_obstacle_distance, wall_distance)

NX, NY = 40, 20


def square_obstacle():
    obstacle = np.zeros((NX, NY), dtype=bool)
    obstacle[15:20, 8:12] = True
    return obstacle


def flow_fields(obstacle, seed=0):
    rng = np.random.default_rng(seed)
    return [np.where(obstacle, 0, rng.uniform(0.01, 0.2, obstacle.shape)) for _ in range(3)]


def brute_force_distance(cell_type, obstacle):
    """the obstacle distance compared with every border point"""
    padded = np.pad(cell_type, 1, mode='edge')
    solid = ((padded[2:, 1:-1] == 1) & (padded[:-2, 1:-1] == 1) & (padded[1:-1, 2:] == 1) & (padded[1:-1, :-2] == 1))
    border = np.argwhere(obstacle & ~solid)
    points = np.argwhere(np.ones(obstacle.shape, dtype=bool))
    distance = np.sqrt(np.min(np.sum((points[:, None, :] - border[None, :, :]) ** 2, axis=2), axis=1))
    distance = distance.reshape(obstacle.shape)
    return np.where(obstacle, -distance, distance)


def test_cell_type_mask():
    cell_type = cell_type_mask(square_obstacle())
    assert cell_type[17, 10] == OBSTACLE and cell_type[5, 5] == FLUID
    assert np.all(cell_type[1:-1, 0] == WALL) and np.all(cell_type[1:-1, -1] == WALL)
    assert np.all(cell_type[0] == INLET) and np.all(cell_type[-1] == OUTLET)


def test_wall_distance():
    distance = wall_distance(NX, NY)
    assert distance.shape == (NX, NY)
    assert distance[0, 0] == 0 and distance[0, NY - 1] == 0 and distance[0, NY // 2] == (NY - 1) // 2
    assert np.array_equal(distance[0], distance[0, ::-1])        # symmetric about the channel centre


def test_signed_obstacle_distance():
    obstacle = square_obstacle()
    distance = signed_obstacle_distance(cell_type_mask(obstacle), obstacle)
    assert np.array_equal(distance, brute_force_distance(cell_type_mask(obstacle), obstacle))

    assert distance[15, 8] == 0                         # border point
    assert distance[17, 10] < 0                         # inside the obstacle
    assert distance[10, 10] == 5                        # 5 points left of the border
    assert np.all(distance[~obstacle] > 0)


def test_channel_order():
    obstacle = square_obstacle()
    ux, uy, p = flow_fields(obstacle)
    sample = build_sample(ux, uy, p, valid=(~obstacle).astype(float))

    # u_x, u_y, p, wall_distance, cell_type, obstacle_sdf
    assert sample.shape == (6, NX, NY)
    assert np.array_equal(sample[0], ux) and np.array_equal(sample[1], uy) and np.array_equal(sample[2], p)
    assert np.array_equal(sample[3], wall_distance(NX, NY))
    assert np.array_equal(sample[4], cell_type_mask(obstacle))
    assert sample[5][17, 10] < 0
