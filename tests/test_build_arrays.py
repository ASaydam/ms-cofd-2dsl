import numpy as np
import pytest

# the obstacle distance needs scipy (k-d tree), the tests run in the container
pytest.importorskip("scipy")

from extraction.build_arrays import (FLUID, INLET, OBSTACLE, OUTLET, WALL, build_arrays, build_case, cell_type_mask,
                                     signed_obstacle_distance, wall_distance)

NX, NY = 40, 20


def square_obstacle():
    obstacle = np.zeros((NX, NY), dtype=bool)
    obstacle[15:20, 8:12] = True
    return obstacle


def write_extract(path, ux, uy, p, valid=None):
    """writes a CSV file like ParaView does: rows along x first, then y"""
    columns = [ux.T.ravel(), uy.T.ravel(), np.zeros(ux.size), p.T.ravel()]
    header = '"U:0","U:1","U:2","p"'
    if valid is not None:
        columns.append(valid.T.ravel())
        header += ',"vtkValidPointMask"'
    np.savetxt(path, np.column_stack(columns), delimiter=',', header=header, comments='')


def flow_fields(obstacle, seed=0):
    rng = np.random.default_rng(seed)
    return [np.where(obstacle, 0, rng.uniform(0.01, 0.2, obstacle.shape)) for _ in range(3)]


def test_cell_type_mask():
    cell_type = cell_type_mask(square_obstacle())
    assert cell_type[17, 10] == OBSTACLE and cell_type[5, 5] == FLUID
    assert np.all(cell_type[1:-1, 0] == WALL) and np.all(cell_type[1:-1, -1] == WALL)
    assert np.all(cell_type[0] == INLET) and np.all(cell_type[-1] == OUTLET)


def test_wall_distance():
    distance = wall_distance(NX, NY)
    assert distance.shape == (NX, NY)
    assert distance[0, 0] == 0 and distance[0, NY - 1] == 1 and distance[0, NY // 2] == NY // 2


def brute_force_distance(cell_type, obstacle):
    """the obstacle distance compared with every border point (how MS-COFD v4 computed it)"""
    padded = np.pad(cell_type, 1, mode='edge')
    solid = ((padded[2:, 1:-1] == 1) & (padded[:-2, 1:-1] == 1) & (padded[1:-1, 2:] == 1) & (padded[1:-1, :-2] == 1))
    border = np.argwhere(obstacle & ~solid)
    points = np.argwhere(np.ones(obstacle.shape, dtype=bool))
    distance = np.sqrt(np.min(np.sum((points[:, None, :] - border[None, :, :]) ** 2, axis=2), axis=1))
    distance = distance.reshape(obstacle.shape)
    return np.where(obstacle, -distance, distance)


def test_signed_obstacle_distance():
    obstacle = square_obstacle()
    distance = signed_obstacle_distance(cell_type_mask(obstacle), obstacle)
    assert np.array_equal(distance, brute_force_distance(cell_type_mask(obstacle), obstacle))

    assert distance[15, 8] == 0                         # border point
    assert distance[17, 10] < 0                         # inside the obstacle
    assert distance[10, 10] == 5                        # 5 points left of the border
    assert np.all(distance[~obstacle] > 0)


def test_channel_order_and_obstacle_from_mask_or_zero_velocity(tmp_path):
    obstacle = square_obstacle()
    ux, uy, p = flow_fields(obstacle)

    write_extract(tmp_path / "new_last_time.csv", ux, uy, p, valid=(~obstacle).astype(float))
    write_extract(tmp_path / "old_last_time.csv", ux, uy, p)
    new, old = build_case(tmp_path / "new_last_time.csv", NX, NY), build_case(tmp_path / "old_last_time.csv", NX, NY)

    assert np.array_equal(new, old)
    # u_x, u_y, p, wall_distance, cell_type, obstacle_sdf
    assert np.allclose(new[0], ux) and np.allclose(new[2], p)
    assert np.array_equal(new[3], wall_distance(NX, NY))
    assert np.array_equal(new[4], cell_type_mask(obstacle))
    assert new[5][17, 10] < 0


def test_build_arrays_writes_one_array_and_the_case_order(tmp_path):
    extracts = tmp_path / "extracts"
    extracts.mkdir()
    for name in ("larger_circle_10", "larger_circle_1", "larger_circle_100"):
        write_extract(extracts / f"{name}_last_time.csv", *flow_fields(square_obstacle()),
                      valid=(~square_obstacle()).astype(float))

    build_arrays(extracts, tmp_path / "arrays", "base", NX, NY)
    array = np.load(tmp_path / "arrays" / "base_tlast_6ch.npy")
    order = (tmp_path / "arrays" / "cases.csv").read_text().split()[1:]

    assert array.shape == (3, 6, NX, NY) and array.dtype == np.float32
    # sorted by file name, as the released arrays: "..._100_last_time" < "..._10_last_time" < "..._1_last_time"
    assert [row.split(',')[1] for row in order] == ["larger_circle_100", "larger_circle_10", "larger_circle_1"]
