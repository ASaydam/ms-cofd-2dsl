"""
Builds the 6 channel sample of a case from its resampled flow fields (extraction/resample_vtk.py).

A sample has the shape (6, nx, ny), nx along the flow (inlet at x = 0), ny across the channel:

    0  u_x            x velocity (m/s)
    1  u_y            y velocity (m/s)
    2  p              kinematic pressure (m^2/s^2)
    3  wall_distance  distance to the closer channel wall, in grid points        (Wall Distance)
    4  cell_type      1 = wall or obstacle, 2 = inlet, 3 = outlet, 4 = fluid     (Cell Type Mask)
    5  obstacle_sdf   distance to the obstacle border, in grid points,           (Signed Obstacle Distance)
                      negative inside the obstacle

The flow channels are the converged solution (last time step). The geometry channels are made from the obstacle
points of the same resampling: the grid points inside the obstacle are outside of the mesh, so they are not valid.
"""

import numpy as np

CHANNELS = ['u_x', 'u_y', 'p', 'wall_distance', 'cell_type', 'obstacle_sdf']

# values of the cell_type channel
WALL = OBSTACLE = 1
INLET = 2
OUTLET = 3
FLUID = 4


def cell_type_mask(obstacle):
    """
    creates the cell_type channel from the obstacle points
    """
    cell_type = np.full(obstacle.shape, FLUID, dtype=float)
    cell_type[obstacle] = OBSTACLE

    cell_type[:, 0] = WALL     # lower wall, no-slip condition
    cell_type[:, -1] = WALL    # upper wall, no-slip condition
    cell_type[0, :] = INLET
    cell_type[-1, :] = OUTLET
    return cell_type


def wall_distance(nx, ny):
    """
    creates the wall_distance channel: min(j, ny - 1 - j) for the grid row j
    (both wall rows have distance 0)
    """
    rows = np.arange(ny)
    distance = np.minimum(ny - 1 - rows, rows).astype(float)
    return np.tile(distance, (nx, 1))


def signed_obstacle_distance(cell_type, obstacle):
    """
    creates the obstacle_sdf channel: the Euclidean distance (in grid points) of every point to the closest
    border point of the obstacle, negative inside the obstacle.
    A border point is an obstacle point with at least one neighbour (up, down, left, right) that is not a
    wall or obstacle point. The closest border point is found with a k-d tree (exact).
    """
    # imported here, building the other channels does not need scipy
    from scipy.spatial import cKDTree

    # neighbours outside of the grid are replaced by the point itself
    padded = np.pad(cell_type, 1, mode='edge')
    all_neighbours_solid = ((padded[2:, 1:-1] == OBSTACLE) & (padded[:-2, 1:-1] == OBSTACLE) &
                            (padded[1:-1, 2:] == OBSTACLE) & (padded[1:-1, :-2] == OBSTACLE))
    border_points = np.argwhere(obstacle & ~all_neighbours_solid)

    if len(border_points) == 0:
        raise ValueError("no obstacle found, the obstacle distance cannot be calculated")

    nx, ny = cell_type.shape
    grid_points = np.argwhere(np.ones((nx, ny), dtype=bool))
    distance, _ = cKDTree(border_points).query(grid_points)

    distance = distance.reshape(nx, ny)
    return np.where(obstacle, -1 * distance, distance)


def build_sample(ux, uy, p, valid):
    """
    creates the 6 channel array of a single case from its resampled fields ((nx, ny) arrays)
    :param valid: valid point mask of the resampling, 0 inside the obstacle
    """
    nx, ny = ux.shape
    sample = np.empty((6, nx, ny))
    sample[0], sample[1], sample[2] = ux, uy, p

    # grid points inside the obstacle are outside of the mesh: they are not valid and their values are 0
    obstacle = valid == 0

    cell_type = cell_type_mask(obstacle)
    sample[3] = wall_distance(nx, ny)
    sample[4] = cell_type
    sample[5] = signed_obstacle_distance(cell_type, obstacle)
    return sample
