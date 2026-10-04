"""
Builds the 6 channel arrays of the dataset from the resampled flow fields.

The fields come from extraction/resample_vtk.py (one case at a time, in the pipeline), or from the CSV files of
extraction/extract_fields.py (ParaView, how MS-COFD v4 was made; kept to compare both). A sample has the shape
(6, nx, ny), nx along the flow (inlet at x = 0), ny across the channel, and the channels of MS-COFD:

    0  u_x            x velocity (m/s)
    1  u_y            y velocity (m/s)
    2  p              kinematic pressure (m^2/s^2)
    3  wall_distance  distance to the closer channel wall, in grid points        (paper: Wall Distance)
    4  cell_type      1 = wall or obstacle, 2 = inlet, 3 = outlet, 4 = fluid     (paper: Cell Type Mask)
    5  obstacle_sdf   distance to the obstacle border, in grid points,           (paper: Signed Obstacle Distance)
                      negative inside the obstacle

The flow channels are the converged solution (last time step). The geometry channels are made from the obstacle
points of the same resampling: the grid points inside the obstacle (outside of the mesh) are not valid.
Extracts without a valid point mask (released v4 extracts) use the points with u_x == 0 as obstacle points.

build_arrays() builds the array of a whole folder of CSV extracts (<resolution>_tlast_6ch.npy and cases.csv).
"""

import csv
from pathlib import Path

import numpy as np

CHANNELS = ['u_x', 'u_y', 'p', 'wall_distance', 'cell_type', 'obstacle_sdf']

# values of the cell_type channel
WALL = OBSTACLE = 1
INLET = 2
OUTLET = 3
FLUID = 4


def read_fields(csv_file, nx, ny):
    """
    reads a CSV file exported by extract_fields.py
    :return: u_x, u_y, p and the valid point mask as (nx, ny) arrays, the mask is None for the released extracts
    """
    with open(csv_file, 'r') as file:
        header = [name.strip().strip('"') for name in file.readline().split(',')]
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1, ndmin=2)

    def column(name):
        # the rows go along x first, then y
        return data[:, header.index(name)].reshape(ny, nx).T

    valid = column('vtkValidPointMask') if 'vtkValidPointMask' in header else None
    return column('U:0'), column('U:1'), column('p'), valid


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
    creates the wall_distance channel: min(j, ny - j) for the grid row j, as in the released arrays
    (so the lower wall row has distance 0, the upper wall row distance 1)
    """
    rows = np.arange(ny)
    distance = np.minimum(ny - rows, rows).astype(float)
    return np.tile(distance, (nx, 1))


def signed_obstacle_distance(cell_type, obstacle):
    """
    creates the obstacle_sdf channel: the Euclidean distance (in grid points) of every point to the closest
    border point of the obstacle, negative inside the obstacle.
    A border point is an obstacle point with at least one neighbour (up, down, left, right) that is not a
    wall or obstacle point. The closest border point is found with a k-d tree (exact, the same values as
    comparing every point with every border point).
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


def build_sample(ux, uy, p, valid=None):
    """
    creates the 6 channel array of a single case from its resampled fields ((nx, ny) arrays)
    :param valid: valid point mask (0 inside the obstacle), None for extracts without it (obstacle where u_x == 0)
    """
    nx, ny = ux.shape
    sample = np.empty((6, nx, ny))
    sample[0], sample[1], sample[2] = ux, uy, p

    # grid points inside the obstacle are outside of the mesh: they are not valid and their values are 0
    obstacle = (valid == 0) if valid is not None else (ux == 0)

    cell_type = cell_type_mask(obstacle)
    sample[3] = wall_distance(nx, ny)
    sample[4] = cell_type
    sample[5] = signed_obstacle_distance(cell_type, obstacle)
    return sample


def build_case(csv_file, nx, ny):
    """
    creates the 6 channel array of a single case from a CSV extract
    """
    return build_sample(*read_fields(csv_file, nx, ny))


def build_arrays(extract_directory, output_directory, resolution, nx, ny, dtype='float32'):
    """
    builds the array of all the exported cases of one resolution from CSV extracts (extract_fields.py)
    :param extract_directory: folder of the CSV files (output of extract_fields.py)
    :param output_directory: folder of the .npy file
    :param resolution: base or 16x, used in the file name
    :param nx: number of grid points along the flow (260 for base, 1040 for 16x)
    :param ny: number of grid points across the channel (120 for base, 480 for 16x)
    :param dtype: float32 (default) or float64 (the type of mscofd2000_v4.h5, twice the size). The exported
     fields have 5 significant digits (OpenFOAM writes 6), float32 keeps all of them.
    """
    extract_directory, output_directory = Path(extract_directory), Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(extract_directory.glob("*_last_time.csv"), key=lambda path: path.name)
    cases = [(csv_file.name[:-len("_last_time.csv")], csv_file) for csv_file in csv_files]
    if not cases:
        raise FileNotFoundError(f"No exported cases found in {extract_directory}")
    print(f"Building the {resolution} array of {len(cases)} cases ({nx} x {ny})")

    array = np.lib.format.open_memmap(output_directory / f"{resolution}_tlast_6ch.npy", mode='w+', dtype=dtype,
                                      shape=(len(cases), 6, nx, ny))
    for i, (case_name, csv_file) in enumerate(cases):
        array[i] = build_case(csv_file, nx, ny)
        if (i + 1) % 50 == 0 or i + 1 == len(cases):
            print(f"{i + 1} / {len(cases)} cases done")
    array.flush()

    with open(output_directory / "cases.csv", 'w', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(['row', 'case_name'])
        writer.writerows((i, case_name) for i, (case_name, _) in enumerate(cases))

    print(f"Array written to {output_directory / f'{resolution}_tlast_6ch.npy'}")
