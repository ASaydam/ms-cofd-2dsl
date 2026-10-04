"""
This module decides the size, rotation and position of an obstacle and creates its coordinates.

Every case is drawn in four steps (all ranges are in config/dataset.yaml):

1. width and height: the nominal size of the geometry times the scale of the size class (normal / smaller /
   larger), changed by a random percentage
2. the corner points (and the arc mid-points of circles) around the nominal center, every corner moved by a
   random offset (see create_geometries.py)
3. a random rotation about the centroid of the obstacle and a random shift of the whole obstacle
4. checks: the obstacle keeps the minimum gaps to the walls, the inlet and the outlet, its points are in
   counter-clockwise order and its outline does not cross itself. An obstacle that fails a check is drawn again,
   starting from step 1, and the number of attempts is recorded in the manifest.

Every case has its own random generator, created by case_rng() from the global seed and the case name. So a single
case can be regenerated without generating all the cases before it, and the resolutions of a case always share
the same geometry.
"""

import math
import random

import numpy as np

from .create_geometries import (create_rectangle, create_diamond, create_left_triangle, create_right_triangle,
                                create_arcs)

# circle corners are created as a rectangle, their arcs are added in draw_obstacle
SHAPE_FUNCTIONS = {
    'rectangle': create_rectangle,
    'diamond': create_diamond,
    'left_triangle': create_left_triangle,
    'right_triangle': create_right_triangle,
    'circle': create_rectangle,
}


def case_rng(seed, size, geometry, index):
    """
    creates the random generator of a single case
    (seeding with a string is deterministic in Python, it does not depend on PYTHONHASHSEED)
    """
    return random.Random(f"{seed}-{size}-{geometry}-{index}")


def sample_dimensions(geometry, size, sizes, rng):
    """
    nominal width and height of the geometry, scaled for the size class and changed by a random percentage
    :param geometry: rectangle, diamond, left_triangle, right_triangle or circle
    :param size: normal, smaller or larger
    :param sizes: sizes block of config/dataset.yaml
    :param rng: random.Random instance of the case
    :return: width and height of the obstacle
    """
    nominal = sizes['nominal'][geometry]
    scale = sizes['scale'][size]
    width = nominal['width'] * scale * (1 + rng.uniform(-sizes['width_jitter'], sizes['width_jitter']))
    height = nominal['height'] * scale * (1 + rng.uniform(-sizes['height_jitter'], sizes['height_jitter']))
    return width, height


def get_origin(width, height, center):
    """
    calculates the lower left corner (before the random offsets) that places the obstacle around the nominal center
    """
    return [center[0] - width / 2, center[1] - height / 2]


def polygon_centroid(points):
    """
    centroid of a polygon (its points in counter-clockwise order)
    """
    x, y = points[:, 0], points[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = 0.5 * np.sum(cross)
    return np.array([np.sum((x + np.roll(x, -1)) * cross) / (6 * area),
                     np.sum((y + np.roll(y, -1)) * cross) / (6 * area)])


def rotate_and_shift(points, angle, shift, pivot):
    """
    rotates points about the pivot by angle (degrees, counter-clockwise) and moves them by shift
    """
    angle = math.radians(angle)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return (np.asarray(points, dtype=float) - pivot) @ rotation.T + pivot + np.asarray(shift, dtype=float)


def draw_obstacle(geometry, size, config, rng):
    """
    draws one obstacle (steps 1-3 of the module docstring), it is not checked yet
    :return: dictionary with the sampled width, height, rotation and shift, the corner coordinates and the arc
     mid-points (circles only, None otherwise), coordinates rounded to 0.01 mm
    """
    placement = config['placement']
    width, height = sample_dimensions(geometry, size, config['sizes'], rng)
    origin = get_origin(width, height, placement['center'])
    variance = config['sizes']['nominal'][geometry]['variance']

    coordinates = SHAPE_FUNCTIONS[geometry](width, height, origin, variance, rng)
    arcs = create_arcs(coordinates, rng) if geometry == 'circle' else None

    rotation = rng.uniform(*placement['rotation'])
    shift = [rng.uniform(*placement['shift_x']), rng.uniform(*placement['shift_y'])]

    # corners and arc mid-points are rotated together, about the centroid of the outline
    pivot = polygon_centroid(obstacle_outline(coordinates, arcs))
    coordinates = np.round(rotate_and_shift(coordinates, rotation, shift, pivot), 2)
    if arcs is not None:
        arcs = np.round(rotate_and_shift(arcs, rotation, shift, pivot), 2)

    return {'width': width, 'height': height, 'rotation': rotation, 'shift_x': shift[0], 'shift_y': shift[1],
            'coordinates': coordinates, 'arcs': arcs}


def get_coordinates(geometry, size, config, rng):
    """
    creates the coordinates of the selected geometry
    :param geometry: rectangle, diamond, left_triangle, right_triangle or circle
    :param size: normal, smaller or larger
    :param config: content of config/dataset.yaml (sizes, placement and domain blocks are used)
    :param rng: random.Random instance of the case
    :return: dictionary of draw_obstacle, plus the number of attempts needed to get a valid obstacle
    """
    max_attempts = config['placement']['max_attempts']

    # draw new obstacles until one is valid (see check_geometry)
    for attempts in range(1, max_attempts + 1):
        obstacle = draw_obstacle(geometry, size, config, rng)
        if check_geometry(obstacle_outline(obstacle['coordinates'], obstacle['arcs']), config['domain'],
                          config['placement']):
            obstacle['attempts'] = attempts
            return obstacle

    raise RuntimeError(f"no valid {size} {geometry} obstacle in {max_attempts} attempts, check the placement settings")


def obstacle_outline(coordinates, arcs=None):
    """
    returns the outline of the obstacle as a polygon, for circles the arc mid-points are put between the corners
    """
    if arcs is None:
        return np.asarray(coordinates, dtype=float)

    outline = []
    for corner, arc in zip(coordinates, arcs):
        outline.append(corner)
        outline.append(arc)
    return np.asarray(outline, dtype=float)


def polygon_area(points):
    """
    signed area of a polygon (shoelace formula), positive when the points are in counter-clockwise order
    """
    x, y = points[:, 0], points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def segments_intersect(p1, p2, q1, q2):
    """
    checks if the line segments p1-p2 and q1-q2 cross each other
    """
    def orientation(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    return (orientation(p1, p2, q1) != orientation(p1, p2, q2)
            and orientation(q1, q2, p1) != orientation(q1, q2, p2))


def check_geometry(outline, domain, placement):
    """
    checks every obstacle has to pass before it is written to a .geo file:
    * the minimum gaps to the top and bottom walls, the inlet and the outlet are kept
    * the points are in counter-clockwise order
    * the outline does not cross itself
    """
    x, y = outline[:, 0], outline[:, 1]
    inside = (y.min() >= placement['min_wall_gap'] and y.max() <= domain['height'] - placement['min_wall_gap']
              and x.min() >= placement['min_inlet_gap'] and x.max() <= domain['length'] - placement['min_outlet_gap'])
    if not inside or polygon_area(outline) <= 0:
        return False

    # compare every side with the sides that are not its neighbours
    n = len(outline)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if segments_intersect(outline[i], outline[(i + 1) % n], outline[j], outline[(j + 1) % n]):
                return False

    return True
