"""
This module decides the size and position of an obstacle and creates its coordinates.

There are two sampling schemes, one for each part of the released dataset (see config.yaml):

* legacy : normal size cases, the original generator in create_obstacle_data_linux/
* jitter : smaller and larger cases, the meshForge generator

Every case has its own random generator, created by case_rng() from the global seed and the case name. So a single
case can be regenerated without generating all the cases before it, and base / 16x files of a case always share
the same geometry.
"""

import random

import numpy as np

from .create_geometries import (create_rectangle, create_diamond, create_left_triangle, create_right_triangle,
                                create_legacy_arcs, create_jitter_arcs)

# circle corners are created as a rectangle, their arcs are added in get_coordinates
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


def sample_legacy_dimensions(geometry, parameters, rng):
    """
    normal size: integer width and height, uniformly drawn from the ranges in config.yaml
    :return: width and height of the obstacle
    """
    shape = parameters[geometry]
    width = rng.randint(*shape['width'])
    height = rng.randint(*shape['height'])
    return width, height


def sample_jitter_dimensions(geometry, parameters, rng):
    """
    smaller / larger sizes: nominal width and height from config.yaml, changed by a random percentage
    :return: width and height of the obstacle
    """
    shape = parameters[geometry]
    width = shape['width'] * (1 + rng.uniform(-parameters['width_jitter'], parameters['width_jitter']))
    height = shape['height'] * (1 + rng.uniform(-parameters['height_jitter'], parameters['height_jitter']))
    return width, height


def get_origin(geometry, width, height, parameters, domain, rng):
    """
    calculates the lower left corner (before the random offsets) that places the obstacle in the middle of the domain
    """
    origin_x = (domain['length'] - width) / 2
    origin_y = (domain['height'] - height) / 2

    # meshForge placed diamonds and circles using the size of a rectangle, which was drawn (and jittered)
    # separately. This moves them a few mm away from the center. Kept so the smaller / larger cases follow
    # the same distribution as the released ones.
    if parameters['scheme'] == 'jitter' and geometry in ('diamond', 'circle'):
        reference_width, reference_height = sample_jitter_dimensions('rectangle', parameters, rng)
        origin_y = (domain['height'] - reference_height) / 2
        if geometry == 'diamond':
            origin_x = (domain['length'] - reference_width) / 2

    return [origin_x, origin_y]


def get_coordinates(geometry, parameters, domain, rng):
    """
    creates the coordinates of the selected geometry
    :param geometry: rectangle, diamond, left_triangle, right_triangle or circle
    :param parameters: size block of config.yaml (sizes: normal / smaller / larger)
    :param domain: domain block of config.yaml
    :param rng: random.Random instance of the case
    :return: dictionary with the sampled width and height, corner coordinates, arc mid-points (circle only)
     and the number of attempts needed to get a valid obstacle
    """
    if parameters['scheme'] == 'legacy':
        width, height = sample_legacy_dimensions(geometry, parameters, rng)
    elif parameters['scheme'] == 'jitter':
        width, height = sample_jitter_dimensions(geometry, parameters, rng)
    else:
        raise ValueError(f"Unknown sampling scheme: {parameters['scheme']}")

    origin = get_origin(geometry, width, height, parameters, domain, rng)
    variance = parameters[geometry]['variance']

    # draw new coordinates until the obstacle is a valid one (see check_geometry)
    attempts = 0
    while True:
        attempts += 1
        coordinates = SHAPE_FUNCTIONS[geometry](width, height, origin, variance, rng)

        arcs = None
        if geometry == 'circle' and parameters['scheme'] == 'legacy':
            arcs = create_legacy_arcs(coordinates, width, height, rng)
        elif geometry == 'circle':
            arcs = create_jitter_arcs(coordinates, rng)

        if check_geometry(obstacle_outline(coordinates, arcs), domain):
            return {'width': width, 'height': height, 'coordinates': coordinates, 'arcs': arcs, 'attempts': attempts}


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


def check_geometry(outline, domain):
    """
    basic checks every obstacle has to pass before it is written to a .geo file:
    * all points are inside the domain, not touching the walls
    * the points are in counter-clockwise order
    * the outline does not cross itself
    The original normal size generator checked the blockMesh mesh with checkMesh instead.
    """
    inside = np.all((outline[:, 0] > 0) & (outline[:, 0] < domain['length']) &
                    (outline[:, 1] > 0) & (outline[:, 1] < domain['height']))
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
