"""
This module provides the functions that create the corner coordinates of the obstacles.

Supported shapes:

* Rectangle
* Diamond
* Left triangle (pointy edge on the left, facing the inlet)
* Right triangle (pointy edge on the right, facing the outlet)
* Circle (a rectangle whose sides are replaced by splines, see create_legacy_arcs / create_jitter_arcs)

Every function takes a `rng` argument, a `random.Random` instance, instead of using the global random module.
This way every case can be regenerated from its seed (see geometries/sampling.py).

The shape functions are the ones used to create the released dataset (create_obstacle_data_linux/ and meshForge/),
only the random number generator is passed in and the retry logic is written as loops.
"""

import numpy as np


def create_variance(max_value, variance, rng):
    """
    creates a random offset using variance
    :param max_value: x and y values will always be in a circle with r = max_value
    :param variance: variance will change the max value
    :param rng: random.Random instance
    :return: list of the randomly generated coordinates [x , y]
    """
    r = max_value * (1 - variance)  # calculate max r
    x = r * rng.randint(-100, 100) / 100  # calculate a random x
    y = np.sqrt(abs(r ** 2 - x ** 2)) * rng.randint(-100, 100) / 100  # calculate a random y
    return [x, y]


def create_rectangle(width, height, origin, variance, rng):
    """
    creates a random rectangle using variance value
    :param width: width of the rectangle
    :param height: height of the rectangle
    :param origin: lower left corner of the rectangle before the random offsets
    :param variance: width and height of the rectangle can vary as much as this value
    :param rng: random.Random instance
    :return: list of the coordinates of corner points,
     starting with lower left and going in a counter-clockwise order
    """
    np_coords = np.empty((4, 2))

    for i in range(2):
        for j in range(2):
            random_point = create_variance(width / 5, variance, rng)

            x = int((origin[0] + random_point[0]) + ((i + j) % 2 * width))
            y = int((origin[1] + random_point[1]) + (i * height))

            np_coords[i * 2 + j, 0] = x
            np_coords[i * 2 + j, 1] = y

    return np_coords


def create_diamond(width, height, origin, variance, rng):
    """
    creates a diamond shape using variance value
    :param width: width of the diamond
    :param height: height of the diamond
    :param origin: lower left corner of the bounding box before the random offsets
    :param variance: corners can move as much as this value
    :param rng: random.Random instance
    :return: list of the coordinates of corner points,
     starting with the bottom corner and going in a counter-clockwise order
    """
    np_coords = np.empty((4, 2))

    multiplier = [[0, -1], [1, 0], [0, 1], [-1, 0]]

    for i in range(4):
        random_point = create_variance(width / 10, variance, rng)

        x = int((origin[0] + random_point[0]) + (multiplier[i][0] * width / 2))
        y = int((origin[1] + random_point[1]) + (multiplier[i][1] * height / 2))

        np_coords[i, 0] = x + width / 2   # adding width / 2 for centering the object
        np_coords[i, 1] = y + height / 2  # adding height / 2 for centering the object

    return np_coords


def create_left_triangle(width, height, origin, variance, rng):
    """
    creates a random triangle pointing to the inlet, using a random rectangle
    :param width: width of the triangle
    :param height: height of the triangle
    :param origin: lower left corner of the bounding box before the random offsets
    :param variance: corners can move as much as this value
    :param rng: random.Random instance
    :return: list of the coordinates of corner points,
     starting with the pointy edge and going in a counter-clockwise order
    """
    np_coords = np.empty((3, 2))

    # making sure that pointy edge is between upper and lower point, draw a new rectangle if it is not
    while True:
        temp_coords = create_rectangle(width, height, origin, variance, rng)

        np_coords[0] = 0.5 * (temp_coords[0] + temp_coords[3])
        np_coords[1:] = temp_coords[1:3]

        if np_coords[0, 1] == np.median(np_coords[:, 1]):
            return np_coords


def create_right_triangle(width, height, origin, variance, rng):
    """
    creates a random triangle pointing to the outlet, using a random rectangle
    :param width: width of the triangle
    :param height: height of the triangle
    :param origin: lower left corner of the bounding box before the random offsets
    :param variance: corners can move as much as this value
    :param rng: random.Random instance
    :return: list of the coordinates of corner points,
     starting with lower left and going in a counter-clockwise order
    """
    np_coords = np.empty((3, 2))

    # making sure that pointy edge is between upper and lower point, draw a new rectangle if it is not
    # (the original code called create_left_triangle here by mistake)
    while True:
        temp_coords = create_rectangle(width, height, origin, variance, rng)

        np_coords[1] = 0.5 * (temp_coords[1] + temp_coords[2])
        np_coords[0] = temp_coords[0]
        np_coords[2] = temp_coords[3]

        if np_coords[1, 1] == np.median(np_coords[:, 1]):
            return np_coords


def create_legacy_arcs(coordinates, width, height, rng):
    """
    creates the arc mid-points of a circular obstacle the way the original blockMeshDict generator did (normal size).
    Top and bottom sides bulge 5 mm, left and right sides bulge width / 20, each one randomly inwards or outwards.
    :param coordinates: corner points of the circle, output of create_rectangle
    :param width: width parameter of the circle (before the random corner offsets)
    :param height: height parameter of the circle (before the random corner offsets)
    :param rng: random.Random instance
    :return: arc mid-points as a (4, 2) array in the order bottom, right, top, left
    """
    random_sign = [[-1, 1][rng.randrange(2)] for i in range(4)]

    lower_left, lower_right, upper_right, upper_left = coordinates
    lb_width = int(upper_left[0])  # left-block width of the blockMeshDict

    bottom = [lb_width + width / 2, lower_right[1] + random_sign[0] * 5]
    right = [upper_right[0] + random_sign[1] * (width / 20), upper_right[1] - height / 2]
    top = [lb_width + width / 2, upper_right[1] + random_sign[2] * 5]
    # the original code used the sign of the top arc for the left arc as well (random_sign[3] was never used),
    # kept as it is so the generated circles follow the same distribution as the released ones
    left = [lower_left[0] + random_sign[2] * (width / 20), upper_left[1] - height / 2]

    return np.array([bottom, right, top, left])


def create_jitter_arcs(coordinates, rng):
    """
    creates the arc mid-points of a circular obstacle the way meshForge did (smaller and larger sizes).
    Every mid-point is the middle of its side plus a random offset, rounded to 0.1 mm.
    :param coordinates: corner points of the circle, output of create_rectangle
    :param rng: random.Random instance
    :return: arc mid-points as a (4, 2) array in the order bottom, right, top, left
    """
    arcs = np.empty((4, 2))

    (x1, y1), (x2, y2), (x3, y3), _ = coordinates
    width = abs(x2 - x1)
    height = abs(y3 - y2)

    # sides in the order bottom (1-2), right (2-3), top (3-4), left (4-1)
    for i in range(4):
        start, end = coordinates[i], coordinates[(i + 1) % 4]
        arcs[i, 0] = round((start[0] + end[0]) / 2 + create_variance(width / 10, 0.10, rng)[0], 1)
        arcs[i, 1] = round((start[1] + end[1]) / 2 + create_variance(height / 10, 0.10, rng)[1], 1)

    return arcs
