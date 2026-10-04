"""
This module writes the obstacle coordinates into .geo files for gmsh (and reads them back).

A .geo file is made of three parts:

1. geo_base_1.geo            : meshing algorithm, unit and the characteristic mesh length (size) of the resolution
2. obstacle coordinates      : x1, y1, ... and z11, z12, ... (arc mid-points, circles only), in mm
3. geo_base_2_<geometry>.geo : obstacle and domain outlines, extrusion and boundary names

The resolutions of a case (base, 4x, 16x) are written from the same coordinates, their files differ only in
the size line.
"""

import re
from itertools import product
from pathlib import Path

from utils import create_folder, manifest_row
from .sampling import case_rng, get_coordinates

GEO_BASE_FOLDER = Path(__file__).resolve().parent / 'geo-base-files'

RE_SIZE = re.compile(r'^size = .*$', re.MULTILINE)
RE_VARIABLE = re.compile(r'^([xyz]\d+)\s*=\s*([-\d.eE+]+)\s*\*\s*unit;', re.MULTILINE)


def file_selector(geometry):
    return {
        'geo_base_file': GEO_BASE_FOLDER / 'geo_base_1.geo',
        'rectangle': GEO_BASE_FOLDER / 'geo_base_2_rectangle_diamond.geo',
        'diamond': GEO_BASE_FOLDER / 'geo_base_2_rectangle_diamond.geo',
        'left_triangle': GEO_BASE_FOLDER / 'geo_base_2_triangle.geo',
        'right_triangle': GEO_BASE_FOLDER / 'geo_base_2_triangle.geo',
        'circle': GEO_BASE_FOLDER / 'geo_base_2_circular.geo',
    }[geometry]


def write_file(file, geometry, coordinates, arcs=None):
    """
    This function takes geometry information as arguments and writes the coordinate lines of the .geo file.
    :param file: opened .geo file
    :param geometry: rectangle, diamond, left_triangle, right_triangle or circle
    :param coordinates: corner points, one [x, y] row per corner
    :param arcs: arc mid-points of a circle, one [x, y] row per side (bottom, right, top, left)
    """
    xs = [float(x) for x in coordinates[:, 0]]
    ys = [float(y) for y in coordinates[:, 1]]

    # x1, x2, ... first, then y1, y2, ...
    for i, x in enumerate(xs):
        file.write(f"x{i + 1} = {x} * unit;\n")
    for i, y in enumerate(ys):
        file.write(f"y{i + 1} = {y} * unit;\n")

    # circle arc mid-points : z11 z21 z31 z41 are the x values, z12 z22 z32 z42 are the y values
    if geometry == "circle":
        for i, x in enumerate(arcs[:, 0]):
            file.write(f"z{i + 1}1 = {float(x)} * unit;\n")
        for i, y in enumerate(arcs[:, 1]):
            file.write(f"z{i + 1}2 = {float(y)} * unit;\n")


def write_geo(file_path, geometry, coordinates, arcs, mesh_size):
    """
    creates a single .geo file
    :param file_path: output file
    :param geometry: rectangle, diamond, left_triangle, right_triangle or circle
    :param coordinates: corner points of the obstacle
    :param arcs: arc mid-points of a circle, None for other geometries
    :param mesh_size: characteristic mesh length (gmsh size) of the resolution, in meters
    """
    # geo_base_1 file contains meshing algorithm info, only its size line changes between resolutions
    header = file_selector('geo_base_file').read_text()
    header = RE_SIZE.sub(f"size = {mesh_size}; // base: 0.001 - 4x: 0.0005 - 16x: 0.00025", header, count=1)

    # newline='\n' keeps the files identical on Windows and Linux
    with open(file_path, 'w', newline='\n') as file:
        file.write(header)
        write_file(file, geometry, coordinates, arcs)
        file.write(file_selector(geometry).read_text())


def read_geo_coordinates(file_path):
    """
    reads the coordinate lines (x1 = 104.0 * unit; ...) and the size of a .geo file
    :return: dictionary of variable name -> value in mm, and the mesh size (None if not found)
    """
    text = Path(file_path).read_text()
    variables = {name: float(value) for name, value in RE_VARIABLE.findall(text)}

    size = re.search(r'^size = ([\d.eE+-]+);', text, re.MULTILINE)
    return variables, float(size.group(1)) if size else None


def create_geo(config, sizes, geometries, resolutions, start=1, end=None):
    """
    creates the .geo files of the selected cases, every case is written once for every resolution
    :param config: content of config.yaml (see utils.load_config)
    :param sizes: list of sizes (normal, smaller, larger)
    :param geometries: list of geometries
    :param resolutions: list of resolutions (base, 4x, 16x)
    :param start: first case number
    :param end: last case number, cases_per_geometry of config.yaml if None
    :return: manifest rows of the created cases
    """
    generated_rows = []

    for resolution in resolutions:
        create_folder(config['paths']['geo_files'] / resolution)

    for size, geometry in product(sizes, geometries):
        last_case = end if end is not None else config['cases_per_geometry'][size]

        for i in range(start, last_case + 1):
            case_name = f"{size}_{geometry}_{i}"

            # every case has its own random generator, see sampling.case_rng
            rng = case_rng(config['seed'], size, geometry, i)
            obstacle = get_coordinates(geometry, config['sizes'][size], config['domain'], rng)

            for resolution in resolutions:
                geo_file = config['paths']['geo_files'] / resolution / f"{case_name}.geo"
                write_geo(geo_file, geometry, obstacle['coordinates'], obstacle['arcs'],
                          config['resolutions'][resolution])

            generated_rows.append(manifest_row(case_name, size, geometry, i, obstacle['coordinates'],
                                               obstacle['arcs'], config['domain'],
                                               source='generated', seed=config['seed'],
                                               sampled_width=obstacle['width'],
                                               sampled_height=obstacle['height']))

        print(f"Created {last_case - start + 1} {size} {geometry} cases")

    return generated_rows
