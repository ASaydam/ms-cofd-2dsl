"""
Helper functions shared by the pipeline stages: folders, config and the case manifest.

The manifest is a CSV file with one row per case. It records the obstacle coordinates (same names as in the
.geo files, in mm) and a few derived values that are useful for describing the dataset.
"""

import csv
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent

MANIFEST_FIELDS = [
    'case_name', 'source', 'source_file', 'size', 'geometry', 'index', 'seed',
    'sampled_width', 'sampled_height',
    'x1', 'x2', 'x3', 'x4', 'y1', 'y2', 'y3', 'y4',
    'z11', 'z21', 'z31', 'z41', 'z12', 'z22', 'z32', 'z42',
    'bbox_width', 'bbox_height', 'centroid_x', 'centroid_y', 'area', 'blockage_ratio',
]


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def load_config(config_path=REPO_ROOT / 'config.yaml'):
    """
    reads config.yaml, the folders under "paths" are turned into absolute paths relative to the config file
    """
    # imported here, so the tools that do not read config.yaml (e.g. extraction/check_runs.py) run without pyyaml
    import yaml

    config_path = Path(config_path).resolve()
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config['paths'] = {key: config_path.parent / value for key, value in config['paths'].items()}
    return config


def describe_obstacle(coordinates, arcs, domain):
    """
    calculates the bounding box, centroid, area and blockage ratio of an obstacle
    (circles are described by the polygon through their corners and arc mid-points)
    """
    # imported here, geometries imports this module as well
    from geometries.sampling import obstacle_outline, polygon_area

    outline = obstacle_outline(coordinates, arcs)
    x, y = outline[:, 0], outline[:, 1]

    area = polygon_area(outline)
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    centroid_x = np.sum((x + np.roll(x, -1)) * cross) / (6 * area)
    centroid_y = np.sum((y + np.roll(y, -1)) * cross) / (6 * area)

    bbox_height = y.max() - y.min()
    return {
        'bbox_width': round(float(x.max() - x.min()), 4),
        'bbox_height': round(float(bbox_height), 4),
        'centroid_x': round(float(centroid_x), 4),
        'centroid_y': round(float(centroid_y), 4),
        'area': round(float(area), 4),
        'blockage_ratio': round(float(bbox_height / domain['height']), 4),
    }


def manifest_row(case_name, size, geometry, index, coordinates, arcs, domain, **extra):
    """
    creates a manifest row, extra keyword arguments (source, seed, sampled_width, ...) are added as they are
    """
    row = {'case_name': case_name, 'size': size, 'geometry': geometry, 'index': index}

    for i, (x, y) in enumerate(coordinates):
        row[f'x{i + 1}'] = float(x)
        row[f'y{i + 1}'] = float(y)

    if arcs is not None:
        for i, (x, y) in enumerate(arcs):
            row[f'z{i + 1}1'] = float(x)
            row[f'z{i + 1}2'] = float(y)

    row.update(describe_obstacle(coordinates, arcs, domain))
    row.update(extra)
    return row


def write_csv(rows, file_path, fields):
    """
    writes a list of dictionaries into a CSV file, missing values are left empty
    """
    create_folder(Path(file_path).parent)
    with open(file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def read_csv(file_path):
    with open(file_path, 'r', newline='') as file:
        return list(csv.DictReader(file))


def update_manifest(rows, manifest_path):
    """
    adds the rows to the manifest, a case that is already in the manifest is replaced by its new row
    """
    manifest = {}
    if os.path.exists(manifest_path):
        manifest = {row['case_name']: row for row in read_csv(manifest_path)}

    for row in rows:
        manifest[row['case_name']] = row

    write_csv(list(manifest.values()), manifest_path, MANIFEST_FIELDS)
    print(f"Manifest has {len(manifest)} cases: {manifest_path}")
