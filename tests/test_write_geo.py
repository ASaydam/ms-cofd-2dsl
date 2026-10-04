import re

import pytest

from conftest import RELEASED_EXTENDED_GEO
from geometries.write_geo import create_geo, read_geo_coordinates
from utils import read_csv, update_manifest

RE_COORDINATE_OR_SIZE = re.compile(r'^([xyz]\d+ = |size = )')


def generate(config, resolutions=('base', '16x'), end=2):
    return create_geo(config, list(config['sizes']), config['geometries'], list(resolutions), start=1, end=end)


def test_resolutions_differ_only_in_the_size_line(config):
    generate(config, resolutions=('base', '4x', '16x'))
    base_files = sorted((config['paths']['geo_files'] / 'base').glob("*.geo"))
    assert len(base_files) == 3 * 5 * 2

    for base_file in base_files:
        base_lines = base_file.read_text().splitlines()
        for resolution, size in (('4x', 0.0005), ('16x', 0.00025)):
            other_lines = (config['paths']['geo_files'] / resolution / base_file.name).read_text().splitlines()
            different = [i for i, (a, b) in enumerate(zip(base_lines, other_lines)) if a != b]
            assert len(base_lines) == len(other_lines)
            assert len(different) == 1 and other_lines[different[0]].startswith(f"size = {size};")


def test_same_seed_gives_identical_files(config, tmp_path):
    generate(config)
    first = {path.name: path.read_bytes() for path in (config['paths']['geo_files'] / 'base').glob("*.geo")}

    config['paths']['geo_files'] = tmp_path / "second-run"
    generate(config)
    second = {path.name: path.read_bytes() for path in (config['paths']['geo_files'] / 'base').glob("*.geo")}

    assert first == second


def test_geo_coordinates_match_the_manifest(config):
    rows = generate(config)
    for row in rows:
        variables, size = read_geo_coordinates(config['paths']['geo_files'] / 'base' / f"{row['case_name']}.geo")
        assert size == config['resolutions']['base']
        for name, value in variables.items():
            assert value == row[name]


def test_manifest_update_replaces_cases(config):
    rows = generate(config, end=1)
    update_manifest(rows, config['paths']['manifest'])
    update_manifest(rows[:3], config['paths']['manifest'])
    assert [row['case_name'] for row in read_csv(config['paths']['manifest'])] == [row['case_name'] for row in rows]


@pytest.mark.skipif(not RELEASED_EXTENDED_GEO.exists(), reason="released smaller / larger .geo files not found")
@pytest.mark.parametrize("geometry", ["rectangle", "diamond", "left_triangle", "right_triangle", "circle"])
def test_geo_template_matches_the_released_files(config, geometry):
    """apart from coordinates and size, a generated file is the same as a released one"""
    generate(config, resolutions=('base',), end=1)

    def template_lines(path):
        lines = path.read_text().replace('\r', '').splitlines()
        return [line for line in lines if not RE_COORDINATE_OR_SIZE.match(line)]

    generated = template_lines(config['paths']['geo_files'] / 'base' / f"larger_{geometry}_1.geo")
    released = template_lines(RELEASED_EXTENDED_GEO / f"larger_{geometry}_1.geo")
    assert generated == released
