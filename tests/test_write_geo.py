from geometries.write_geo import create_geo, read_geo_coordinates
from utils import read_csv, update_manifest


def generate(config, resolutions=('base', '16x'), end=2):
    return create_geo(config, list(config['cases_per_geometry']), config['geometries'], list(resolutions),
                      start=1, end=end)


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


def test_domain_size_is_written(config):
    rows = generate(config, resolutions=('base',), end=1)
    text = (config['paths']['geo_files'] / 'base' / f"{rows[0]['case_name']}.geo").read_text()
    assert f"domain_length = {config['domain']['length']} * unit;" in text
    assert f"domain_height = {config['domain']['height']} * unit;" in text
    assert rows[0]['case_name'].endswith('_0001')
