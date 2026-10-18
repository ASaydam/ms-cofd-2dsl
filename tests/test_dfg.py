import math

import pytest

from validation.dfg import dfg


def test_geo_has_the_patches_and_the_size():
    text = dfg.geo_text(0.005)
    assert 'size = 0.005;' in text and 'Mesh.Algorithm = 8;' in text and 'Layers{1};' in text
    for patch in ('frontAndBack', 'outlet', 'fixedWalls', 'cylinder', 'inlet'):
        assert f'Physical Surface("{patch}")' in text
    assert 'Point(3) = {0.15, 0.2, 0, size};' in text   # front point of the cylinder


def test_setup_keeps_the_production_solver_files(tmp_path):
    case = tmp_path / 'coarse'
    dfg.setup(case, 0.005)
    for part in ('system/fvSchemes', 'system/fvSolution'):
        assert (case / part).read_text() == (dfg.TEMPLATE / part).read_text()
    assert 'endTime         50000;' in (case / 'system' / 'controlDict').read_text()
    assert 'nu              1.0e-03;' in (case / 'constant' / 'transportProperties').read_text()
    assert 'cylinder' in (case / '0' / 'U').read_text() and 'cylinder' in (case / '0' / 'p').read_text()


def test_inlet_parabola_at_the_face_centres(tmp_path):
    case = tmp_path / 'coarse'
    dfg.setup(case, 0.005)
    centres = [(0.0, y, 0.005) for y in (0.005, 0.205, 0.405)]
    c_text = ("boundaryField\n{\n    inlet\n    {\n        type            calculated;\n"
              "        value           nonuniform List<vector> 3\n(\n" +
              '\n'.join(f"({x} {y} {z})" for x, y, z in centres) + "\n)\n;\n    }\n}\n")
    assert dfg.inlet_centres(c_text) == centres
    text = dfg.set_inlet((case / '0' / 'U').read_text(), centres)
    assert 'nonuniform List<vector> 3' in text and 'noSlip' in text
    assert abs(dfg.parabola(0.205) - 0.3) < 1e-12 and abs(dfg.parabola(0.0)) < 1e-12
    # mean of the parabola over the height: 2/3 of the maximum = 0.2 m/s
    ys = [i * dfg.HEIGHT / 10000 for i in range(10001)]
    assert abs(sum(dfg.parabola(y) for y in ys) / len(ys) - dfg.U_MEAN) < 1e-4


def test_pressure_interpolated_in_the_angle():
    rows = []
    for k in range(36):
        angle = (k + 0.5) * 2 * math.pi / 36   # no face exactly at 0 or pi
        rows.append([0.2 + 0.05 * math.cos(angle), 0.2 + 0.05 * math.sin(angle), 0.005, math.cos(angle)])
    # p = cos(angle): front (pi) -1, rear (0) +1, up to the linear interpolation error
    assert dfg.pressure_at(rows, math.pi) == pytest.approx(-1, abs=5e-3)
    assert dfg.pressure_at(rows, 0.0) == pytest.approx(1, abs=5e-3)


def test_recirculation_length():
    x = [0.25 + i * 0.001 for i in range(201)]
    ux = [0.0] + [(xi - 0.3347) for xi in x[1:]]   # zero on the wall, negative up to x = 0.3347
    assert dfg.recirculation_length(x, ux) == pytest.approx(0.0847, abs=1e-9)
    assert dfg.recirculation_length(x, [1.0] * len(x)) == 0.0


def test_force_coefficients_per_unit_depth():
    # F = cD * 0.5 * U_mean^2 * D per unit depth, times the depth of the mesh
    fx = 5.57954 * 0.5 * 0.2 ** 2 * 0.1 * dfg.DEPTH
    fy = 0.010619 * 0.5 * 0.2 ** 2 * 0.1 * dfg.DEPTH
    cd, cl = dfg.force_coefficients([1000, fx, fy, 0])
    assert cd == pytest.approx(5.57954) and cl == pytest.approx(0.010619)
