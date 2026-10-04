import math

from tools.mesh_study import observed_order, study


def value(h, order, exact=1.0, constant=0.3):
    """attribute of a mesh with spacing h and a discretization error of the given order"""
    return exact + constant * h ** order


def test_second_order_values_are_extrapolated_to_the_exact_value():
    phi = [value(h, 2) for h in (0.25, 0.5, 1.0)]  # 16x, 4x, base
    cells = [16.0, 4.0, 1.0]
    result = study(*phi, *cells)
    assert result['convergence'] == 'monotonic'
    assert math.isclose(result['order'], 2.0, rel_tol=1e-6)
    assert math.isclose(result['extrapolated'], 1.0, rel_tol=1e-9)
    assert result['16x_error_%'] < result['4x_error_%'] < result['base_error_%']


def test_observed_order_with_unequal_refinement_ratios():
    r21, r32 = 2.1, 1.9
    h1 = 0.25
    phi1, phi2, phi3 = value(h1, 1.5), value(h1 * r21, 1.5), value(h1 * r21 * r32, 1.5)
    assert math.isclose(observed_order(phi1, phi2, phi3, r21, r32), 1.5, rel_tol=1e-6)


def test_growing_differences_are_not_converged():
    # a peak value at a sharp corner: the differences grow with refinement
    result = study(4.0, 2.0, 1.0, 16.0, 4.0, 1.0)
    assert result['convergence'] == 'none'
    assert 'extrapolated' not in result
