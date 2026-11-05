import numpy as np


def line(ux):
    """a wake line as tools/extract_line_samples.py stores it: rows (y in m, p, u_x, u_y), float32, 121 points"""
    y = np.linspace(0, 0.12, 121)
    return np.column_stack([y, np.zeros_like(y), ux, np.zeros_like(y)]).astype(np.float32)


def test_wake_row_takes_the_minimum_behind_the_obstacle():
    from tools.wake_band import wake_row

    y = np.linspace(0, 120, 121)
    ux = 0.15 - 0.1229286 * np.exp(-((y - 60) / 10) ** 2)                        # trough of 0.0270714 at y = 60 mm
    ux[[0, -1]] = 0.0                                                            # no-slip walls
    row = wake_row('16x', 'normal_circle_0001', line(ux), gap_bottom=40.0, gap_top=40.0, height=120.0, value=0.0)
    assert row['line_min'] == 0.0 and row['metadata'] == 0.0                     # the production value: the wall
    assert row['band_min'] == 0.0270714 and row['band_y_mm'] == 60.0             # 6 significant digits, not float32
    assert row['band_points'] == 41 and not row['at_edge']


def test_wake_row_keeps_reverse_flow():
    from tools.wake_band import wake_row

    y = np.linspace(0, 120, 121)
    ux = 0.15 - 0.17 * np.exp(-((y - 60) / 10) ** 2)                             # reverse flow behind the obstacle
    ux[[0, -1]] = 0.0
    row = wake_row('16x', 'normal_circle_0001', line(ux), 40.0, 40.0, 120.0, -0.02)
    assert row['band_min'] == row['line_min'] == round(0.15 - 0.17, 6)


def test_wake_row_keeps_the_edge_point():
    """120 - 64.65 is 55.349999999999994 in double precision; the line point at 55.35 mm is still in the extent"""
    from tools.wake_band import wake_row

    y = np.round(np.arange(0, 120.0001, 0.05), 2)
    ux = np.full_like(y, 0.1)
    ux[[0, -1]] = 0.0
    wake = np.column_stack([y / 1000, np.zeros_like(y), ux, np.zeros_like(y)]).astype(np.float32)
    row = wake_row('16x', 'smaller_diamond_0170', wake, gap_bottom=40.0, gap_top=64.65, height=120.0, value=0.0)
    assert row['band_points'] == round((55.35 - 40.0) / 0.05) + 1
