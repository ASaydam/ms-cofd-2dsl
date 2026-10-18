import numpy as np
import pytest

from tools.uncertainty import ARRAY_ATTRIBUTES, RUNS, array_study, scale_of, uncertainty

H = np.array([1.0, 2.0, 4.0, 8.0])


def run(*rows):
    phi = np.array(rows, dtype=float)
    return uncertainty(np.tile(H, (len(rows), 1)), phi)


def test_known_behaviour_in_one_call():
    rows = [1 + 0.1 * H ** 1.5,            # second-order-like convergence (p = 1.5)
            1 + 0.1 * H ** 0.3,            # slow convergence
            1 + 0.1 / H,                   # differences grow with refinement: no convergence
            np.full(4, 2.0)]               # all levels equal
    result = run(*rows)
    assert list(result['method']) == ['power', 'low_order', 'anomalous', 'equal']

    assert result['p'][0] == pytest.approx(1.5, abs=0.02)
    assert result['phi_0'][0] == pytest.approx(1.0, abs=1e-3)
    assert result['U'][0] == pytest.approx(1.25 * 0.1, rel=1e-3)    # exact data: Fs times the error at h = 1

    assert result['U'][1] > 0.1 * 0.3                               # wide for slow convergence (error at h = 1: 0.1)
    assert result['U'][2] == pytest.approx(3 * (rows[2].max() - rows[2].min()))
    assert result['U'][3] == 0


def test_scatter_widens_the_uncertainty():
    smooth = 1 + 0.1 * H ** 2
    scattered = smooth + np.array([0.0, 0.004, -0.004, 0.004])
    u_smooth, u_scattered = run(smooth, scattered)['U']
    assert u_scattered > u_smooth


def test_array_study_reads_every_level(tmp_path):
    pd = pytest.importorskip("pandas")
    cases = ['normal_circle_0001', 'normal_circle_0002']
    for k, run in enumerate(RUNS.values()):
        table = {'case_name': cases[::-1] if k else cases}             # row order differs between the files
        for name in ARRAY_ATTRIBUTES:
            fine = {'normal_circle_0001': 1.0, 'normal_circle_0002': 2.0}
            table[f'{name}_16x_array'] = [fine[c] for c in table['case_name']]
            table[f'{name}_{run}_array'] = [fine[c] + 0.1 * (k + 1) for c in table['case_name']]
        pd.DataFrame(table).to_csv(tmp_path / f"flow_attributes_{run}_16x.csv", index=False)

    study = array_study(tmp_path).set_index(['attribute', 'case_name'])
    assert len(study) == 2 * len(ARRAY_ATTRIBUTES)
    row = study.loc[('dissipation', 'normal_circle_0002')]
    assert (row['16x'], row['4x'], row['base'], row['0.25x']) == pytest.approx((2.0, 2.1, 2.2, 2.3))

    table = pd.read_csv(tmp_path / "flow_attributes_4x_16x.csv")
    table['dissipation_16x_array'] += 1                                # a 16x value that disagrees
    table.to_csv(tmp_path / "flow_attributes_4x_16x.csv", index=False)
    with pytest.raises(ValueError):
        array_study(tmp_path)


def test_scales():
    value = np.array([0.5, -0.2])
    drag = np.array([2.0, 4.0])
    assert list(scale_of('Fx_viscous', value, drag)) == [2.0, 4.0]       # the drag parts relative to the drag
    assert list(scale_of('Fy', value, drag)) == [2.0, 4.0]
    assert list(scale_of('Fx', value, drag)) == [0.5, 0.2]
    assert scale_of('base_pressure', value, drag)[0] == pytest.approx(0.005)
