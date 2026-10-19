import csv
import sys

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


def write_csv(path, table):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list(table))
        writer.writerows(zip(*table.values()))


def test_array_study_reads_every_level(tmp_path):
    pytest.importorskip("pyarrow")
    cases = ['normal_circle_0001', 'normal_circle_0002']
    fine = {'normal_circle_0001': 1.0, 'normal_circle_0002': 2.0}
    tables = {}
    for k, run in enumerate(RUNS.values()):
        table = tables[run] = {'case_name': cases[::-1] if k else cases}   # row order differs between the files
        for name in ARRAY_ATTRIBUTES:
            table[f'{name}_16x_array'] = [fine[c] for c in table['case_name']]
            table[f'{name}_{run}_array'] = [fine[c] + 0.1 * (k + 1) for c in table['case_name']]
        write_csv(tmp_path / f"flow_attributes_{run}_16x.csv", table)

    study = array_study(tmp_path)
    assert len(study['case_name']) == 2 * len(ARRAY_ATTRIBUTES)
    i = next(i for i, (a, c) in enumerate(zip(study['attribute'], study['case_name']))
             if (a, c) == ('dissipation', 'normal_circle_0002'))
    assert (study['16x'][i], study['4x'][i], study['base'][i], study['0.25x'][i]) == pytest.approx((2.0, 2.1, 2.2, 2.3))

    table = tables['4x']
    table['dissipation_16x_array'] = [v + 1 for v in table['dissipation_16x_array']]   # a 16x value that disagrees
    write_csv(tmp_path / "flow_attributes_4x_16x.csv", table)
    with pytest.raises(ValueError):
        array_study(tmp_path)


def test_main_writes_the_table(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from tools import uncertainty as tool

    cases = ['normal_circle_0001', 'normal_circle_0002']
    phi = {'Fx': 1 + 0.1 * H ** 1.5, 'Fy': 0.05 + 0.01 * H ** 1.5}          # second-order-like data, fine to coarse
    rows = [{'case_name': c, 'attribute': a, **dict(zip(tool.LEVELS, values))} for c in cases for a, values in phi.items()]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / 'mesh_study.parquet')
    cells = [16000, 4000, 1000, 250]                                          # h = 1, 2, 4, 8
    pq.write_table(pa.Table.from_pylist([{'case_name': c, **dict(zip(tool.CELLS, cells))} for c in cases]),
                   tmp_path / 'metadata.parquet')
    monkeypatch.setattr(sys, 'argv', ['uncertainty', str(tmp_path), str(tmp_path / 'out')])
    tool.main()

    table = pq.read_table(tmp_path / 'out' / 'label_uncertainty.parquet').to_pydict()
    assert list(table) == ['case_name', 'attribute', 'source', '16x', '4x', 'base', '0.25x', 'U', 'p', 'method',
                           'phi_0', 'U_%', 'U1_%']
    assert set(table['source']) == {'openfoam'} and set(table['method']) == {'power'}
    fx = table['attribute'].index('Fx')
    assert table['U_%'][fx] == pytest.approx(100 * 1.25 * 0.1 / 1.1, rel=1e-3)    # Fs e, relative to the label
    fy = table['attribute'].index('Fy')
    assert table['U_%'][fy] == pytest.approx(100 * 1.25 * 0.01 / 1.1, rel=1e-3)   # lift relative to the drag
    with open(tmp_path / 'out' / 'label_uncertainty.csv') as file:
        assert next(csv.reader(file)) == list(table)


def test_scales():
    value = np.array([0.5, -0.2])
    drag = np.array([2.0, 4.0])
    assert list(scale_of('Fx_viscous', value, drag)) == [2.0, 4.0]       # the drag parts relative to the drag
    assert list(scale_of('Fy', value, drag)) == [2.0, 4.0]
    assert list(scale_of('Fx', value, drag)) == [0.5, 0.2]
    assert scale_of('base_pressure', value, drag)[0] == pytest.approx(0.005)
