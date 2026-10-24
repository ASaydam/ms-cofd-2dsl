import pytest

from validation import tolerance_check


def iteration(time, ux, uy, p):
    return (f"Time = {time}\n\n"
            f"smoothSolver:  Solving for Ux, Initial residual = {ux}, Final residual = 1e-9, No Iterations 2\n"
            f"smoothSolver:  Solving for Uy, Initial residual = {uy}, Final residual = 1e-9, No Iterations 2\n"
            f"GAMG:  Solving for p, Initial residual = {p}, Final residual = 1e-9, No Iterations 3\n"
            f"GAMG:  Solving for p, Initial residual = {p / 10}, Final residual = 1e-10, No Iterations 2\n"
            f"ExecutionTime = {time / 10} s  ClockTime = {time // 10} s\n\n")


def test_residual_history_every_100_iterations_and_the_last(tmp_path):
    log = tmp_path / 'log.simpleFoam'
    log.write_text("Starting time loop\n\n" + ''.join(iteration(t, 1e-6 / t, 2e-6 / t, 3e-6 / t)
                                                       for t in range(51400, 51651)))
    rows = tolerance_check.residual_history(log)
    assert [row['iteration'] for row in rows] == [51400, 51500, 51600, 51650]
    # the first p solve of the iteration (the one residualControl checks)
    assert float(rows[1]['p']) == pytest.approx(3e-6 / 51500)
    assert float(rows[-1]['Uy']) == pytest.approx(2e-6 / 51650)


def test_live_row_skips_superseded_archives(tmp_path):
    (tmp_path / 'archives').mkdir()
    (tmp_path / 'archives' / 'archive_index.csv').write_text(
        "run,case,file,bytes,md5,remote,uploaded\n"
        "16x,a_0001,superseded/a_0001.cap45000.tar.zst,1,x,r:16x/superseded/a_0001.cap45000.tar.zst,t\n"
        "16x,a_0001,a_0001.tar.zst,2,y,r:16x/a_0001.tar.zst,t\n"
        "4x,a_0001,a_0001.tar.zst,3,z,r:4x/a_0001.tar.zst,t\n")
    assert tolerance_check.live_row(tmp_path, 'a_0001')['md5'] == 'y'
    with pytest.raises(SystemExit):
        tolerance_check.live_row(tmp_path, 'b_0001')
