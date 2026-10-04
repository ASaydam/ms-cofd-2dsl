from extraction.check_runs import check_case, summed_runs


def solver_run(host, iterations, seconds_per_iteration, converged):
    """text of one simpleFoam run as it appears in log.simpleFoam"""
    lines = [f'Host   : "{host}"', "Starting time loop", ""]
    for i in iterations:
        lines += [f"Time = {i}", "", "smoothSolver:  Solving for Ux, Initial residual = 1e-06, Final residual = 1e-07, No Iterations 3",
                  "smoothSolver:  Solving for Uy, Initial residual = 2e-06, Final residual = 1e-07, No Iterations 3",
                  "GAMG:  Solving for p, Initial residual = 3e-07, Final residual = 1e-09, No Iterations 4",
                  f"ExecutionTime = {(i - iterations[0] + 1) * seconds_per_iteration} s  ClockTime = {(i - iterations[0] + 1) * seconds_per_iteration + 1} s", ""]
    lines += ["SIMPLE solution converged in 1 iterations" if converged else "", "End", ""]
    return "\n".join(lines)


def test_times_of_a_continued_run_are_added_up():
    assert summed_runs([1, 2, 3]) == 3
    assert summed_runs([1, 2, 3, 0.5, 1.5]) == 4.5


def test_continued_case_with_run_info(tmp_path):
    case = tmp_path / "larger_circle_1"
    case.mkdir()
    (case / "log.simpleFoam").write_text(solver_run("be05", range(1, 101), 2.0, False) +
                                         solver_run("da01", range(101, 151), 1.0, True))
    (case / "run_info.csv").write_text("mode,start,host,cpu_model,node_cpus,node_cpus_allocated,slurm_job_id\n"
                                       "run,2026-10-01T10:00:00+03:00,be05,Intel Xeon E5-2640,12,10,1\n"
                                       "continue,2026-10-01T12:00:00+03:00,da01,Intel Xeon E5-2670 v2,20,4,2\n")
    row = check_case(case)
    assert row['status'] == 'converged' and row['runs'] == 2
    assert row['last_iteration'] == '150'
    assert row['execution_time'] == 200 + 50
    assert row['seconds_per_iteration'] == round(250 / 150, 4)
    assert row['host'] == 'be05;da01' and row['cpu_model'] == 'Intel Xeon E5-2640' and row['node_cpus_allocated'] == '10'


def test_host_from_the_log_of_older_runs(tmp_path):
    case = tmp_path / "normal_circle_5"
    case.mkdir()
    (case / "log.simpleFoam").write_text(solver_run("be01", range(1, 11), 0.5, True))
    row = check_case(case)
    assert row['host'] == 'be01' and row['cpu_model'] == '' and row['execution_time'] == 5.0
