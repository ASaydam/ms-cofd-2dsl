"""
Checks the solver logs of all OpenFOAM cases of one resolution and writes a table of the runs.

    python -m extraction.check_runs outputs/cases/base

For every case (log.simpleFoam, log.checkMesh):

    status          converged   : stopped by residualControl (SIMPLE solution converged)
                    end_time    : reached endTime without converging, can be continued with resubmit.sh
                                  (with the converged settings endTime is a cap: residuals that stall there
                                  can mean that the flow has no steady solution, e.g. vortex shedding)
                    unfinished  : log has no End line (still running, or the solver crashed / was killed)
                    not_started : no log.simpleFoam
    last_iteration  last "Time = " value of the log
    execution_time, clock_time  solver time of the case in s ("ExecutionTime = ... s  ClockTime = ... s"); a
                    continued case has several runs in its log, whose clocks start at 0 again: their times are added
    seconds_per_iteration       execution_time / last_iteration
    runs            number of solver runs in the log (1, more for continued cases)
    host, cpu_model, node_cpus_allocated   where the first run ran and how many CPUs of its node were allocated
                    when it started (run_info.csv of run_case.sh); hosts of all runs separated by ';'. Cases run
                    before run_info.csv existed only have the host, from the header of the solver log
    mesh_ok         checkMesh printed "Mesh OK."
    residual_*      last initial residuals of Ux, Uy and p
    init_from       case whose solution the run started from (mapFields), empty for the uniform 0 folder

Writes run_status.csv to the cases folder, plus not_converged.txt and not_finished.txt (one case name per line),
which can be given to resubmit.sh.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import read_csv, write_csv

RE_TIME = re.compile(r"^Time = ([\d.eE+-]+)\s*$", re.MULTILINE)
RE_EXECUTION = re.compile(r"ExecutionTime = ([\d.eE+-]+) s\s+ClockTime = ([\d.eE+-]+) s")
RE_HOST = re.compile(r'^Host\s*:\s*"?([^"\s]+)"?', re.MULTILINE)
RE_RESIDUAL = {field: re.compile(rf"Solving for {field}, Initial residual = ([\d.eE+-]+)") for field in ('Ux', 'Uy', 'p')}

FIELDS = ['case_name', 'status', 'last_iteration', 'execution_time', 'clock_time', 'seconds_per_iteration', 'runs',
          'host', 'cpu_model', 'node_cpus_allocated', 'mesh_ok', 'residual_Ux', 'residual_Uy', 'residual_p', 'init_from']


def summed_runs(values):
    """total of a time that every solver run of a log counts from 0 again (the last value of each run, added up)"""
    total, previous = 0.0, 0.0
    for value in values:
        if value < previous:
            total += previous
        previous = value
    return total + previous


def run_info(case_folder, text):
    """host, CPU model and allocated CPUs of the node of the first run, hosts of all runs (see run_case.sh)"""
    info_file = case_folder / "run_info.csv"
    if info_file.exists():
        runs = read_csv(info_file)
        if runs:
            return {'host': ';'.join(run['host'] for run in runs), 'cpu_model': runs[0]['cpu_model'],
                    'node_cpus_allocated': runs[0]['node_cpus_allocated']}
    # older runs: only the host, from the header of the solver log
    return {'host': ';'.join(host.split('.')[0] for host in RE_HOST.findall(text)), 'cpu_model': '',
            'node_cpus_allocated': ''}


def check_case(case_folder):
    """
    reads the logs of a single case
    :return: row of the run status table
    """
    row = {'case_name': case_folder.name}

    check_mesh_log = case_folder / "log.checkMesh"
    row['mesh_ok'] = check_mesh_log.exists() and "Mesh OK." in check_mesh_log.read_text(errors='replace')

    init_from = case_folder / "init_from"
    row['init_from'] = init_from.read_text().strip() if init_from.exists() else ''

    solver_log = case_folder / "log.simpleFoam"
    if not solver_log.exists():
        row['status'] = 'not_started'
        return row

    text = solver_log.read_text(errors='replace')

    times = RE_TIME.findall(text)
    row['last_iteration'] = times[-1] if times else ''

    execution = RE_EXECUTION.findall(text)
    if execution:
        row['execution_time'] = round(summed_runs([float(e) for e, _ in execution]), 2)
        row['clock_time'] = round(summed_runs([float(c) for _, c in execution]), 2)
        if row['last_iteration'] and float(row['last_iteration']) > 0:
            row['seconds_per_iteration'] = round(row['execution_time'] / float(row['last_iteration']), 4)
    row['runs'] = text.count("Starting time loop")
    row.update(run_info(case_folder, text))

    # initial residuals of the last iteration (its first p solve is the one residualControl checks)
    last_iteration = [match.start() for match in RE_TIME.finditer(text)]
    last_block = text[last_iteration[-1]:] if last_iteration else text
    for field, pattern in RE_RESIDUAL.items():
        values = pattern.findall(last_block)
        row[f'residual_{field}'] = values[0] if values else ''

    # a continued case has more than one run in its log, only the last run counts
    last_start = text.rfind("Starting time loop")
    last_run = text[last_start:] if last_start >= 0 else text
    if "SIMPLE solution converged" in last_run:
        row['status'] = 'converged'
    elif re.search(r"^End\s*$", last_run, re.MULTILINE):
        row['status'] = 'end_time'
    else:
        row['status'] = 'unfinished'

    return row


def check_runs(cases_directory):
    cases_directory = Path(cases_directory)
    rows = [check_case(folder) for folder in sorted(cases_directory.iterdir()) if folder.is_dir()]

    write_csv(rows, cases_directory / "run_status.csv", FIELDS)

    not_converged = [row['case_name'] for row in rows if row['status'] == 'end_time']
    not_finished = [row['case_name'] for row in rows if row['status'] in ('unfinished', 'not_started')]
    (cases_directory / "not_converged.txt").write_text("".join(f"{name}\n" for name in not_converged))
    (cases_directory / "not_finished.txt").write_text("".join(f"{name}\n" for name in not_finished))

    for status in ('converged', 'end_time', 'unfinished', 'not_started'):
        print(f"{status:>12}: {sum(row['status'] == status for row in rows)}")
    print(f"mesh not OK : {sum(not row['mesh_ok'] for row in rows)}")
    print(f"Run status written to {cases_directory / 'run_status.csv'}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the OpenFOAM runs of one resolution")
    parser.add_argument("cases_dir", help="folder of the OpenFOAM cases, e.g. outputs/cases/base")
    check_runs(parser.parse_args().cases_dir)
