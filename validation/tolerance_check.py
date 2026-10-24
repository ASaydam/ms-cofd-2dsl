"""
Tolerance check (docs/validation.md): converged 0.25 mm runs of the dataset continued from the convergence criterion of
the production (every initial residual below 1e-6) to 1e-8, to measure how much the labels still change. Runs in a
validation data root (MSCOFD_DATA_ROOT, e.g. hpc_run/ms-cofd-2dsl/tol-check); Drive and the production root are only
read.

    python validation/tolerance_check.py run <case> [--extra 30000] [--stop-after <s>]      in a job, in the container
    python validation/tolerance_check.py compare <production root> <label_uncertainty.parquet>

run, for one case of the 16x run (again after a job stopped at its time limit: continues where it stopped):
1. the 16x archive of the case from Drive (the live row of archives/archive_index.csv of the production, MD5 checked)
   into restored/16x/, its case folder into work/16x/; the logs of the released run are kept as *.1e-6
2. only two changes: residualControl 1e-8 for p and U (system/fvSolution), endTime = released iterations + extra
   (system/controlDict); simpleFoam continues from the latest time step (startFrom latestTime)
3. stopped after --stop-after seconds (before the job time limit, exit code 3): the next job continues from the last
   written step
4. at the end the arrays and attributes as in the production (hpc/run_task.py Task.arrays and Task.attributes) to
   results/16x/, the initial residuals every 100 iterations to results/16x/<case>_residuals.csv and the record of the
   continuation to results/16x/<case>_tolerance.json

compare: per case, the change of the arrays (u_x, u_y, p: RMS of the change over the RMS of the released field, points
that are not wall or obstacle, as validation/compare_init.py) and of the 19 attributes against the release (production
results/16x): in % on the scale of the attribute (validation/audit_dataset.py level_difference) and as |change| / U,
U the label uncertainty of the case (source openfoam, ms-cofd-2dsl-bench references/label_uncertainty.parquet).
Writes results/tolerance_attributes.csv and results/tolerance_summary.csv in the data root.
"""

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.check_runs import RE_RESIDUAL, RE_TIME, check_case  # noqa: E402
from utils import REPO_ROOT, read_csv  # noqa: E402

RUN = '16x'
TOLERANCE = '1e-8'
RELEASED = '1e-6'           # suffix of the files of the released run kept in the case folder
KEPT = ['log.simpleFoam', 'log.time', 'log.attributes', 'provenance.json']
CHANNELS = {'u_x': 0, 'u_y': 1, 'p': 2}


def shell(command, cwd, log, append=False):
    with open(Path(cwd) / log, 'a' if append else 'w') as file:
        return subprocess.run(command, cwd=cwd, stdout=file, stderr=subprocess.STDOUT).returncode


def live_row(production, case):
    """the row of the released 16x archive of a case in the production archive index (not a superseded one)"""
    rows = [row for row in read_csv(Path(production) / 'archives' / 'archive_index.csv')
            if row['run'] == RUN and row['case'] == case and row['file'] == f"{case}.tar.zst"]
    if len(rows) != 1:
        sys.exit(f"{case}: {len(rows)} rows of the released archive in the archive index")
    return rows[0]


def restore(root, production, case):
    """step 1: the released archive from Drive, its case folder into work/16x"""
    folder = root / 'work' / RUN / case
    if (folder / f"log.simpleFoam.{RELEASED}").exists():
        return folder
    row = live_row(production, case)
    archive = root / 'restored' / RUN / f"{case}.tar.zst"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or md5_of(archive) != row['md5']:
        subprocess.run(['rclone', 'copyto', row['remote'], str(archive)], check=True)
    if md5_of(archive) != row['md5']:
        sys.exit(f"{case}: MD5 of the restored archive differs from the archive index")
    shutil.rmtree(folder, ignore_errors=True)
    folder.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['tar', '--zstd', '-xf', str(archive), '-C', str(folder.parent), case], check=True)
    for name in KEPT:
        if (folder / name).exists():
            os.replace(folder / name, folder / f"{name}.{RELEASED}")
    return folder


def md5_of(path):
    return subprocess.run(['md5sum', str(path)], capture_output=True, text=True, check=True).stdout.split()[0]


def released_iterations(folder):
    """iterations of the released run, from its provenance"""
    return int(float(json.loads((folder / f"provenance.json.{RELEASED}").read_text())['iterations']))


def set_up(folder, end_time):
    """step 2: residualControl 1e-8 and the new cap, the run continues from the latest time step"""
    for entry, value in (('SIMPLE/residualControl/p', TOLERANCE), ('SIMPLE/residualControl/U', TOLERANCE)):
        shell(['foamDictionary', 'system/fvSolution', '-entry', entry, '-set', value], folder, 'log.foamDictionary',
              append=True)
    for entry, value in (('endTime', str(end_time)), ('startFrom', 'latestTime')):
        shell(['foamDictionary', 'system/controlDict', '-entry', entry, '-set', value], folder, 'log.foamDictionary',
              append=True)


def residual_history(log, every=100):
    """initial residuals of Ux, Uy and p (the first solve of each iteration) every <every> iterations and the last"""
    text = log.read_text(errors='replace')
    starts = [(match.start(), match.group(1)) for match in RE_TIME.finditer(text)]
    rows = []
    for k, (start, value) in enumerate(starts):
        iteration = float(value)
        if int(iteration) % every and k != len(starts) - 1:
            continue
        block = text[start:starts[k + 1][0] if k + 1 < len(starts) else len(text)]
        found = {field: pattern.search(block) for field, pattern in RE_RESIDUAL.items()}
        rows.append({'iteration': int(iteration), **{field: m.group(1) if m else '' for field, m in found.items()}})
    return rows


def run(case, extra, stop_after):
    from hpc.run_task import Task, partial_time

    started = time.time()
    root = Path(os.environ['MSCOFD_DATA_ROOT'])
    production = Path(os.environ['MSCOFD_PRODUCTION_ROOT'])
    if root.resolve() == production.resolve():
        sys.exit("the data root is the production root")
    folder = restore(root, production, case)
    start_iteration = released_iterations(folder)
    end_time = start_iteration + extra
    jobs = root / 'results' / RUN / f"{case}_jobs.txt"
    jobs.parent.mkdir(parents=True, exist_ok=True)
    with open(jobs, 'a') as file:
        file.write(f"start {os.environ.get('SLURM_JOB_ID', '')} {platform.node().split('.')[0]} "
                   f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")

    status = check_case(folder) if (folder / 'log.simpleFoam').exists() else {'status': 'not_started'}
    if status['status'] not in ('converged', 'end_time'):
        continued = (folder / 'log.simpleFoam').exists()
        if continued:
            partial = partial_time(folder)
            while partial:
                shutil.rmtree(folder / partial)
                with open(folder / 'dropped_times', 'a') as file:
                    file.write(f"{partial} {os.environ.get('SLURM_JOB_ID', '')}\n")
                partial = partial_time(folder)
        else:
            set_up(folder, end_time)
        # timeout stops simpleFoam itself (exit code 124, 137 after the kill) before the job time limit
        limit = ['timeout', '--kill-after=60', str(int(stop_after))] if stop_after else []
        code = shell(['/usr/bin/time', '-v', '-a', '-o', 'log.time'] + limit + ['simpleFoam'], folder,
                     'log.simpleFoam', append=continued)
        status = check_case(folder)
        if status['status'] not in ('converged', 'end_time'):
            stopped = bool(stop_after) and code in (124, 137)
            with open(jobs, 'a') as file:
                file.write(f"{'stopped' if stopped else 'failed'} {os.environ.get('SLURM_JOB_ID', '')} at "
                           f"{status.get('last_iteration')} exit {code} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
            if not stopped:
                sys.exit(f"{case}: simpleFoam ended with exit code {code} at {status.get('last_iteration')}, see "
                         f"log.simpleFoam")
            print(f"{case}: stopped at {status.get('last_iteration')} before the time limit, continues in the next job")
            sys.exit(3)

    # step 4: arrays and attributes exactly as in the production
    task = Task(RUN, case)
    exported_time = task.arrays()
    task.attributes()
    rows = residual_history(folder / 'log.simpleFoam')
    with open(task.results / f"{case}_residuals.csv", 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['iteration', 'Ux', 'Uy', 'p'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    user = [float(line.split(':')[1]) for line in open(folder / 'log.time') if 'User time' in line]
    system = [float(line.split(':')[1]) for line in open(folder / 'log.time') if 'System time' in line]
    record = {
        'case': case, 'run': RUN, 'residual_control': TOLERANCE, 'status': status['status'],
        'released_iterations': start_iteration, 'cap': end_time, 'iterations': status.get('last_iteration'),
        'exported_time': exported_time, 'solver_runs': status.get('runs'),
        'residual_Ux': status.get('residual_Ux'), 'residual_Uy': status.get('residual_Uy'),
        'residual_p': status.get('residual_p'),
        'execution_time_s': status.get('execution_time'), 'clock_time_s': status.get('clock_time'),
        'cpu_time_s': round(sum(user) + sum(system), 1),
        'dropped_times': (folder / 'dropped_times').read_text().split('\n')[:-1]
        if (folder / 'dropped_times').exists() else [],
        'jobs': jobs.read_text().splitlines(), 'host': platform.node().split('.')[0],
        'cpu_model': next((line.split(':', 1)[1].strip() for line in open('/proc/cpuinfo')
                           if line.startswith('model name')), ''),
        'code_revision': (REPO_ROOT / 'REVISION').read_text().strip() if (REPO_ROOT / 'REVISION').exists() else '',
        'container_digest': os.environ.get('MSCOFD_IMAGE_DIGEST', ''),
        'post_processing_s': round(time.time() - started, 1),
        'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    (task.results / f"{case}_tolerance.json").write_text(json.dumps(record, indent=1, default=str) + '\n')
    print(json.dumps(record, indent=1, default=str))


def compare(production, uncertainty_file):
    import pyarrow.parquet as pq

    from dataset.results import load_attributes, load_sample
    from validation.audit_dataset import inlet_velocity, level_difference

    root = Path(os.environ['MSCOFD_DATA_ROOT'])
    table = pq.read_table(uncertainty_file, columns=['case_name', 'attribute', 'source', '16x', 'U', 'U_%']).to_pydict()
    uncertainty = {(c, a): (v, u, up) for c, a, s, v, u, up in zip(*table.values()) if s == 'openfoam'}
    names = sorted({a for _, a in uncertainty})
    manifest = {row['case_name']: row for row in read_csv(production / 'geometry' / 'manifest.csv')}
    inlet = inlet_velocity()
    detail, summary = [], []
    for path in sorted((root / 'results' / RUN).glob('*_tolerance.json')):
        record = json.loads(path.read_text())
        case = record['case']
        released, tight = load_sample(production, RUN, case), load_sample(root, RUN, case)
        points = released[4] != 1
        field = {}
        for name, channel in CHANNELS.items():
            a, b = tight[channel][points], released[channel][points]
            field[name] = 100 * np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2))
        old, new = load_attributes(production, RUN, case), load_attributes(root, RUN, case)
        ratios, mismatch = {}, []
        for name in names:
            value, reference = float(new[name]), float(old[name])
            u_value, u, u_pct = uncertainty[(case, name)]
            if not np.isclose(u_value, reference, rtol=1e-12, atol=0):
                mismatch.append(name)
            change = level_difference(name, value, reference, float(old['Fx']), inlet,
                                      float(manifest[case]['bbox_height']))
            ratio = abs(value - reference) / u if u > 0 else float('inf') if value != reference else 0.0
            ratios[name] = ratio
            detail.append({'case_name': case, 'attribute': name, 'released': reference, 'tight': value,
                           'change_%': None if change is None else 100 * change, 'U': u, 'U_%': u_pct,
                           'change_over_U': ratio})
        top = max(ratios, key=ratios.get)
        largest_field = max(field, key=field.get)
        summary.append({
            'case_name': case, 'status': record['status'], 'released_iterations': record['released_iterations'],
            'iterations': record['iterations'], 'extra_iterations': int(float(record['iterations'])) -
            record['released_iterations'], 'cpu_time_h': round(record['cpu_time_s'] / 3600, 2),
            'residual_Ux': record['residual_Ux'], 'residual_Uy': record['residual_Uy'], 'residual_p': record['residual_p'],
            **{f"{name}_%": round(value, 6) for name, value in field.items()},
            'largest_field': largest_field, 'largest_field_%': round(field[largest_field], 6),
            'largest_change_over_U': round(ratios[top], 6), 'attribute': top,
            'change_%_of_it': next((round(row['change_%'], 6) for row in detail if row['case_name'] == case
                                    and row['attribute'] == top and row['change_%'] is not None), None),
            'released_attributes_match_uncertainty_table': not mismatch,
        })
    for name, rows in (('tolerance_attributes.csv', detail), ('tolerance_summary.csv', summary)):
        with open(root / 'results' / name, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    for row in summary:
        print('; '.join(f"{key} {value}" for key, value in row.items()))
    by_attribute = {}
    for row in detail:
        by_attribute.setdefault(row['attribute'], []).append(row)
    print("\nper attribute over the cases: largest |change| % (on its scale), largest and median |change| / U")
    for name, rows in sorted(by_attribute.items()):
        changes = [abs(r['change_%']) for r in rows if r['change_%'] is not None]
        over = [r['change_over_U'] for r in rows]
        print(f"  {name:24s} {max(changes) if changes else float('nan'):10.5f} %  {max(over):9.5f}  "
              f"{float(np.median(over)):9.5f}")


def main():
    parser = argparse.ArgumentParser(description="Converged 0.25 mm runs continued to residualControl 1e-8")
    commands = parser.add_subparsers(dest='command', required=True)
    one = commands.add_parser('run')
    one.add_argument('case')
    one.add_argument('--extra', type=int, default=30000, help="iterations beyond the released run (cap)")
    one.add_argument('--stop-after', type=float, help="seconds of simpleFoam in this job (before the time limit)")
    two = commands.add_parser('compare')
    two.add_argument('production', type=Path)
    two.add_argument('uncertainty')
    args = parser.parse_args()
    if args.command == 'run':
        run(args.case, args.extra, args.stop_after)
    else:
        compare(args.production, args.uncertainty)


if __name__ == "__main__":
    main()
