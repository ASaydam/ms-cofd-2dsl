"""
Final audit of the dataset before it is frozen (docs/validation.md; the Technical Validation of the dataset paper).
Reads the results of a data root and checks every run of every case of the dataset; nothing is changed.

    python -m validation.audit_dataset <report folder>                    data root: MSCOFD_DATA_ROOT
    python -m validation.audit_dataset <report folder> --remote           also the archives on the remote (rclone)
    python -m validation.audit_dataset <report folder> --skip arrays      leave out checks (names below)
    python -m validation.audit_dataset <report folder> --cases list.txt   only these cases (except members)

Checks:

    members      cases_per_geometry cases of every size and family (reserve cases for the dropped ones), the dropped
                 cases are those whose base_lu stopped at its cap
    states       every run of every case done and converged
    provenance   status converged, final initial residuals of U and p <= 1e-6, mesh OK, templates, init_from and cap
                 as set by config/dataset.yaml, case_settings.txt and iterations.txt; the iterations agree with the
                 state marker and the latest row of summary.csv
    archives     one live row of archive_index.csv per run (the newest is kept when there are more); with --remote
                 the size and MD5 of every live archive on the remote, and remote files without a row
    arrays       shape and dtype of every level, finite values, 0 inside the obstacle, geometry channels consistent;
                 the obstacle of the arrays against the manifest (area, centroid: a mixed-up case shows here); the
                 volume flux through the cross-sections away from the inlet and the obstacle
    attributes   every attribute present and finite, signs, cells and time as in the provenance; the flags of the
                 data card (reverse flow at the outlet, recirculation length below 2 mm)
    levels       the attributes of every level against the 0.25 mm level: robust outliers per run, size and family,
                 and the median differences
    mesh_study   tools/mesh_study.py on every case: type of convergence, observed order and GCI of every attribute
    splits       the fixed splits (dataset/splits.py): counts, per size and family; written to splits.csv
    geometry     the .geo files of every case written again from the seed: identical to those of the data root
    leftovers    seeds, work folders and local archives left, hold list, superseded results

Output in <report folder>: findings.csv (level, check, run, case, message), summary.json (the numbers for the paper
and the data card), report.md, splits.csv. Levels: error (to be fixed before the freeze), warning (to be looked at),
info. Exit code 1 when there is an error.

Needs numpy and pyyaml (the container); --remote needs rclone.
"""

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import REPO_ROOT, load_config  # noqa: E402
from dataset.build_dataset import case_files, mesh_study_rows  # noqa: E402
from dataset.splits import assign_splits  # noqa: E402
from extraction.build_arrays import FLUID, INLET, OBSTACLE, OUTLET, WALL, wall_distance  # noqa: E402
from hpc import ledger, topup  # noqa: E402
from tools import mesh_study  # noqa: E402
from tools.flow_attributes import spacing  # noqa: E402

CHECKS = ['members', 'states', 'provenance', 'archives', 'arrays', 'attributes', 'levels', 'mesh_study', 'splits',
          'geometry', 'leftovers']

TOLERANCE = 1e-6            # residualControl of every run of the dataset (case templates)
FLUX_TOLERANCE = 0.01       # largest relative deviation of the volume flux of a cross-section from the inlet flux
FLUX_START_MM = 20          # cross-sections closer to the inlet are left out (developing flow, corner points)
OUTLIER_MADS = 10           # levels: a difference this many (scaled) median absolute deviations from the median
OUTLIER_FLOOR = 0.05        # and at least this far (relative) is an outlier
SHORT_RECIRCULATION_MM = 2  # data card flag: a recirculation shorter than this is hardly resolved on the 2 mm grid
LISTED = 20                 # findings listed per check in report.md (all of them are in findings.csv)

# attributes of every run (tools/openfoam_attributes.py); 'time' is the iteration of the exported time step
REQUIRED = mesh_study.ATTRIBUTES + ['outlet_reverse_flow', 'cells', 'faces', 'time']
# attributes that are positive for every steady flow past an obstacle in the channel (drag, losses)
POSITIVE = ['Fx', 'Fx_pressure', 'mean_wall_shear', 'max_wall_shear', 'dissipation', 'dp_inlet_outlet', 'dp_sections']
# attributes compared between the levels: the peak values grow with the refinement at sharp corners (singular, see
# README), they are no mesh-independent targets
LEVEL_ATTRIBUTES = [name for name in mesh_study.ATTRIBUTES if name not in ('max_wall_shear', 'peak_vorticity')]


class Audit:
    """the findings and the summary numbers of the audit"""

    def __init__(self):
        self.findings, self.summary = [], {}

    def add(self, level, check, message, run='', case=''):
        self.findings.append({'level': level, 'check': check, 'run': run, 'case': case, 'message': message})

    def count(self, check=None, level=None):
        return sum(1 for f in self.findings if check in (None, f['check']) and level in (None, f['level']))


def as_float(value):
    """a provenance or attribute value as float, None when it is not a number"""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def statistics(values):
    """median, 95th percentile and largest of a list of numbers (None for an empty list)"""
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {'n': len(values), 'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)),
            'max': float(np.max(values))}


def inlet_velocity(template=REPO_ROOT / 'openfoam' / 'case-templates' / 'second-order' / 'base' / '0' / 'U'):
    """x velocity of the inlet of the case templates (fixedValue, m/s)"""
    match = re.search(r'inlet\s*\{[^}]*?value\s+uniform\s+\(\s*([-\d.eE+]+)', Path(template).read_text())
    return float(match.group(1))


# ------------------------------------------------------------------------------------------------------- checks


def check_members(audit, root, dataset, states):
    """
    the cases of the dataset (hpc/topup.py dataset_cases, as in dataset/build_dataset.py)
    :return: member cases, dropped cases, reserve cases in use, manifest rows by case name
    """
    cases, dropped, reserves = topup.dataset_cases(root, dataset, states, topup.cases_file_of(root))
    members = [case for case in cases if case not in set(dropped)]
    manifest = {row['case_name']: row for row in ledger.read_manifest(root / 'geometry' / 'manifest.csv')}

    per_class = Counter((manifest[case]['size'], manifest[case]['geometry']) for case in members)
    for size, number in dataset['cases_per_geometry'].items():
        for geometry in dataset['geometries']:
            if per_class[(size, geometry)] != number:
                audit.add('error', 'members', f"{size} {geometry}: {per_class[(size, geometry)]} cases, "
                                              f"expected {number}")
    expected = sum(dataset['cases_per_geometry'].values()) * len(dataset['geometries'])
    if len(members) != expected:
        audit.add('error', 'members', f"{len(members)} cases, expected {expected}")
    for case in dropped:
        audit.add('info', 'members', "dropped: no steady solution (base_lu at its cap)", 'base_lu', case)
    audit.summary['members'] = {'cases': len(members), 'dropped': len(dropped), 'reserves_in_use': len(reserves),
                                'dropped_cases': dropped, 'reserve_cases': reserves,
                                'dropped_per_class': dict(Counter(f"{manifest[c]['size']} {manifest[c]['geometry']}"
                                                                  for c in dropped))}
    return members, dropped, reserves, manifest


def check_states(audit, runs, members, states):
    """every run of every case done and converged (marker "converged <iterations> <job id>")"""
    for run in runs:
        for case in members:
            state = states.get((run, case))
            if not ledger.converged(state):
                audit.add('error', 'states', f"state {' '.join(state) if state else 'none'}", run, case)


def latest_summary_rows(root, run):
    """the last row of every case in results/<run>/summary.csv (one row per finished job), and the number of rows"""
    path = Path(root) / 'results' / run / 'summary.csv'
    latest, count = {}, Counter()
    if path.exists():
        with open(path, newline='') as file:
            for row in csv.DictReader(file):
                latest[row['case']] = row
                count[row['case']] += 1
    return latest, count


def check_provenance(audit, root, dataset, runs, members, states):
    """
    the provenance of every run: converged to TOLERANCE with the settings the run should have had
    :return: {(run, case): provenance record}
    """
    caps = {(run, case): int(values[0]) for run, case, values in ledger.task_lines(Path(root) / 'iterations.txt')
            if values}
    records, revisions, digests = {}, Counter(), Counter()
    audit.summary['provenance'] = {}
    for run in runs:
        latest, rows_per_case = latest_summary_rows(root, run)
        for case in members:
            path = case_files(root, run, case)['provenance']
            if not path.exists():
                audit.add('error', 'provenance', "no provenance", run, case)
                continue
            record = json.loads(path.read_text())
            records[(run, case)] = record
            settings = {**dataset['runs'][run], **ledger.case_settings(run, case, root)}

            problems = []
            if (record.get('run'), record.get('case')) != (run, case):
                problems.append(f"provenance of {record.get('run')} {record.get('case')}")
            if record.get('status') != 'converged':
                problems.append(f"status {record.get('status')}")
            residuals = {name: as_float(record.get(f"residual_{name}")) for name in ('Ux', 'Uy', 'p')}
            above = {name: value for name, value in residuals.items() if value is None or value > TOLERANCE}
            if above:
                problems.append("residuals above the criterion: " + ', '.join(f"{n} {v}" for n, v in above.items()))
            if record.get('mesh_ok') is not True:
                problems.append(f"mesh_ok {record.get('mesh_ok')}")
            if record.get('templates') != settings['templates']:
                problems.append(f"templates {record.get('templates')}, expected {settings['templates']}")
            if (record.get('init_from') or '') != (settings.get('init_from') or ''):
                problems.append(f"init_from '{record.get('init_from')}', expected '{settings.get('init_from') or ''}'")
            if record.get('end_time_override') != caps.get((run, case)):
                problems.append(f"cap {record.get('end_time_override')}, iterations.txt {caps.get((run, case))}")
            state = states.get((run, case))
            if state and ledger.converged(state) and state[1].split()[1:2] != [str(record.get('iterations'))]:
                problems.append(f"iterations {record.get('iterations')}, state marker '{state[1]}'")
            for problem in problems:
                audit.add('error', 'provenance', problem, run, case)

            row = latest.get(case)
            if row is None or row['status'] != 'converged' or row['iterations'] != str(record.get('iterations')):
                audit.add('warning', 'provenance', "latest row of summary.csv: " +
                          (f"{row['status']} {row['iterations']}" if row else "none") +
                          f", provenance {record.get('status')} {record.get('iterations')}", run, case)
            revisions[record.get('code_revision') or ''] += 1
            digests[record.get('container_digest') or ''] += 1

        found = [records[(run, case)] for case in members if (run, case) in records]
        audit.summary['provenance'][run] = {
            'runs': len(found),
            'iterations': statistics([as_float(r.get('iterations')) for r in found]),
            'solver_time_h': statistics([(as_float(r.get('execution_time_s')) or 0) / 3600 for r in found]),
            'solver_core_hours': round(sum(as_float(r.get('execution_time_s')) or 0 for r in found) / 3600, 1),
            'peak_memory_mb': statistics([(r.get('peak_memory_kb') or 0) / 1024 for r in found]),
            'continued': sum(1 for r in found if (r.get('solver_runs') or 1) > 1),
            'raised_cap': sum(1 for r in found if r.get('end_time_override')),
            'dropped_time_steps': sum(1 for r in found if r.get('dropped_times')),
            'case_settings': sum(1 for r in found if r.get('case_settings')),
            'templates': dict(Counter(r.get('templates') for r in found)),
            'mesh_variants': dict(Counter((r.get('mesh_variant') or '').split(' ')[0] for r in found)),
            'cpu_models': dict(Counter(r.get('cpu_model') for r in found)),
            'summary_rows_more_than_one': sum(1 for case in members if rows_per_case[case] > 1),
        }
    audit.summary['code_revisions'] = dict(revisions)
    audit.summary['container_digests'] = dict(digests)
    if len(digests) > 1:
        audit.add('warning', 'provenance', f"{len(digests)} container digests: {dict(digests)}")
    return records


def remote_listing(folder):
    """{file name: (size, md5)} of the files in a remote folder (rclone lsjson, the hashes come from the remote)"""
    result = subprocess.run(['rclone', 'lsjson', '--hash', '--files-only', folder], capture_output=True, text=True,
                            check=True)
    return {entry['Path']: (entry['Size'], (entry.get('Hashes') or {}).get('md5'))
            for entry in json.loads(result.stdout)}


def check_archives(audit, root, runs, members, remote=False):
    """
    one archive per run in archives/archive_index.csv; rows with /superseded/ in the remote path belong to runs that
    were continued (hpc/continue_runs.py). A run with more than one live row (solved or continued again before
    hpc/continue_runs.py existed) keeps its newest row; with remote, every live archive is compared with the remote
    """
    path = Path(root) / 'archives' / 'archive_index.csv'
    rows = list(csv.DictReader(open(path, newline=''))) if path.exists() else []
    live = {}
    for row in rows:
        if '/superseded/' not in row['remote']:
            live.setdefault((row['run'], row['case']), []).append(row)

    newest = {}
    for run in runs:
        for case in members:
            found = sorted(live.get((run, case), []), key=lambda r: r['uploaded'])
            if not found:
                audit.add('error', 'archives', "no live row in archive_index.csv", run, case)
                continue
            if len(found) > 1:
                audit.add('warning', 'archives', f"{len(found)} live rows ({len({r['md5'] for r in found})} MD5, "
                                                 f"{len({r['remote'] for r in found})} remote files), the newest of "
                                                 f"{found[-1]['uploaded']} is kept", run, case)
            newest[(run, case)] = found[-1]
    others = sorted({key for key in live if key not in newest and key[0] in runs})
    audit.summary['archives'] = {'archives': len(newest), 'bytes': sum(int(r['bytes']) for r in newest.values()),
                                 'duplicate_live_rows': sum(1 for key in newest if len(live[key]) > 1),
                                 'superseded_rows': sum(1 for r in rows if '/superseded/' in r['remote']),
                                 'rows_of_other_cases': len(others), 'remote_checked': bool(remote)}

    if remote:
        files = {}
        for folder in sorted({row['remote'].rsplit('/', 1)[0] for row in newest.values()}):
            files.update({f"{folder}/{name}": entry for name, entry in remote_listing(folder).items()})
        for (run, case), row in newest.items():
            entry = files.get(row['remote'])
            if entry is None:
                audit.add('error', 'archives', f"not on the remote: {row['remote']}", run, case)
            elif entry[1] != row['md5']:
                audit.add('error', 'archives', f"remote MD5 {entry[1]}, index {row['md5']}", run, case)
            elif int(entry[0]) != int(row['bytes']):
                audit.add('error', 'archives', f"remote size {entry[0]}, index {row['bytes']}", run, case)
        referenced = {row['remote'] for row in rows}
        for name in sorted(set(files) - referenced):
            audit.add('warning', 'archives', f"remote file without a row in archive_index.csv: {name}")
        audit.summary['archives']['remote_files'] = len(files)


def sample_problems(sample, shape, walls):
    """
    errors of one sample (extraction/build_arrays.py): shape, dtype, finite values, the geometry channels, 0 in the
    obstacle
    :return: list of messages, the obstacle points (None when the shape is wrong)
    """
    if sample.shape != shape:
        return [f"shape {sample.shape}, expected {shape}"], None
    problems = []
    if sample.dtype != np.float32:
        problems.append(f"dtype {sample.dtype}")
    if not np.isfinite(sample).all():
        problems.append(f"{int((~np.isfinite(sample)).sum())} values not finite")
    cell_type = sample[4]
    if not ((cell_type[0] == INLET).all() and (cell_type[-1] == OUTLET).all() and
            (cell_type[1:-1, 0] == WALL).all() and (cell_type[1:-1, -1] == WALL).all()):
        problems.append("cell_type: inlet, outlet or wall points wrong")
    if not np.isin(cell_type[1:-1, 1:-1], [OBSTACLE, FLUID]).all():
        problems.append("cell_type: values other than obstacle and fluid inside the domain")
    obstacle = np.zeros(cell_type.shape, bool)
    obstacle[1:-1, 1:-1] = cell_type[1:-1, 1:-1] == OBSTACLE
    if not obstacle.any():
        return problems + ["no obstacle points"], None
    if np.abs(sample[:3][:, obstacle]).max() != 0:
        problems.append("flow channels not 0 inside the obstacle")
    if not np.array_equal(sample[3], walls):
        problems.append("wall_distance differs from extraction/build_arrays.py wall_distance")
    if not np.array_equal(sample[5] <= 0, obstacle):
        problems.append("obstacle_sdf: sign does not match the obstacle points")
    return problems, obstacle


def obstacle_and_flux(sample, obstacle, row, domain, inlet):
    """
    the obstacle of the arrays against the manifest, and the volume flux of the cross-sections
    :return: area difference (mm^2), its tolerance, centroid distance (mm), its tolerance, largest relative deviation
             of the flux from inlet * height
    """
    nx, ny = obstacle.shape
    hx, hy = spacing(nx, ny, domain)
    i, j = np.nonzero(obstacle)
    area = len(i) * hx * hy
    # the obstacle is convex: its perimeter is at most that of its bounding box; every grid point on the border
    # counts at most half a cell too much or too little
    area_tolerance = (float(row['bbox_width']) + float(row['bbox_height'])) * max(hx, hy)
    centroid = math.hypot(i.mean() * hx - float(row['centroid_x']), j.mean() * hy - float(row['centroid_y']))

    columns = np.arange(nx)
    near = (columns >= i.min() - 2) & (columns <= i.max() + 2)
    used = (columns * hx >= FLUX_START_MM) & ~near
    ux = sample[0][used].astype(float)
    flux = hy * (ux[:, 1:-1].sum(axis=1) + 0.5 * (ux[:, 0] + ux[:, -1]))      # trapezoidal rule, m/s mm
    deviation = float(np.abs(flux / (inlet * domain['height']) - 1).max()) if used.any() else 0.0
    return abs(area - float(row['area'])), area_tolerance, centroid, max(hx, hy), deviation


def check_arrays(audit, root, dataset, runs, members, manifest, records):
    """every sample of every run, see sample_problems and obstacle_and_flux"""
    domain, inlet = dataset['domain'], inlet_velocity()
    audit.summary['arrays'] = {'inlet_velocity': inlet}
    for run in runs:
        nx, ny = dataset['extraction'][dataset['runs'][run]['resolution']]
        walls = wall_distance(nx, ny)
        areas, centroids, fluxes = [], [], []
        for case in members:
            path = case_files(root, run, case)['npz']
            if not path.exists():
                audit.add('error', 'arrays', "no array", run, case)
                continue
            with np.load(path) as data:
                sample, exported = data['sample'], float(data['time'])
            problems, obstacle = sample_problems(sample, (6, nx, ny), walls)
            record = records.get((run, case), {})
            if as_float(record.get('exported_time')) not in (None, exported):
                problems.append(f"time {exported}, provenance {record.get('exported_time')}")
            for problem in problems:
                audit.add('error', 'arrays', problem, run, case)
            if obstacle is None:
                continue
            area, area_tolerance, centroid, centroid_tolerance, flux = obstacle_and_flux(
                sample, obstacle, manifest[case], domain, inlet)
            areas.append(100 * area / float(manifest[case]['area']))
            centroids.append(centroid)
            fluxes.append(100 * flux)
            if area > area_tolerance:
                audit.add('warning', 'arrays', f"obstacle area differs from the manifest by {area:.1f} mm^2 "
                                               f"(tolerance {area_tolerance:.1f})", run, case)
            if centroid > centroid_tolerance:
                audit.add('warning', 'arrays', f"obstacle centroid {centroid:.2f} mm from the manifest's "
                                               f"(tolerance {centroid_tolerance:.2f})", run, case)
            if flux > FLUX_TOLERANCE:
                audit.add('warning', 'arrays', f"volume flux of a cross-section off by {100 * flux:.2f} %", run, case)
        audit.summary['arrays'][run] = {'grid': [nx, ny], 'area_difference_%': statistics(areas),
                                        'centroid_distance_mm': statistics(centroids),
                                        'flux_deviation_%': statistics(fluxes)}


def check_attributes(audit, root, runs, members, records):
    """
    the OpenFOAM attributes of every run
    :return: {(run, case): attributes}
    """
    values, flags = {}, {run: {'outlet_reverse_flow': [], 'short_recirculation': []} for run in runs}
    for run in runs:
        for case in members:
            path = case_files(root, run, case)['attributes']
            if not path.exists():
                audit.add('error', 'attributes', "no attributes", run, case)
                continue
            attributes = json.loads(path.read_text())
            values[(run, case)] = attributes
            missing = [name for name in REQUIRED if name not in attributes]
            invalid = [name for name in REQUIRED if name in attributes and as_float(attributes[name]) is None]
            if missing:
                audit.add('error', 'attributes', f"missing: {', '.join(missing)}", run, case)
            if invalid:
                audit.add('error', 'attributes', f"not a finite number: {', '.join(invalid)}", run, case)
            for name in POSITIVE:
                value = as_float(attributes.get(name))
                if value is not None and value <= 0:
                    audit.add('error' if name == 'Fx' else 'warning', 'attributes', f"{name} = {value} <= 0", run,
                              case)
            record = records.get((run, case), {})
            if as_float(attributes.get('cells')) != as_float(record.get('cells')):
                audit.add('error', 'attributes', f"cells {attributes.get('cells')}, provenance {record.get('cells')}",
                          run, case)
            if as_float(attributes.get('time')) != as_float(record.get('iterations')):
                audit.add('error', 'attributes', f"time {attributes.get('time')}, provenance iterations "
                                                 f"{record.get('iterations')}", run, case)
            if (as_float(attributes.get('outlet_reverse_flow')) or 0) > 0:
                flags[run]['outlet_reverse_flow'].append(case)
            length = as_float(attributes.get('recirculation_length_mm'))
            if length is not None and length < SHORT_RECIRCULATION_MM:
                flags[run]['short_recirculation'].append(case)
    audit.summary['attributes'] = {run: {name: {'count': len(cases), 'cases': cases} for name, cases in f.items()}
                                   for run, f in flags.items()}
    return values


def level_difference(name, value, reference, reference_fx, inlet):
    """
    difference of an attribute from its value on the reference level, on a scale that stays meaningful when the
    reference is close to 0: lift relative to the drag, the wake velocity relative to the inlet velocity, the stagnation
    and base pressure relative to the dynamic pressure; None when there is no scale (no recirculation)
    """
    if name.startswith('Fy'):
        scale = abs(reference_fx)
    elif name == 'wake_min_u_x':
        scale = inlet
    elif name in ('front_pressure', 'base_pressure'):
        scale = 0.5 * inlet ** 2
    elif name.startswith('recirculation'):
        scale = abs(reference) if abs(reference) >= 1 else 0   # below 1 mm / 1 mm^2 there is no recirculation
    else:
        scale = abs(reference)
    return (value - reference) / scale if scale > 0 else None


def check_levels(audit, runs, members, manifest, values, reference='16x'):
    """
    the attributes of every other level against the reference level: per run, attribute, size and family, a
    difference far from the others is an outlier (OUTLIER_MADS, OUTLIER_FLOOR); median of |difference| for the paper
    """
    inlet = inlet_velocity()
    differences = {}
    for case in members:
        fine = values.get((reference, case))
        if not fine:
            continue
        for run in runs:
            coarse = values.get((run, case))
            if run == reference or not coarse:
                continue
            for name in LEVEL_ATTRIBUTES:
                value, ref = as_float(coarse.get(name)), as_float(fine.get(name))
                if value is None or ref is None:
                    continue
                difference = level_difference(name, value, ref, as_float(fine.get('Fx')) or 0, inlet)
                if difference is not None:
                    differences.setdefault((run, name), {})[case] = difference

    medians = {}
    for (run, name), by_case in sorted(differences.items()):
        medians.setdefault(run, {})[name] = round(100 * float(np.median(np.abs(list(by_case.values())))), 3)
        classes = {}
        for case, difference in by_case.items():
            classes.setdefault((manifest[case]['size'], manifest[case]['geometry']), {})[case] = difference
        for cases in classes.values():
            numbers = np.array(list(cases.values()))
            median = np.median(numbers)
            spread = 1.4826 * np.median(np.abs(numbers - median))
            for case, difference in cases.items():
                if abs(difference - median) > max(OUTLIER_MADS * spread, OUTLIER_FLOOR):
                    audit.add('warning', 'levels', f"{name}: {100 * difference:+.1f} % against {reference}, "
                                                   f"class median {100 * median:+.1f} %", run, case)
    audit.summary['levels'] = {'reference': reference, 'median_abs_difference_%': medians}


def check_mesh_study(audit, root, members):
    """tools/mesh_study.py of every case and attribute (dataset/build_dataset.py mesh_study_rows)"""
    rows = list(mesh_study_rows(root, members))
    per_attribute = {}
    for row in rows:
        per_attribute.setdefault(row['attribute'], []).append(row)
    summary = {}
    for name, found in per_attribute.items():
        kinds = Counter(row['convergence'] for row in found)
        monotonic = [row for row in found if row['convergence'] == 'monotonic']
        summary[name] = {'cases': len(found), 'convergence': dict(kinds),
                         'order': statistics([row['order'] for row in monotonic if not math.isnan(row['order'])]),
                         'gci_16x_%': statistics([row.get('gci_16x_%') for row in monotonic
                                                  if as_float(row.get('gci_16x_%')) is not None])}
        if kinds['none']:   # a finding for the paper (e.g. lift close to 0), not an error of the data
            audit.add('info', 'mesh_study', f"{name}: no convergence in {kinds['none']} of {len(found)} cases")
    if len({row['case_name'] for row in rows}) != len(members):
        audit.add('error', 'mesh_study', f"{len({row['case_name'] for row in rows})} cases in the mesh study, "
                                         f"{len(members)} in the dataset")
    audit.summary['mesh_study'] = summary


def check_splits(audit, dataset, members, manifest, output):
    """the fixed splits (dataset/splits.py), written to <output>/splits.csv"""
    split_of = assign_splits([manifest[case] for case in members], dataset['splits'], dataset['placement'])
    missing = [case for case in members if case not in split_of]
    for case in missing:
        audit.add('error', 'splits', "no split", case=case)
    test = Counter(f"{manifest[c]['size']} {manifest[c]['geometry']}" for c in members if split_of.get(c) == 'test')
    audit.summary['splits'] = {'counts': dict(Counter(split_of.values())), 'test_per_class': dict(test)}
    with open(output / 'splits.csv', 'w', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(['case_name', 'split'])
        writer.writerows((case, split_of.get(case, '')) for case in sorted(members))


def check_geometry(audit, members, manifest):
    """the .geo files of every case, written again from the seed (geometries/write_geo.py create_geo)"""
    from geometries.sampling import case_rng, get_coordinates
    from geometries.write_geo import write_geo

    config = load_config()
    resolutions = config['default_resolutions']
    with tempfile.TemporaryDirectory() as folder:
        for case in members:
            row = manifest[case]
            rng = case_rng(config['seed'], row['size'], row['geometry'], int(row['index']))
            obstacle = get_coordinates(row['geometry'], row['size'], config, rng)
            if str(obstacle['attempts']) != row['attempts'] or str(config['seed']) != row['seed']:
                audit.add('error', 'geometry', f"seed {row['seed']} / attempts {row['attempts']} of the manifest, "
                                               f"{config['seed']} / {obstacle['attempts']} again", case=case)
            for resolution in resolutions:
                again = Path(folder) / f"{resolution}.geo"
                write_geo(again, row['geometry'], obstacle['coordinates'], obstacle['arcs'],
                          config['resolutions'][resolution], config['domain'])
                released = config['paths']['geo_files'] / resolution / f"{case}.geo"
                if not released.exists():
                    audit.add('error', 'geometry', f"no {resolution} .geo file", case=case)
                elif released.read_bytes() != again.read_bytes():
                    audit.add('error', 'geometry', f"{resolution} .geo file differs from the one written again",
                              case=case)
    audit.summary['geometry'] = {'cases': len(members), 'resolutions': resolutions}


def check_leftovers(audit, root, runs, members, dropped):
    """files the production should not have left: seeds of finished cases, work folders, local archives, hold.txt"""
    root, members, dropped = Path(root), set(members), set(dropped)
    seeds = sorted(p.name for p in (root / 'seeds').iterdir()) if (root / 'seeds').exists() else []
    for name in seeds:
        level = 'warning' if name in members else 'info'
        audit.add(level, 'leftovers', f"seed {name}" + (" (dropped case)" if name in dropped else ""), 'base_lu',
                  name if name in members | dropped else '')
    for run in runs:
        work = root / 'work' / run
        for path in sorted(work.iterdir()) if work.exists() else []:
            audit.add('warning', 'leftovers', f"work folder {path.name}", run)
        for path in sorted((root / 'archives' / run).glob('*.tar.zst*')):
            audit.add('warning', 'leftovers', f"local archive {path.name}", run)
    if (root / 'hold.txt').exists():
        audit.add('warning', 'leftovers', "hold.txt exists")
    audit.summary['leftovers'] = {
        'seeds': len(seeds),
        'superseded_results': {run: len(list((root / 'results' / run / 'superseded').glob('*_provenance.json')))
                               for run in runs},
        'case_settings_lines': len(ledger.task_lines(root / 'case_settings.txt')),
        'iterations_lines': len(ledger.task_lines(root / 'iterations.txt'))}


# ------------------------------------------------------------------------------------------------------- report


def write_report(audit, output, root, checks, elapsed):
    with open(output / 'findings.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['level', 'check', 'run', 'case', 'message'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(audit.findings)
    revision = REPO_ROOT / 'REVISION'
    audit.summary['audit'] = {'data_root': str(root), 'checks': checks, 'elapsed_s': round(elapsed, 1),
                              'revision': revision.read_text().strip() if revision.exists() else '',
                              'finished': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                              'errors': audit.count(level='error'), 'warnings': audit.count(level='warning')}
    (output / 'summary.json').write_text(json.dumps(audit.summary, indent=1, default=str) + '\n')

    lines = [f"# Dataset audit", "", f"Data root `{root}`, revision `{audit.summary['audit']['revision']}`, "
             f"{audit.summary['audit']['finished']}, {elapsed / 60:.1f} min.", "",
             "| Check | Errors | Warnings | Info |", "|---|---|---|---|"]
    lines += [f"| {check} | {audit.count(check, 'error')} | {audit.count(check, 'warning')} | "
              f"{audit.count(check, 'info')} |" for check in checks]
    for check in checks:
        found = [f for f in audit.findings if f['check'] == check and f['level'] != 'info']
        if found:
            lines += ["", f"## {check}", ""]
            lines += [f"- {f['level']}: {f['run']} {f['case']} {f['message']}".replace('  ', ' ') for f in found[:LISTED]]
            if len(found) > LISTED:
                lines.append(f"- ... {len(found) - LISTED} more in findings.csv")
    lines += ["", "The numbers of every check are in summary.json.", ""]
    (output / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description="Final audit of the dataset before it is frozen")
    parser.add_argument("output", help="report folder")
    parser.add_argument("--remote", action="store_true", help="compare the archives with the remote (rclone)")
    parser.add_argument("--skip", nargs='*', default=[], choices=CHECKS, help="checks to leave out")
    parser.add_argument("--cases", help="file with the cases to check (the members check counts all of them)")
    args = parser.parse_args()

    start = time.time()
    root = ledger.data_root()
    _, dataset = topup.load_settings(root)
    runs = list(dataset['runs'])
    states = ledger.read_states(runs, root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checks = [check for check in CHECKS if check not in args.skip]
    audit = Audit()

    members, dropped, reserves, manifest = check_members(audit, root, dataset, states)
    if args.cases:
        selected = set(Path(args.cases).read_text().split())
        members = [case for case in members if case in selected]
    print(f"{len(members)} cases, {len(dropped)} dropped, {len(reserves)} reserve cases in use", flush=True)

    records, values = {}, {}
    for check in checks:
        if check == 'states':
            check_states(audit, runs, members, states)
        elif check == 'provenance':
            records = check_provenance(audit, root, dataset, runs, members, states)
        elif check == 'archives':
            check_archives(audit, root, runs, members, remote=args.remote)
        elif check == 'arrays':
            check_arrays(audit, root, dataset, runs, members, manifest, records)
        elif check == 'attributes':
            values = check_attributes(audit, root, runs, members, records)
        elif check == 'levels':
            check_levels(audit, runs, members, manifest, values)
        elif check == 'mesh_study':
            check_mesh_study(audit, root, members)
        elif check == 'splits':
            check_splits(audit, dataset, members, manifest, output)
        elif check == 'geometry':
            check_geometry(audit, members, manifest)
        elif check == 'leftovers':
            check_leftovers(audit, root, runs, members, dropped)
        print(f"{check}: {audit.count(check, 'error')} errors, {audit.count(check, 'warning')} warnings "
              f"({time.time() - start:.0f} s)", flush=True)

    write_report(audit, output, root, checks, time.time() - start)
    print(f"report in {output}: {audit.count(level='error')} errors, {audit.count(level='warning')} warnings")
    sys.exit(1 if audit.count(level='error') else 0)


if __name__ == "__main__":
    main()
