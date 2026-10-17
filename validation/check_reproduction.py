"""
Reproduction check of the released results (docs/validation.md): the archives give back the released arrays and
attributes, and the container gives back the solutions. The production root is only read.

    python validation/check_reproduction.py archives <production root> <work folder> <output> <run>:<case> ...
    python validation/check_reproduction.py setup <production root> <check root> <case> ...
    python validation/check_reproduction.py solved <check root> <production root> <output>

archives   every <run>:<case>: its archive (the live row of archives/archive_index.csv, the newest when there are
           more) is copied from the remote with rclone, its MD5 checked against the index and unpacked (zstd, tar)
           into <work folder>. The arrays are made again from the restored case (extraction/resample_vtk.py,
           extraction/build_arrays.py) and the attributes computed again with OpenFOAM
           (openfoam/scripts/openfoam_attributes.sh, tools/openfoam_attributes.py collect_case). Both are compared
           with the released results: arrays identical, attributes identical. The work folder is removed at the end.
setup      a data root for cases solved again (validation/repro_check: 0.25x and base_lu with the production
           settings, from rest): the manifest rows and .geo files of the cases, cases.txt, dataset.yaml and
           slurm.yaml. The runs are then made with hpc/topup.py in that root.
solved     every run of the check root against the released one: iterations, the largest and the RMS difference of
           every channel (fluid points) and the largest relative difference of the attributes.

Every command writes reproduction_<command>.csv to <output> and prints a summary; exit code 1 when a released result
is not given back (archives: arrays or attributes differ; solved: a field differs by more than SOLVED_TOLERANCE).

archives needs the container (OpenFOAM, VTK, rclone, zstd) with MSCOFD_REMOTE's rclone configuration.
"""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils import load_config, write_csv  # noqa: E402
from dataset.results import load_attributes, load_sample  # noqa: E402
from hpc import ledger  # noqa: E402
from hpc.upload import md5_of  # noqa: E402

ATTRIBUTES_SCRIPT = REPO / 'openfoam' / 'scripts' / 'openfoam_attributes.sh'
CHANNELS = {'u_x': 0, 'u_y': 1, 'p': 2}
SKIP = ('time', 'cells', 'faces')
# solved: largest difference of a field from the released one, relative to its largest magnitude. Both runs converge
# to 1e-6 from the same start with the same settings; the initialization check found differences below 1e-4 %
SOLVED_TOLERANCE = 1e-5
# folders of the production root a check root must not be in (as validation/select_loose_audit.py)
PRODUCTION_FOLDERS = ('geometry', 'state', 'work', 'seeds', 'results', 'archives', 'logs', 'dataset', 'config',
                      'container', 'repo', 'cache')


def archive_row(production, run, case):
    """the live row of a run in archives/archive_index.csv (the newest when there are more), None when there is none"""
    with open(Path(production) / 'archives' / 'archive_index.csv', newline='') as file:
        rows = [row for row in csv.DictReader(file) if row['run'] == run and row['case'] == case and
                '/superseded/' not in row['remote']]
    return max(rows, key=lambda row: row['uploaded']) if rows else None


def attribute_difference(again, released):
    """largest relative difference over the attributes (lift relative to the drag), and its attribute"""
    worst, name = 0.0, ''
    for key, value in released.items():
        if key in SKIP or not isinstance(value, (int, float)) or not isinstance(again.get(key), (int, float)):
            continue
        scale = abs(released['Fx']) if key.startswith('Fy') else abs(value)
        difference = abs(again[key] - value) / scale if scale > 0 else abs(again[key] - value)
        if difference > worst:
            worst, name = difference, key
    missing = sorted(k for k in released if k not in SKIP and k not in again)
    return worst, name, missing


def field_difference(again, released):
    """per channel: largest |difference| relative to the largest |released value|, and the RMS difference relative to
    the RMS of the released field, over the fluid points"""
    fluid = released[4] != 1
    row = {}
    for name, channel in CHANNELS.items():
        a, b = again[channel][fluid].astype(float), released[channel][fluid].astype(float)
        peak, rms = np.abs(b).max(), np.sqrt(np.mean(b ** 2))
        row[f"max_{name}"] = float(np.abs(a - b).max() / peak) if peak > 0 else float(np.abs(a - b).max())
        row[f"rms_{name}"] = float(np.sqrt(np.mean((a - b) ** 2)) / rms) if rms > 0 else 0.0
    row['geometry_equal'] = bool(np.array_equal(again[3:], released[3:]))
    return row


def restore(row, work):
    """copies the archive of an index row from the remote, checks its MD5 and unpacks it into work"""
    work.mkdir(parents=True, exist_ok=True)
    archive = work / Path(row['remote']).name
    subprocess.run(['rclone', 'copyto', row['remote'], str(archive)], check=True)
    if md5_of(archive) != row['md5']:
        raise RuntimeError(f"MD5 of {archive.name} differs from archive_index.csv")
    subprocess.run(['bash', '-o', 'pipefail', '-c', f"zstd -dc '{archive}' | tar -xf - -C '{work}'"], check=True)
    archive.unlink()


def archives(production, work, output, tasks):
    from extraction.build_arrays import build_sample
    from extraction.resample_vtk import resample_case
    from tools.openfoam_attributes import collect_case, obstacle_geometry

    config = load_config()
    domain = config['domain']
    manifest = {row['case_name']: row for row in ledger.read_manifest(Path(production) / 'geometry' / 'manifest.csv')}
    rows, failed = [], False
    for task in tasks:
        run, case = task.split(':')
        result = {'run': run, 'case': case}
        row = archive_row(production, run, case)
        if row is None:
            result['problem'] = 'no live archive row'
            rows.append(result)
            failed = True
            continue
        folder = Path(work) / f"{run}-{case}"
        restore(row, folder)
        nx, ny = config['extraction'][config['runs'][run]['resolution']]
        fields = resample_case(folder / case, nx, ny)
        sample = build_sample(fields['u_x'], fields['u_y'], fields['p'], fields['valid']).astype(np.float32)
        with np.load(Path(production) / 'results' / run / f"{case}.npz") as data:
            released, time = data['sample'], float(data['time'])
        same_shape = sample.shape == released.shape
        result.update({'arrays_identical': same_shape and bool(np.array_equal(sample, released)),
                       'arrays_max_difference': float(np.abs(sample.astype(float) - released).max())
                       if same_shape else float('nan'),
                       'time_identical': fields['time'] == time})

        box, x_centroid, x_wake = obstacle_geometry(manifest[case], domain)
        again = folder / f"{case}_attributes_again"
        command = (['bash', str(ATTRIBUTES_SCRIPT), str(folder / case), str(again)] +
                   [str(v) for v in box + [x_centroid, x_wake] + [domain['length'] / 1000, domain['height'] / 1000]])
        with open(folder / 'log.attributes_again', 'w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        worst, name, missing = attribute_difference(collect_case(str(again)), load_attributes(production, run, case))
        result.update({'attributes_max_difference': worst, 'attributes_worst': name,
                       'attributes_missing': ' '.join(missing)})
        failed |= not (result['arrays_identical'] and result['time_identical']) or worst > 0 or bool(missing)
        rows.append(result)
        shutil.rmtree(folder)
        print(f"  {run} {case}: arrays identical {result['arrays_identical']} (largest difference "
              f"{result['arrays_max_difference']:.3g}), attributes largest difference {worst:.3g} {name}", flush=True)
    write_csv(rows, Path(output) / 'reproduction_archives.csv', list(dict.fromkeys(k for r in rows for k in r)))
    return failed


def check_roots(production, check):
    if check == production or production in check.parents and check.relative_to(production).parts[0] in \
            PRODUCTION_FOLDERS:
        sys.exit(f"{check} is the production root or one of its data folders")


def setup(production, check, cases):
    production, check = Path(production).resolve(), Path(check).resolve()
    check_roots(production, check)
    with open(production / 'geometry' / 'manifest.csv', newline='') as file:
        reader = csv.DictReader(file)
        columns, manifest = reader.fieldnames, {row['case_name']: row for row in reader}
    unknown = [case for case in cases if case not in manifest]
    if unknown:
        sys.exit(f"not in the manifest: {unknown}")
    (check / 'geometry').mkdir(parents=True, exist_ok=True)
    with open(check / 'geometry' / 'manifest.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(manifest[case] for case in cases)
    (check / 'cases.txt').write_text(''.join(f"{case}\n" for case in cases))
    for name in ('dataset.yaml', 'slurm.yaml'):
        shutil.copy2(REPO / 'validation' / 'repro_check' / name, check / name)
    import yaml
    runs = yaml.safe_load((check / 'dataset.yaml').read_text())['runs']
    for resolution in sorted({settings['resolution'] for settings in runs.values()}):
        folder = check / 'geometry' / 'geo-files' / resolution
        folder.mkdir(parents=True, exist_ok=True)
        for case in cases:
            shutil.copy2(production / 'geometry' / 'geo-files' / resolution / f"{case}.geo", folder / f"{case}.geo")
    print(f"{len(cases)} cases, runs {', '.join(runs)}: written to {check}")


def iterations_of(root, run, case):
    path = Path(root) / 'results' / run / f"{case}_provenance.json"
    return json.loads(path.read_text()).get('iterations') if path.exists() else None


def solved(check, production, output):
    import yaml
    runs = yaml.safe_load((Path(check) / 'dataset.yaml').read_text())['runs']
    cases = (Path(check) / 'cases.txt').read_text().split()
    rows, failed = [], False
    for case in cases:
        for run in runs:
            if not (Path(check) / 'results' / run / f"{case}.npz").exists():
                rows.append({'run': run, 'case': case, 'problem': 'not solved'})
                failed = True
                continue
            result = {'run': run, 'case': case, 'iterations': iterations_of(check, run, case),
                      'iterations_released': iterations_of(production, run, case)}
            result.update(field_difference(load_sample(check, run, case), load_sample(production, run, case)))
            worst, name, missing = attribute_difference(load_attributes(check, run, case),
                                                        load_attributes(production, run, case))
            result.update({'attributes_max_difference': worst, 'attributes_worst': name})
            largest = max(result[f"max_{name}"] for name in CHANNELS)
            failed |= largest > SOLVED_TOLERANCE or not result['geometry_equal']
            rows.append(result)
            print(f"  {run} {case}: iterations {result['iterations']} / {result['iterations_released']}, largest field "
                  f"difference {largest:.3g}, attributes {worst:.3g} {name}", flush=True)
    write_csv(rows, Path(output) / 'reproduction_solved.csv', list(dict.fromkeys(k for r in rows for k in r)))
    return failed


def main():
    if len(sys.argv) < 4 or sys.argv[1] not in ('archives', 'setup', 'solved'):
        sys.exit(__doc__)
    command = sys.argv[1]
    if command == 'setup':
        setup(sys.argv[2], sys.argv[3], sys.argv[4:])
        return
    output = Path(sys.argv[4])
    output.mkdir(parents=True, exist_ok=True)
    if command == 'archives':
        failed = archives(Path(sys.argv[2]), Path(sys.argv[3]), output, sys.argv[5:])
    else:
        failed = solved(Path(sys.argv[2]), Path(sys.argv[3]), output)
    print(f"{'NOT reproduced' if failed else 'reproduced'}; written to {output}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
