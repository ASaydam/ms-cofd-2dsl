"""
Cases and data root of the loose-criterion audit (validation/loose_audit/dataset.yaml, docs/validation.md):

    python3 validation/select_loose_audit.py <production data root> <audit data root>

Cases: the test split of the dataset (dataset/splits.py on the member cases of the production), 7 per size and family,
stratified by the iterations of the converged 0.25 mm run. The test cases of a class are sorted by those iterations (a
run still going counts with its current iteration, a lower bound), cut into 7 bins of (nearly) equal count, and one case
is drawn per bin with random.Random("2026-loose-<size>-<family>"). Also one ood_blockage case per class
(random.Random("2026-loose-ood-<size>-<family>")), reported separately. selection.csv records the bin of every case
and its inclusion probability (1 / cases in its bin, or in its ood_blockage class), so results can be weighted to the
whole split.

The audit root gets selection.csv, cases.txt, geometry/manifest.csv (the selected rows only), the .geo files of the
1 mm and 0.25 mm meshes, and dataset.yaml and slurm.yaml from validation/loose_audit/. The production root is only
read; the audit root must not be the production root or one of its data folders.
"""

import csv
import json
import random
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'hpc'))

import ledger  # noqa: E402
import topup  # noqa: E402
from dataset.splits import assign_splits  # noqa: E402

BINS = 7
RESOLUTIONS = ('base', '16x')   # meshes of the audit runs (base_lu, base_lu_loose_scratch; 16x_loose, 16x_loose_scratch)
PRODUCTION_FOLDERS = ('geometry', 'state', 'work', 'seeds', 'results', 'archives', 'logs', 'dataset', 'config',
                      'container', 'repo', 'cache')
FIELDS = ['case', 'size', 'family', 'group', 'bin', 'bin_cases', 'inclusion_probability', 'iterations_16x',
          'iterations_16x_final', 'blockage_ratio', 'mesh_variant_base', 'mesh_variant_16x', 'cells_16x']


def bins_of(names, count=BINS):
    """names (sorted) cut into count consecutive bins of nearly equal size, the larger bins first"""
    size, extra = divmod(len(names), count)
    bins, start = [], 0
    for i in range(count):
        end = start + size + (1 if i < extra else 0)
        bins.append(names[start:end])
        start = end
    return [b for b in bins if b]


def current_iteration(log):
    """last "Time = <n>" of a running simpleFoam log (read from its end)"""
    with open(log, 'rb') as file:
        file.seek(0, 2)
        file.seek(max(0, file.tell() - (1 << 16)))
        lines = file.read().decode(errors='replace').splitlines()
    times = [line.split('=')[1].strip() for line in lines if line.startswith('Time = ')]
    return int(float(times[-1])) if times else 0


def iterations_16x(production, states, cases):
    """{case: (iterations of the 16x run, final)}: the last summary row of a done run, else the current iteration"""
    with open(production / 'results' / '16x' / 'summary.csv', newline='') as file:
        last = {row['case']: row for row in csv.DictReader(file)}
    out = {}
    for case in cases:
        state = states.get(('16x', case), ('', ''))[0]
        summary = int(float(last[case]['iterations'])) if case in last and last[case]['iterations'] else 0
        if state == 'done' and last.get(case, {}).get('status') == 'converged':
            out[case] = (summary, True)
        else:
            log = production / 'work' / '16x' / case / 'log.simpleFoam'
            out[case] = (max(summary, current_iteration(log) if log.exists() else 0), False)
    return out


def provenance_value(production, run, case, key):
    path = production / 'results' / run / f"{case}_provenance.json"
    return json.loads(path.read_text()).get(key, '') if path.exists() else ''


def check_roots(production, audit):
    if audit == production or production in audit.parents and audit.relative_to(production).parts[0] in \
            PRODUCTION_FOLDERS:
        sys.exit(f"{audit} is the production root or one of its data folders")
    if not (production / 'geometry' / 'manifest.csv').exists():
        sys.exit(f"{production} is not a data root (no geometry/manifest.csv)")


def main():
    production, audit = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    check_roots(production, audit)

    _, dataset = topup.load_settings(production)
    states = ledger.read_states(list(dataset['runs']), production)
    cases, dropped, _ = topup.dataset_cases(production, dataset, states, topup.cases_file_of(production))
    members = [case for case in cases if case not in set(dropped)]
    with open(production / 'geometry' / 'manifest.csv', newline='') as file:
        reader = csv.DictReader(file)
        columns, manifest = reader.fieldnames, {row['case_name']: row for row in reader}
    split = assign_splits([manifest[case] for case in members], dataset['splits'], dataset['placement'])
    iterations = iterations_16x(production, states, members)

    classes = {}
    for case in members:
        classes.setdefault((manifest[case]['size'], manifest[case]['geometry']), []).append(case)
    selected = []
    for (size, family), names in sorted(classes.items()):
        test = sorted((c for c in names if split[c] == 'test'), key=lambda c: (iterations[c][0], c))
        rng = random.Random(f"2026-loose-{size}-{family}")
        for number, members_of_bin in enumerate(bins_of(test), 1):
            selected.append((rng.choice(members_of_bin), 'test', number, len(members_of_bin)))
        ood = sorted(c for c in names if split[c] == 'ood_blockage')
        if ood:
            selected.append((random.Random(f"2026-loose-ood-{size}-{family}").choice(ood), 'ood_blockage', '', len(ood)))

    rows = []
    for case, group, number, count in selected:
        row = manifest[case]
        rows.append({'case': case, 'size': row['size'], 'family': row['geometry'], 'group': group, 'bin': number,
                     'bin_cases': count, 'inclusion_probability': f"{1 / count:.6f}",
                     'iterations_16x': iterations[case][0], 'iterations_16x_final': iterations[case][1],
                     'blockage_ratio': row['blockage_ratio'],
                     'mesh_variant_base': provenance_value(production, 'base_lu', case, 'mesh_variant'),
                     'mesh_variant_16x': provenance_value(production, '16x', case, 'mesh_variant'),
                     'cells_16x': provenance_value(production, '16x', case, 'cells')})

    # the audit root (a new one, or the same selection again)
    audit.mkdir(parents=True, exist_ok=True)
    selection = audit / 'selection.csv'
    if selection.exists():
        with open(selection, newline='') as file:
            earlier = [row['case'] for row in csv.DictReader(file)]
        if earlier != [row['case'] for row in rows]:
            sys.exit(f"{selection} exists with another selection; remove it first")
    with open(selection, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (audit / 'cases.txt').write_text(''.join(f"{row['case']}\n" for row in rows))
    (audit / 'geometry').mkdir(exist_ok=True)
    chosen = {row['case'] for row in rows}
    with open(audit / 'geometry' / 'manifest.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(row for name, row in manifest.items() if name in chosen)
    geo = Path(dataset['paths']['geo_files'])
    for resolution in RESOLUTIONS:
        (audit / geo / resolution).mkdir(parents=True, exist_ok=True)
        for case in chosen:
            shutil.copy2(production / geo / resolution / f"{case}.geo", audit / geo / resolution / f"{case}.geo")
    for name in ('dataset.yaml', 'slurm.yaml'):
        shutil.copy2(REPO / 'validation' / 'loose_audit' / name, audit / name)

    test = [row for row in rows if row['group'] == 'test']
    print(f"{len(members)} member cases, {sum(1 for s in split.values() if s == 'test')} in the test split; "
          f"selected {len(test)} test cases ({len(classes)} classes x {BINS} bins) and "
          f"{len(rows) - len(test)} ood_blockage cases; {sum(1 for row in rows if not row['iterations_16x_final'])} "
          f"of them with a 16x run still going; written to {audit}")


if __name__ == '__main__':
    main()
