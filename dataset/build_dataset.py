"""
Builds the dataset files from the per-case results of a data root (hpc/run_task.py). The cases are those of the
production: the active cases without the dropped ones (hpc/topup.py dataset_cases), so reserve cases are in and cases
without a steady solution are out.

    python -m dataset.build_dataset <output folder>                      data root: MSCOFD_DATA_ROOT
    python -m dataset.build_dataset <output folder> --allow-missing      leave out cases with missing runs (pilots)

Output in <output folder>:

    <run>/<split>.h5      fields (N, 6, nx, ny) float32, one sample per chunk, not compressed; case_name (N,); time
                          (N,); attributes: channels, run, label, grid, mean / std per channel over the train split of
                          the run (normalisation). Row i of every <run>/<split>.h5 is the same case.
    metadata.parquet/csv  one row per case: dataset_index, split, the manifest (sampled parameters, geometry), and per
                          run <run>.<field>: status, iterations, residuals, mesh variant, cap, cells, solver and task
                          time, peak memory, host, CPU, image digest, code revision, and the OpenFOAM attributes
    surface.parquet       pressure and wall shear along the obstacle, every case and run (one row per obstacle face)
    mesh_study.parquet    order, extrapolated value, errors of every level and GCI of every attribute and case
    dropped_cases.csv     the cases dropped without a steady solution, with their parameters and residuals
    dataset_info.json     name, version, counts per split, runs and grids, code revision, image digest
    checksums.md5         MD5 of every file

Needs numpy, h5py and pyarrow (the container).
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.splits import SPLITS, assign_splits  # noqa: E402
from extraction.build_arrays import CHANNELS  # noqa: E402
from hpc import ledger, topup  # noqa: E402
from tools import mesh_study  # noqa: E402

LABELS = {'0.25x': '2 mm, second order', 'base_fo': '1 mm, first order', 'base_lu': '1 mm, second order',
          '4x': '0.5 mm, second order', '16x': '0.25 mm, second order'}
PROVENANCE = ['status', 'iterations', 'solver_runs', 'residual_Ux', 'residual_Uy', 'residual_p', 'mesh_ok',
              'mesh_variant', 'end_time_override', 'templates', 'init_from', 'cells', 'exported_time',
              'execution_time_s', 'clock_time_s',
              'peak_memory_kb', 'host', 'cpu_model', 'node_cpus_allocated', 'slurm_job_id', 'container_digest',
              'code_revision', 'finished']


def number_or_text(value):
    """a manifest or provenance value as int or float when it is one, None when empty, else the text"""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if text == '':
        return None
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def case_files(root, run, case):
    folder = Path(root) / 'results' / run
    return {'npz': folder / f"{case}.npz", 'attributes': folder / f"{case}_attributes.json",
            'provenance': folder / f"{case}_provenance.json", 'surface': folder / f"{case}_surface.csv"}


def complete(root, runs, case):
    """the runs of the case whose array, attributes and provenance are all there"""
    return [run for run in runs if all(case_files(root, run, case)[k].exists() for k in ('npz', 'attributes', 'provenance'))]


def md5_of(path, block=1 << 24):
    digest = hashlib.md5()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(block), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_fields(output, root, run, split_cases):
    """<run>/<split>.h5 for every split; the normalisation comes from the train split (all cases when it is empty)"""
    import h5py

    stats_split = 'train' if split_cases.get('train') else None
    total = sumsq = None
    count = 0
    folder = output / run
    folder.mkdir(parents=True, exist_ok=True)
    for split, cases in split_cases.items():
        if not cases:
            continue
        with np.load(case_files(root, run, cases[0])['npz']) as data:
            shape = data['sample'].shape
        with h5py.File(folder / f"{split}.h5", 'w') as h5:
            fields = h5.create_dataset('fields', (len(cases),) + shape, dtype='float32', chunks=(1,) + shape)
            times = np.zeros(len(cases))
            for i, case in enumerate(cases):
                with np.load(case_files(root, run, case)['npz']) as data:
                    sample = data['sample']
                    times[i] = float(data['time'])
                if sample.shape != shape:
                    raise ValueError(f"{run} {case}: array shape {sample.shape}, expected {shape}")
                fields[i] = sample
                if stats_split is None or split == stats_split:
                    values = sample.reshape(shape[0], -1).astype(np.float64)
                    total = values.sum(axis=1) if total is None else total + values.sum(axis=1)
                    sumsq = (values ** 2).sum(axis=1) if sumsq is None else sumsq + (values ** 2).sum(axis=1)
                    count += values.shape[1]
            h5.create_dataset('case_name', data=np.array(cases, dtype=object), dtype=h5py.string_dtype())
            h5.create_dataset('time', data=times)
            h5.attrs.update({'run': run, 'label': LABELS.get(run, run), 'split': split, 'channels': CHANNELS,
                             'layout': 'fields[sample, channel, x, y], x along the flow from the inlet',
                             'grid': list(shape[1:])})
    mean = total / count
    std = np.sqrt(np.maximum(sumsq / count - mean ** 2, 0))
    for split, cases in split_cases.items():
        if cases:
            with h5py.File(folder / f"{split}.h5", 'a') as h5:
                h5.attrs.update({'mean': mean, 'std': std, 'normalisation_split': stats_split or 'all'})
    return {'grid': list(shape[1:]), 'mean': mean.tolist(), 'std': std.tolist()}


def metadata_row(root, runs, row, split, index):
    """one row of metadata.parquet: the manifest row, the split and per run provenance and attributes"""
    out = {'dataset_index': index, 'split': split}
    out.update({key: number_or_text(value) if key not in ('case_name', 'size', 'geometry') else value
                for key, value in row.items()})
    for run in runs:
        files = case_files(root, run, row['case_name'])
        provenance = json.loads(files['provenance'].read_text())
        for key in PROVENANCE:
            value = provenance.get(key)
            text = key in ('mesh_variant', 'templates', 'init_from', 'code_revision', 'host', 'slurm_job_id',
                           'cpu_model', 'container_digest')
            out[f"{run}.{key}"] = value if text else number_or_text(value)
        timings = provenance.get('timings_s') or {}
        out[f"{run}.task_time_s"] = round(sum(timings.values()), 2) if timings else None
        for key, value in json.loads(files['attributes'].read_text()).items():
            if key not in ('time', 'cells', 'faces') and isinstance(value, (int, float)):
                out[f"{run}.{key}"] = value
    return out


def write_table(rows, path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = list(dict.fromkeys(key for row in rows for key in row))
    table = pa.table({name: [row.get(name) for row in rows] for name in columns})
    pq.write_table(table, path)
    return columns


SURFACE_COLUMNS = ['x_mm', 'y_mm', 'p', 'wall_shear_x', 'wall_shear_y']


def write_surface(root, runs, cases, path, batch=200):
    """surface.parquet: one row per obstacle face of every case and run, written in batches of cases"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([('case_name', pa.string()), ('run', pa.string())] +
                       [(name, pa.float64()) for name in SURFACE_COLUMNS])
    with pq.ParquetWriter(path, schema) as writer:
        for start in range(0, len(cases), batch):
            names, run_names, columns = [], [], []
            for case in cases[start:start + batch]:
                for run in runs:
                    surface = case_files(root, run, case)['surface']
                    if surface.exists():
                        data = np.loadtxt(surface, delimiter=',', skiprows=1, ndmin=2)
                        names += [case] * len(data)
                        run_names += [run] * len(data)
                        columns.append(data)
            if columns:
                data = np.concatenate(columns)
                writer.write_table(pa.table({'case_name': names, 'run': run_names,
                                             **{n: data[:, i] for i, n in enumerate(SURFACE_COLUMNS)}}, schema=schema))


def mesh_study_rows(root, cases):
    """tools/mesh_study.py on base_lu / 4x / 16x of every case, with the error of 0.25x and base_fo as well"""
    levels = {'base': 'base_lu', '4x': '4x', '16x': '16x'}
    for case in cases:
        attributes = {}
        for level, run in list(levels.items()) + [('0.25x', '0.25x'), ('base_fo', 'base_fo')]:
            path = case_files(root, run, case)['attributes']
            if path.exists():
                attributes[level] = json.loads(path.read_text())
        if not all(level in attributes for level in levels):
            continue
        cells = [float(attributes[m]['cells']) for m in mesh_study.MESHES]
        for name in mesh_study.ATTRIBUTES:
            phi = [attributes[m].get(name) for m in mesh_study.MESHES]
            if any(not isinstance(v, (int, float)) or math.isnan(v) for v in phi):
                continue
            row = {'case_name': case, 'attribute': name, 'base': phi[2], '4x': phi[1], '16x': phi[0]}
            row.update(mesh_study.study(*phi, *cells))
            for extra in ('0.25x', 'base_fo'):
                value = attributes.get(extra, {}).get(name)
                if isinstance(value, (int, float)):
                    row[extra] = value
                    if row.get('extrapolated'):
                        row[f"{extra}_error_%"] = 100 * abs(value - row['extrapolated']) / abs(row['extrapolated'])
            yield row


def main():
    parser = argparse.ArgumentParser(description="Build the dataset files from the per-case results")
    parser.add_argument("output", help="output folder")
    parser.add_argument("--allow-missing", action="store_true", help="leave out cases with missing runs")
    args = parser.parse_args()

    root = ledger.data_root()
    slurm, dataset = topup.load_settings(root)
    runs = list(dataset['runs'])
    states = ledger.read_states(runs, root)
    cases, dropped, reserves = topup.dataset_cases(root, dataset, states, topup.cases_file_of(root))
    members = [case for case in cases if case not in set(dropped)]
    incomplete = {case: sorted(set(runs) - set(complete(root, runs, case))) for case in members}
    incomplete = {case: missing for case, missing in incomplete.items() if missing}
    if incomplete and not args.allow_missing:
        sys.exit(f"{len(incomplete)} cases have missing runs, e.g. {list(incomplete.items())[:3]}")
    members = [case for case in members if case not in incomplete]
    print(f"{len(members)} cases ({len(reserves)} reserve cases in use), {len(dropped)} dropped, "
          f"{len(incomplete)} left out with missing runs")

    manifest = {row['case_name']: row for row in ledger.read_manifest(root / 'geometry' / 'manifest.csv')}
    split_of = assign_splits([manifest[case] for case in members], dataset['splits'], dataset['placement'])
    order = SPLITS + sorted({s for s in split_of.values() if s not in SPLITS})
    members.sort(key=lambda case: (order.index(split_of[case]), case))   # dataset index: split, then name
    split_cases = {split: [case for case in members if split_of[case] == split] for split in order}

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    grids = {}
    for run in runs:
        grids[run] = write_fields(output, root, run, split_cases)
        print(f"{run}: {len(members)} samples, grid {grids[run]['grid']}")

    rows = [metadata_row(root, runs, manifest[case], split_of[case], i) for i, case in enumerate(members)]
    columns = write_table(rows, output / 'metadata.parquet')
    with open(output / 'metadata.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    write_surface(root, runs, members, output / 'surface.parquet')
    write_table(list(mesh_study_rows(root, members)) or [{'case_name': None}], output / 'mesh_study.parquet')

    with open(output / 'dropped_cases.csv', 'w', newline='') as file:
        fields = list(next(iter(manifest.values())).keys()) + ['iterations', 'residual_Ux', 'residual_Uy', 'residual_p',
                                                                'reason']
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for case in dropped:
            provenance = case_files(root, 'base_lu', case)['provenance']
            record = json.loads(provenance.read_text()) if provenance.exists() else {}
            writer.writerow({**manifest[case], **{k: record.get(k) for k in fields[-5:-1]},
                             'reason': 'no steady solution: the 1 mm second-order run stopped at its iteration cap'})

    revisions = sorted({row.get(f"{run}.code_revision") for row in rows for run in runs} - {None})
    digests = sorted({row.get(f"{run}.container_digest") for row in rows for run in runs} - {None, ''})
    info = {'name': dataset['name'], 'version': dataset['version'], 'created': datetime.now().astimezone().isoformat(timespec='seconds'),
            'cases': len(members), 'splits': {split: len(c) for split, c in split_cases.items() if c},
            'dropped_cases': len(dropped), 'reserve_cases': len(reserves), 'channels': CHANNELS,
            'runs': {run: {'label': LABELS.get(run, run), **grids[run]} for run in runs},
            'code_revisions': revisions, 'container_digests': digests,
            'builder_revision': (Path(__file__).resolve().parent.parent / 'REVISION').read_text().strip()
            if (Path(__file__).resolve().parent.parent / 'REVISION').exists() else ''}
    (output / 'dataset_info.json').write_text(json.dumps(info, indent=1) + '\n')

    with open(output / 'checksums.md5', 'w') as file:
        for path in sorted(p for p in output.rglob('*') if p.is_file() and p.name != 'checksums.md5'):
            file.write(f"{md5_of(path)}  {path.relative_to(output).as_posix()}\n")
    print(f"dataset written to {output}: {json.dumps(info['splits'])}")


if __name__ == "__main__":
    main()
