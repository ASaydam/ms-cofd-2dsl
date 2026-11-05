"""
The numbers of the dataset paper that are computed from the released data, from the release files alone, grouped by
the section of the paper that quotes them (labels in the paper's wording):

    python validation/paper_numbers.py <release folder>

Reads metadata.parquet, mesh_study.parquet, label_uncertainty.parquet (its rows of source "openfoam"),
dropped_cases.csv, dataset_info.json and checksums.md5 of a release folder (dataset/build_dataset.py), and the splits,
placement and cases_per_geometry blocks of config/dataset.yaml, the settings the release was split with. The HDF5
arrays are not read, only their number and sizes are listed. A file that is not in the folder is reported, and the
numbers that need it are left out.

Definitions, each the one behind the number in the paper:

    Table 2         median and 95th percentile over the cases of <run>.cells, <run>.iterations and
                    <run>.execution_time_s (the simpleFoam time on one core, summed over the solver runs of a run)
    excluded cases  dropped_cases.csv; the largest residual of a case is the largest of residual_Ux, residual_Uy and
                    residual_p. Every excluded case was replaced by a reserve case of its size and family (index above
                    cases_per_geometry), so the reserve cases in metadata.parquet give the same counts
    remeshed        runs whose mesh_variant is not 0: meshed again after checkMesh failed (config/dataset.yaml meshing)
    relaxed runs    0.25x runs of the template second-order-relaxed
    splits          the measures and thresholds of dataset/splits.py: a tail holds the cases above the value at the
                    ood_quantile of its measure
    fallback        method "anomalous" of tools/uncertainty.py: fitted order p <= 0, U = 3 x the largest difference
    cost            solver core-hours: execution_time_s summed over all runs. The cost of a 0.25 mm run against a 1 mm
                    (base_lu) run is the ratio of their median solver times, about the product of the ratios of the
                    median cells, the median iterations and the median time per iteration and cell. The last differs
                    between the two CPU models; the paper quotes the one of the E5-2640
    GCI undefined   the three-level convergence of tools/mesh_study.py is not monotonic (its GCI needs monotonic)
    observed order  the order of tools/mesh_study.py, median over the cases of monotonic convergence (all cases, and
                    the range of the medians per family)
    dissipation     median U_% against median U1_% = 3 |phi_0.25 - phi_0.5| (tools/uncertainty.py summary)

Needs numpy, pyarrow and pyyaml (the container).
"""

import csv
import json
import math
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.splits import SPLITS, assign_splits, ood_measures  # noqa: E402
from tools.uncertainty import floats  # noqa: E402

RUNS = [('0.25x', '2 mm'), ('base_fo', '1 mm first order'), ('base_lu', '1 mm second order'), ('4x', '0.5 mm'),
        ('16x', '0.25 mm')]
CASE_FIELDS = ['case_name', 'size', 'geometry', 'index', 'split', 'blockage_ratio', 'rotation', 'shift_x', 'shift_y']
RUN_FIELDS = ['status', 'iterations', 'residual_Ux', 'residual_Uy', 'residual_p', 'mesh_variant', 'templates',
              'init_from', 'cells', 'execution_time_s', 'peak_memory_kb', 'cpu_model', 'container_digest']
ATTRIBUTE_FIELDS = ['base_fo.recirculation_length_mm', '16x.recirculation_length_mm', '16x.outlet_reverse_flow']
RELAXED = 'second-order-relaxed'
REFERENCE_CPU = 'E5-2640'                    # the CPU of most 0.25 mm runs, whose time per iteration the paper quotes
UNDEFINED_GCI = [('Fx', 'drag'), ('Fx_pressure', 'pressure drag'), ('Fx_viscous', 'friction drag'),
                 ('recirculation_length_mm', 'recirculation length'), ('wake_min_u_x', 'wake velocity')]


# ---------------------------------------------------------------------------------------------------------- reading
def load(folder):
    """
    the release files of a folder, None for a file that is not in it
    :return: {'metadata': {column: list}, 'metadata_columns': number of columns, 'mesh_study', 'uncertainty':
              {column: list}, 'dropped': [csv row], 'info': dict, 'checksums': [published path]}
    """
    import pyarrow.parquet as pq

    folder = Path(folder)
    release = dict.fromkeys(['metadata', 'metadata_columns', 'mesh_study', 'uncertainty', 'dropped', 'info',
                             'checksums'])
    if (folder / 'metadata.parquet').exists():
        columns = CASE_FIELDS + [f'{run}.{name}' for run, _ in RUNS for name in RUN_FIELDS] + ATTRIBUTE_FIELDS
        release['metadata'] = pq.read_table(folder / 'metadata.parquet', columns=columns).to_pydict()
        release['metadata_columns'] = len(pq.read_schema(folder / 'metadata.parquet').names)
    if (folder / 'mesh_study.parquet').exists():
        release['mesh_study'] = pq.read_table(folder / 'mesh_study.parquet',
                                              columns=['case_name', 'attribute', 'convergence', 'order']).to_pydict()
    if (folder / 'label_uncertainty.parquet').exists():
        table = pq.read_table(folder / 'label_uncertainty.parquet').to_pydict()
        table.setdefault('source', ['openfoam'] * len(table['case_name']))     # files before the array rows
        release['uncertainty'] = table
    if (folder / 'dropped_cases.csv').exists():
        with open(folder / 'dropped_cases.csv', newline='') as file:
            release['dropped'] = list(csv.DictReader(file))
    if (folder / 'dataset_info.json').exists():
        release['info'] = json.loads((folder / 'dataset_info.json').read_text())
    if (folder / 'checksums.md5').exists():
        release['checksums'] = [line.split(None, 1)[1].strip() for line in
                                (folder / 'checksums.md5').read_text().splitlines() if line.strip()]
    return release


def absent(name):
    return [(name, "not in the release folder, its numbers are left out")]


# ---------------------------------------------------------------------------------------------------------- formats
def count(n, total):
    """'n of total (share %)'"""
    return f"{n:,} of {total:,} ({100 * n / total:.1f} %)" if total else f"{n:,} of 0"


def number(x):
    """a median with thousands separators: whole numbers without decimals, the others with one"""
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.1f}"


def seconds(t):
    """a solver time in s, the longer ones also in min or h"""
    if t < 120:
        return f"{t:.1f} s"
    return f"{t:,.0f} s = {t / 60:.1f} min" if t < 7200 else f"{t:,.0f} s = {t / 3600:.2f} h"


def classes(pairs):
    """'n size family, ...' of (size, family) pairs, the most frequent first"""
    return ', '.join(f"{n} {size} {family}" for (size, family), n in Counter(pairs).most_common())


def tally(values):
    """'value n, ...', the most frequent first"""
    return ', '.join(f"{value} {n:,}" for value, n in Counter(values).most_common())


def cpu_of(model):
    """the short name of a CPU model of cpu_model: 'E5-2640', 'E5-2670 v2'"""
    found = re.search(r'E5-\d+(?: v\d)?', model or '')
    return found.group(0) if found else str(model)


def variant_of(value):
    """the index of a mesh variant ('0', '1 Mesh.Smoothing = 5;'), 0 when not recorded"""
    return int(str(value).split()[0]) if value not in (None, '') else 0


def threshold(values, quantile):
    """the value a case of an ood set must exceed, as in dataset/splits.py assign_splits"""
    ordered = np.sort(values)
    return ordered[min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)]


# ---------------------------------------------------------------------------------------------------------- sections
def table_2(meta):
    """Table 2: cells, iterations and solver time of every run, median (95th percentile)"""
    lines = []
    for run, label in RUNS:
        cells, iterations, time = (floats(meta[f'{run}.{name}'])
                                   for name in ('cells', 'iterations', 'execution_time_s'))
        lines.append((f"{run} ({label})", f"cells {number(np.median(cells))} ({np.quantile(cells, 0.95):,.0f}); "
                                          f"iterations to 1e-6 {number(np.median(iterations))} "
                                          f"({np.quantile(iterations, 0.95):,.0f}); solver time on one core "
                                          f"{seconds(np.median(time))} ({seconds(np.quantile(time, 0.95))})"))
    return lines


def geometry(meta, dropped, config):
    """Geometry: the blockage ratio and the excluded cases"""
    blockage = floats(meta['blockage_ratio'])
    lines = [("the blockage ratio ranges from ... to ... (median)",
              f"{blockage.min():.4f} to {blockage.max():.4f} ({np.median(blockage):.4f})")]
    reserves = [(s, g) for s, g, i in zip(meta['size'], meta['geometry'], meta['index'])
                if int(i) > config['cases_per_geometry'][s]]
    lines.append(("reserve cases in the dataset, one per excluded case", f"{len(reserves)}: {classes(reserves)}"))
    if dropped is None:
        return lines + absent('dropped_cases.csv')
    if not dropped:
        return lines + [("cases were excluded and replaced", "0")]
    # the residuals and iterations of the base_lu provenance; empty fields where a case had no provenance
    residuals = floats([np.nanmax(floats([row.get(k) or None for k in ('residual_Ux', 'residual_Uy', 'residual_p')]))
                        for row in dropped])
    blockage = floats([row['blockage_ratio'] for row in dropped])
    return lines + [("cases were excluded and replaced",
                     f"{len(dropped)}: {classes((r['size'], r['geometry']) for r in dropped)}"),
                    ("all had a blockage of", f"{blockage.min():.4f} to {blockage.max():.4f}"),
                    ("and largest residuals of ... at the iteration cap",
                     f"{np.nanmin(residuals):.2e} to {np.nanmax(residuals):.2e}"),
                    ("iterations of their 1 mm second-order run (the cap)",
                     tally(f"{float(row['iterations']):,.0f}" if row.get('iterations') else 'not recorded'
                           for row in dropped))]


def meshing(meta):
    """Meshing: the meshes generated again with additional smoothing"""
    remeshed = {run: [case for case, value in zip(meta['case_name'], meta[f'{run}.mesh_variant']) if variant_of(value)]
                for run, _ in RUNS}
    variants = [value for run, _ in RUNS for value in meta[f'{run}.mesh_variant'] if variant_of(value)]
    same = remeshed['base_fo'] == remeshed['base_lu']
    return [("needed for ... meshes (mesh_variant not 0)", ', '.join(f"{len(remeshed[run])} at {label}"
                                                                    for run, label in RUNS)),
            ("the 1 mm runs (each meshed on its own): the same cases",
             f"{'yes' if same else 'no'}: base_lu {', '.join(remeshed['base_lu'])}"),
            ("variants used", tally(variants))]


def flow_solution(meta):
    """Flow solution: residuals, the longest runs, the relaxed 2 mm runs and the starting fields"""
    n = len(meta['case_name'])
    converged = sum(status == 'converged' for run, _ in RUNS for status in meta[f'{run}.status'])
    residual = max(np.nanmax(floats(meta[f'{run}.residual_{q}'])) for run, _ in RUNS for q in ('Ux', 'Uy', 'p'))
    lines = [("runs converged", count(converged, n * len(RUNS))),
             ("the largest final residual in the dataset (u_x, u_y, p)", f"{residual:.6g}")]
    for run, label in RUNS:
        iterations = floats(meta[f'{run}.iterations'])
        k = int(np.nanargmax(iterations))
        lines.append((f"the longest runs needed ... iterations at {label}",
                      f"{iterations[k]:,.0f} ({meta['case_name'][k]})"))
    relaxed = [i for i, template in enumerate(meta['0.25x.templates']) if template == RELAXED]
    iterations = floats(meta['0.25x.iterations'])[relaxed]
    lines += [("2 mm runs started again from their 1 mm solution (relaxed template), started from",
               f"{len(relaxed)}: {tally(meta['0.25x.init_from'][i] for i in relaxed)}"),
              ("these relaxed 2 mm runs converged after ... iterations",
               f"{iterations.min():,.0f} to {iterations.max():,.0f}" if relaxed else "none")]
    for run, label in RUNS:
        starts = [value or 'not recorded' for value in meta[f'{run}.init_from']]
        lines.append((f"{label} runs started from (init_from)", tally(starts)))
    return lines


def attributes(meta, study):
    """Attributes: the outlet reverse flow and the front pressure"""
    reverse = floats(meta['16x.outlet_reverse_flow'])
    lines = [("0.25 mm runs whose outlet reverse flow is not zero", count(int(np.sum(reverse != 0)), len(reverse)))]
    if study is None:
        return lines + absent('mesh_study.parquet')
    kinds = [k for a, k in zip(study['attribute'], study['convergence']) if a == 'front_pressure']
    return lines + [("the front pressure did not converge with the mesh (convergence none)",
                     count(kinds.count('none'), len(kinds))),
                    ("the front pressure, convergence not monotonic", count(sum(k != 'monotonic' for k in kinds),
                                                                           len(kinds)))]


def openfoam_rows(table):
    """the rows of source openfoam of label_uncertainty.parquet, as arrays"""
    keep = np.array(table['source']) == 'openfoam'
    return {name: np.array(values, dtype=object)[keep] for name, values in table.items()}


def label_uncertainty(table):
    """Label uncertainty: the fallback rule, the equal and the zero values"""
    rows = openfoam_rows(table)
    attribute, method = rows['attribute'].astype(str), rows['method'].astype(str)
    u, label = floats(rows['U']), floats(rows['16x'])
    anomalous = method == 'anomalous'

    def share(name):
        return count(int(np.sum(anomalous[attribute == name])), int(np.sum(attribute == name)))

    zero = label == 0
    return [("the fallback rule (fitted order p <= 0) applies to ... of the attribute-case pairs",
             count(int(anomalous.sum()), len(anomalous))),
            ("including ... of the drag values", share('Fx')),
            ("... of the two peak quantities: peak wall shear", share('max_wall_shear')),
            ("... of the two peak quantities: peak vorticity", share('peak_vorticity')),
            ("but ... of the dissipation values", share('dissipation')),
            ("where all four values are equal (U = 0): pairs", f"{int(np.sum(method == 'equal'))}: "
                                                              f"{tally(attribute[method == 'equal'])}"),
            ("pairs whose 0.25 mm value is zero (percentage undefined)",
             f"{int(zero.sum())}: {tally(attribute[zero])}"),
            ("of these, U = 0 / U > 0", f"{int(np.sum(zero & (u == 0)))} / {int(np.sum(zero & (u > 0)))}")]


def splits(meta, config):
    """Splits (Table 5, Fig. 4): the thresholds, tails and composition of the ood sets, the other splits"""
    settings, placement = config['splits'], config['placement']
    rows = [dict(zip(CASE_FIELDS, values)) for values in zip(*(meta[name] for name in CASE_FIELDS))]
    measures = ood_measures(placement)
    tails, lines = {}, []
    for name in settings['ood']:
        values = np.array([measures[name](row) for row in rows])
        limit = threshold(values, settings['ood_quantile'])
        tails[name] = values > limit
        lines.append((f"{name}: above the {100 * settings['ood_quantile']:g}th percentile (threshold), tail",
                      f"{limit:.4g}, {int(tails[name].sum())} cases"))
    for group in list(combinations(settings['ood'], 2)) + [tuple(settings['ood'])]:
        both = np.logical_and.reduce([tails[name] for name in group])
        lines.append((f"cases in the tails of {' and '.join(group)}", f"{int(both.sum())}"))
    in_two = np.sum([tails[name] for name in settings['ood']], axis=0) >= 2
    lines.append(("cases in two tails or more", f"{int(in_two.sum())}"))

    split = np.array(meta['split'])
    sizes, families = np.array(meta['size']), np.array(meta['geometry'])
    for name in settings['ood']:
        mask = split == f'ood_{name}'
        lines.append((f"ood_{name}: cases, by size, by family", f"{int(mask.sum())}; {tally(sizes[mask])}; "
                                                               f"{tally(families[mask])}"))
    blockage, rotation = split == 'ood_blockage', split == 'ood_rotation'
    lines += [("ood_blockage holds mostly larger obstacles", count(int(np.sum(blockage & (sizes == 'larger'))),
                                                                   int(blockage.sum()))),
              ("and triangles (left and right)", count(int(np.sum(blockage & np.isin(families, ['left_triangle',
                                                                                              'right_triangle']))),
                                                      int(blockage.sum()))),
              ("and ... diamonds", f"{int(np.sum(blockage & (families == 'diamond')))}"),
              ("ood_rotation holds mostly smaller and normal obstacles",
               count(int(np.sum(rotation & np.isin(sizes, ['smaller', 'normal']))), int(rotation.sum())))]
    inside = np.isin(split, SPLITS)
    tail = np.logical_or.reduce(list(tails.values()))
    lines += [("cases of the other splits in any tail", f"{int(np.sum(inside & tail))}"),
              ("the other cases are split into training / validation / test",
               f"{int(inside.sum()):,}: " + ' / '.join(f"{int(np.sum(split == s)):,}" for s in SPLITS))]
    again = assign_splits(rows, settings, placement)
    same = sum(again[row['case_name']] == row['split'] for row in rows)
    lines.append(("split column equal to dataset/splits.py on the metadata", count(same, len(rows))))
    return lines


def computation(meta):
    """Computation and provenance: environment, memory, core-hours and the cost of 0.25 against 1 mm"""
    n = len(meta['case_name'])
    digests = [d for run, _ in RUNS for d in meta[f'{run}.container_digest']]
    cpus = [cpu_of(m) for run, _ in RUNS for m in meta[f'{run}.cpu_model']]
    memory = max(np.nanmax(floats(meta[f'{run}.peak_memory_kb'])) for run, _ in RUNS)
    hours = {run: np.nansum(floats(meta[f'{run}.execution_time_s'])) / 3600 for run, _ in RUNS}
    total = sum(hours.values())
    lines = [("runs (cases x runs)", f"{n * len(RUNS):,} ({n:,} x {len(RUNS)})"),
             ("image digests of the runs", f"{len(set(digests))}: {tally(digests)}"),
             ("processors (runs)", tally(cpus)),
             # peak_memory_kb is the maximum resident set size of /usr/bin/time -v, in KiB
             ("peak solver memory was at most", f"{memory:,.0f} KiB = {memory / 1024:,.0f} MiB = "
                                                f"{memory / 1024 ** 2:.2f} GiB ({memory * 1024 / 1e9:.2f} GB)"),
             ("the runs took ... solver core-hours", f"{total:,.1f} (" + ', '.join(f"{run} {hours[run]:,.1f}"
                                                                               for run, _ in RUNS) + ')'),
             ("... of them at 0.25 mm", f"{100 * hours['16x'] / total:.1f} %")]

    def median(run, name):
        return np.median(floats(meta[f'{run}.{name}']))

    def per_iteration(run, cpu=None):
        """median ms per iteration and 1000 cells, of the runs on one CPU model (all runs: None)"""
        time, iterations, cells = (floats(meta[f'{run}.{name}'])
                                   for name in ('execution_time_s', 'iterations', 'cells'))
        mask =np.array([cpu is None or cpu_of(m) == cpu for m in meta[f'{run}.cpu_model']])
        return np.median((time / iterations / (cells / 1000) * 1e3)[mask]) if mask.any() else np.nan

    ratio = median('16x', 'execution_time_s') / median('base_lu', 'execution_time_s')
    per_case = np.median(floats(meta['16x.execution_time_s']) / floats(meta['base_lu.execution_time_s']))
    cells = median('16x', 'cells') / median('base_lu', 'cells')
    iterations = median('16x', 'iterations') / median('base_lu', 'iterations')
    lines += [("a 0.25 mm run costs about ... times as much as a 1 mm run (ratio of the median solver times)",
               f"{ratio:.1f} (median of the per-case ratios {per_case:.1f})"),
              ("it has ... times the cells (ratio of the medians)", f"{cells:.2f}"),
              ("needs ... times the iterations (ratio of the medians)", f"{iterations:.2f}")]
    for cpu in sorted(set(cpus)) + [None]:
        fine, coarse = per_iteration('16x', cpu), per_iteration('base_lu', cpu)
        lines.append((f"and ... times the time per iteration and cell, {cpu or 'all runs'}",
                      f"{fine / coarse:.2f} ({coarse:.2f} -> {fine:.2f} ms per iteration and 1000 cells)"))
    product = cells * iterations * per_iteration('16x', REFERENCE_CPU) / per_iteration('base_lu', REFERENCE_CPU)
    lines.append((f"product cells x iterations x time per iteration and cell ({REFERENCE_CPU})", f"{product:.1f}"))
    return lines


def release_files(folder, listed):
    """the files of the release folder: number and size (the arrays per run folder), against checksums.md5"""
    folder = Path(folder)
    files = sorted(p for p in folder.rglob('*') if p.is_file())
    size = {}
    for path in files:
        part = path.relative_to(folder).parts
        key = part[0] if len(part) > 1 else 'files at the top level'
        size[key] = size.get(key, 0) + path.stat().st_size
    lines = [("files in the release folder, in all", f"{len(files)}, {sum(size.values()) / 1e9:.1f} GB"),
             ("size per folder, GB", ', '.join(f"{key} {value / 1e9:.1f}" for key, value in sorted(size.items())))]
    if listed is None:
        return lines + absent('checksums.md5')
    present = {p.relative_to(folder).as_posix() for p in files}
    missing = [name for name in listed if name not in present]
    unlisted = sorted(present - set(listed) - {'checksums.md5'})
    examples = f": {(missing + unlisted)[:5]}" if missing or unlisted else ''
    return lines + [("data files listed in checksums.md5 (and the checksum file itself)",
                     f"{len(listed)} (+1 = {len(listed) + 1})"),
                    ("listed but not in the folder / in the folder but not listed",
                     f"{len(missing)} / {len(unlisted)}{examples}")]


def data_records(folder, release):
    """Data Records: rows and columns of the tables, dataset_info.json and the files"""
    meta, study, table, info = release['metadata'], release['mesh_study'], release['uncertainty'], release['info']
    lines = []
    if meta is not None:
        per_class = Counter(zip(meta['size'], meta['geometry'])).values()
        lines += [("cases (rows of metadata.parquet)", f"{len(meta['case_name']):,}"),
                  ("cases per size and family",
                   f"{min(per_class)} to {max(per_class)} in the {len(per_class)} classes"),
                  ("metadata.parquet has one row per case and ... columns", f"{release['metadata_columns']}")]
    if study is not None:
        names = set(study['attribute'])
        reverse = 'yes' if 'outlet_reverse_flow' in names else 'no'
        lines.append(("mesh_study.parquet: rows, attributes (outlet_reverse_flow among them)",
                      f"{len(study['case_name']):,}, {len(names)} ({reverse})"))
    else:
        lines += absent('mesh_study.parquet')
    if table is not None:
        sources = np.array(table['source'])
        lines.append(("label_uncertainty.parquet: rows (attributes) per source",
                      ', '.join(f"{s} {int(np.sum(sources == s)):,} "
                                f"({len({a for a, t in zip(table['attribute'], sources) if t == s})})"
                                for s in sorted(set(sources)))))
    else:
        lines += absent('label_uncertainty.parquet')
    if release['dropped'] is not None:
        lines.append(("dropped_cases.csv lists the parameters and final residuals of the ... excluded cases",
                      f"{len(release['dropped'])}"))
    else:
        lines += absent('dropped_cases.csv')
    if info is not None:
        runs, counts = info.get('runs', {}), info.get('splits', {})
        lines += [("dataset_info.json: cases, dropped cases, reserve cases",
                   f"{info.get('cases')}, {info.get('dropped_cases')}, {info.get('reserve_cases')}"),
                  ("dataset_info.json: counts per split", ', '.join(f"{k} {v:,}" for k, v in counts.items())),
                  ("dataset_info.json: runs and grids",
                   ', '.join(f"{run} {' x '.join(map(str, value.get('grid', [])))}" for run, value in runs.items())),
                  ("dataset_info.json: code revisions, image digests",
                   f"{len(info.get('code_revisions', []))}, {len(info.get('container_digests', []))}"),
                  ("HDF5 files (runs x splits)", f"{len(runs) * len(counts)}")]
    else:
        lines += absent('dataset_info.json')
    return lines + release_files(folder, release['checksums'])


def discretisation(meta, study, table):
    """Technical Validation, discretisation error and label uncertainty"""
    lines = []
    if study is not None:
        attribute, kinds = np.array(study['attribute']), np.array(study['convergence'])
        for name, label in UNDEFINED_GCI:
            mask = attribute == name
            lines.append((f"the three-level GCI is undefined (not monotonic) for the {label}",
                          count(int(np.sum(mask & (kinds != 'monotonic'))), int(mask.sum()))))
        family = dict(zip(meta['case_name'], meta['geometry']))
        order, cases = floats(study['order']), np.array(study['case_name'])
        for name, label in (('Fx_viscous', 'friction drag'), ('Fx_pressure', 'pressure drag')):
            mask = (attribute == name) & (kinds == 'monotonic')
            if not mask.any():
                lines.append((f"observed order of the {label}", "no monotonic cases"))
                continue
            families = np.array([family.get(case) for case in cases[mask]])
            medians = {f: np.nanmedian(order[mask][families == f]) for f in sorted(set(families))}
            lines.append((f"observed order of the {label} (median of the monotonic cases; per family)",
                          f"{np.nanmedian(order[mask]):.2f}; {min(medians.values()):.2f} to "
                          f"{max(medians.values()):.2f} (" + ', '.join(f"{f} {m:.2f}" for f, m in medians.items())
                          + ')'))
        mask = attribute == 'front_pressure'
        lines.append(("the front pressure did not converge (convergence none)",
                      count(int(np.sum(mask & (kinds == 'none'))), int(mask.sum()))))
    else:
        lines += absent('mesh_study.parquet')
    if table is not None:
        rows = openfoam_rows(table)
        mask = rows['attribute'].astype(str) == 'dissipation'
        u, simple = np.nanmedian(floats(rows['U_%'][mask])), np.nanmedian(floats(rows['U1_%'][mask]))
        lines.append(("dissipation: median U against the simple estimate 3|phi_0.25 - phi_0.5|, %",
                      f"{u:.2f} against {simple:.2f}"))
    else:
        lines += absent('label_uncertainty.parquet')
    first, label = floats(meta['base_fo.recirculation_length_mm']), floats(meta['16x.recirculation_length_mm'])
    lines.append(("the first-order 1 mm recirculation length is shorter than the label",
                  count(int(np.sum(first < label)), len(label))))
    return lines


def paper_numbers(folder, config=None):
    """
    :param config: config/dataset.yaml (its splits, placement and cases_per_geometry), read when not given
    :return: [(section of the paper, [(label, number)])], in the order of the paper
    """
    if config is None:
        from utils import load_config
        config = load_config()
    release = load(folder)
    meta, study, table = release['metadata'], release['mesh_study'], release['uncertainty']
    if meta is None:
        return [('Release', absent('metadata.parquet'))]
    return [("Table 2: the five runs of every case (median over the cases, 95th percentile in brackets)",
             table_2(meta)),
            ("Methods, Geometry", geometry(meta, release['dropped'], config)),
            ("Methods, Meshing", meshing(meta)),
            ("Methods, Flow solution", flow_solution(meta)),
            ("Methods, Attributes", attributes(meta, study)),
            ("Methods, Label uncertainty",
             label_uncertainty(table) if table is not None else absent('label_uncertainty.parquet')),
            ("Methods, Splits (Table 5, Fig. 4)", splits(meta, config)),
            ("Methods, Computation and provenance", computation(meta)),
            ("Data Records", data_records(folder, release)),
            ("Technical Validation, Discretisation error and label uncertainty", discretisation(meta, study, table))]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(f"numbers of the dataset paper from the release files in {sys.argv[1]}")
    for section, lines in paper_numbers(sys.argv[1]):
        print(f"\n{section}")
        for label, value in lines:
            print(f"  {label}: {value}")
