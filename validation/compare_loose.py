"""
Loose-criterion audit (docs/validation.md, validation/select_loose_audit.py): what the stopping criterion of common
practice (residualControl p 1e-2, U 1e-3) does to the labels and to the coarse-to-fine gap, against the converged
runs of the production (1e-6). Both data roots are only read.

    python validation/compare_loose.py <audit root> <production root> <output folder>

Runs of the audit root, each compared with the converged run of the same mesh in the production root:

    16x_loose               0.25 mm, loose, from the converged 1 mm solution (as the production starts)   -> 16x
    16x_loose_scratch       0.25 mm, loose, from rest                                                      -> 16x
    base_lu_loose_scratch   1 mm, loose, from rest                                                         -> base_lu

For every case and loose run (loose_runs.csv):

    iterations, iterations_converged     at the loose stop and of the converged run
    rms_<channel>_of_scale_%             RMS of (loose - converged) over the fluid points, relative to the inlet
                                         velocity (u_x, u_y) or the dynamic pressure 0.5 U^2 (p)
    rms_<channel>_of_field_%             the same relative to the RMS of the converged field
    <attribute>                          difference of every attribute from the converged one, on the scales of
                                         validation/audit_dataset.py level_difference (relative, lift and the drag
                                         parts relative to the drag, ...)

The coarse-to-fine gap (loose_gap.csv, tools/upsample_mse.py compare: every interpolation method and channel) for

    converged      base_lu -> 16x                                    the gap of the dataset
    loose_mapped   base_lu -> 16x_loose                              labels stopped early, started as in production
    loose_scratch  base_lu_loose_scratch -> 16x_loose_scratch        both stopped early from rest (common practice)

loose_summary.json: per group (test, ood_blockage) and run the weighted median and 90th percentile of every number,
each case weighted with 1 / its inclusion probability (selection.csv), so the test numbers stand for the whole test
split; the ood_blockage cases are reported on their own. The table printed at the end is the one of the paper.

Needs numpy and scipy (the container).
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import load_config, write_csv  # noqa: E402
from dataset.results import load_attributes, load_sample  # noqa: E402
from hpc import ledger  # noqa: E402
from tools import upsample_mse  # noqa: E402
from validation.audit_dataset import LEVEL_ATTRIBUTES, as_float, inlet_velocity, level_difference  # noqa: E402

LOOSE = {'16x_loose': '16x', '16x_loose_scratch': '16x', 'base_lu_loose_scratch': 'base_lu'}
# coarse-to-fine pairs: (coarse root, coarse run), (fine root, fine run)
GAPS = {'converged': (('production', 'base_lu'), ('production', '16x')),
        'loose_mapped': (('production', 'base_lu'), ('audit', '16x_loose')),
        'loose_scratch': (('audit', 'base_lu_loose_scratch'), ('audit', '16x_loose_scratch'))}
CHANNELS = {'u_x': 0, 'u_y': 1, 'p': 2}
CELL_TYPE = 4
# the attributes of the summary table
TABLE = ['Fx', 'Fx_viscous', 'recirculation_length_mm', 'base_pressure', 'wake_min_u_x']


def iterations_of(root, run, case):
    path = Path(root) / 'results' / run / f"{case}_provenance.json"
    return as_float(json.loads(path.read_text()).get('iterations')) if path.exists() else None


def field_change(loose, converged, inlet):
    """RMS of loose - converged over the fluid points of the converged sample, per channel"""
    if loose.shape != converged.shape:
        raise ValueError(f"shapes {loose.shape} and {converged.shape}")
    fluid = converged[CELL_TYPE] != 1
    row = {}
    for name, channel in CHANNELS.items():
        difference = np.sqrt(np.mean((loose[channel][fluid] - converged[channel][fluid]) ** 2))
        scale = inlet if name != 'p' else 0.5 * inlet ** 2
        field = float(np.sqrt(np.mean(converged[channel][fluid] ** 2)))
        row[f"rms_{name}_of_scale_%"] = 100 * float(difference) / scale
        row[f"rms_{name}_of_field_%"] = 100 * float(difference) / field if field > 0 else None
    return row


def attribute_changes(loose, converged, inlet, height):
    """differences of the attributes, see validation/audit_dataset.py level_difference"""
    row = {}
    for name in LEVEL_ATTRIBUTES:
        value, reference = as_float(loose.get(name)), as_float(converged.get(name))
        if value is not None and reference is not None:
            difference = level_difference(name, value, reference, as_float(converged.get('Fx')) or 0, inlet, height)
            row[name] = None if difference is None else 100 * difference
    return row


def weighted_quantile(values, weights, q):
    """quantile q of values whose points carry the given weights (midpoint rule, linear between the points)"""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    positions = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return float(np.interp(q, positions, values))


def summarise(rows, fields, absolute=()):
    """weighted median and 90th percentile of every field (of |field| for those in absolute)"""
    summary = {}
    for field in fields:
        pairs = [(r[field], r['weight']) for r in rows if isinstance(r.get(field), (int, float)) and
                 not math.isnan(r[field])]
        if not pairs:
            continue
        values, weights = zip(*pairs)
        magnitudes = np.abs(values)
        summary[field] = {'n': len(values), 'median': weighted_quantile(values, weights, 0.5),
                          'median_abs': weighted_quantile(magnitudes, weights, 0.5),
                          'p90_abs': weighted_quantile(magnitudes, weights, 0.9)}
        if field in absolute:
            summary[field].pop('median')
    return summary


def read_selection(audit_root):
    with open(Path(audit_root) / 'selection.csv', newline='') as file:
        return list(csv.DictReader(file))


def main():
    if len(sys.argv) != 4:
        sys.exit("usage: python validation/compare_loose.py <audit root> <production root> <output folder>")
    roots = {'audit': Path(sys.argv[1]), 'production': Path(sys.argv[2])}
    output = Path(sys.argv[3])
    if output.resolve() in (roots['audit'].resolve(), roots['production'].resolve()):
        sys.exit("the output folder must not be a data root")
    output.mkdir(parents=True, exist_ok=True)

    inlet, domain = inlet_velocity(), load_config()['domain']
    manifest = {row['case_name']: row for row in ledger.read_manifest(roots['production'] / 'geometry' / 'manifest.csv')}
    selection = read_selection(roots['audit'])
    print(f"{len(selection)} cases: " + ', '.join(f"{g} {sum(1 for r in selection if r['group'] == g)}"
                                                for g in sorted({r['group'] for r in selection})))

    run_rows, gap_rows = [], []
    for row in selection:
        case = row['case']
        common = {'case_name': case, 'group': row['group'], 'size': row['size'], 'family': row['family'],
                  'weight': 1 / float(row['inclusion_probability'])}
        height = float(manifest[case]['bbox_height'])
        for run, reference in LOOSE.items():
            loose = load_sample(roots['audit'], run, case)
            converged = load_sample(roots['production'], reference, case)
            if not np.array_equal(loose[CELL_TYPE], converged[CELL_TYPE]):
                raise ValueError(f"{case}: the obstacle of {run} differs from that of the production {reference}")
            result = dict(common, run=run, iterations=iterations_of(roots['audit'], run, case),
                          iterations_converged=iterations_of(roots['production'], reference, case))
            result.update(field_change(loose, converged, inlet))
            result.update(attribute_changes(load_attributes(roots['audit'], run, case),
                                            load_attributes(roots['production'], reference, case), inlet, height))
            run_rows.append(result)
        for pair, ((coarse_root, coarse_run), (fine_root, fine_run)) in GAPS.items():
            coarse = load_sample(roots[coarse_root], coarse_run, case)
            fine = load_sample(roots[fine_root], fine_run, case)
            for gap in upsample_mse.compare(coarse, fine, domain['length']):
                gap_rows.append(dict(common, pair=pair, **gap))
        print(f"  {case}", flush=True)

    run_fields = list(dict.fromkeys(k for r in run_rows for k in r))
    write_csv(run_rows, output / 'loose_runs.csv', run_fields)
    write_csv(gap_rows, output / 'loose_gap.csv', list(gap_rows[0]))

    number_fields = [f for f in run_fields if f not in ('case_name', 'group', 'size', 'family', 'weight', 'run')]
    summary = {'inlet_velocity': inlet, 'cases': {g: sum(1 for r in selection if r['group'] == g)
                                                  for g in {r['group'] for r in selection}}, 'runs': {}, 'gap': {}}
    for group in sorted({r['group'] for r in selection}):
        for run in LOOSE:
            rows = [r for r in run_rows if r['group'] == group and r['run'] == run]
            summary['runs'].setdefault(group, {})[run] = summarise(
                rows, number_fields, absolute=[f for f in number_fields if f.startswith('rms_') or
                                               f.startswith('iterations')])
        for pair in GAPS:
            for method in upsample_mse.METHODS:
                for channel in CHANNELS:
                    rows = [dict(r, value=r['rel_rmse_%']) for r in gap_rows if r['group'] == group and
                            r['pair'] == pair and r['method'] == method and r['channel'] == channel]
                    summary['gap'].setdefault(group, {}).setdefault(pair, {}).setdefault(method, {})[channel] = \
                        summarise(rows, ['value'], absolute=['value']).get('value')
    (output / 'loose_summary.json').write_text(json.dumps(summary, indent=1) + '\n')

    for group in sorted(summary['runs']):
        print(f"\n{group} (weighted, n = {summary['cases'][group]}): median (90th percentile of |.|)")
        print(f"  {'run':24s} {'iterations':>18s} {'rms u_x % U':>12s} " + ' '.join(f"{name[:14]:>16s}" for name in TABLE))
        for run in LOOSE:
            s = summary['runs'][group][run]
            print(f"  {run:24s} {s['iterations']['median_abs']:7.0f} / {s['iterations_converged']['median_abs']:7.0f} "
                  f"{s['rms_u_x_of_scale_%']['median_abs']:12.2f} " +
                  ' '.join(f"{s[name]['median']:+7.2f} ({s[name]['p90_abs']:5.1f})" if name in s else f"{'-':>16s}"
                           for name in TABLE))
        print("  coarse-to-fine gap, relative RMS of u_x (%), bicubic_filled: " +
              ', '.join(f"{pair} {summary['gap'][group][pair]['bicubic_filled']['u_x']['median_abs']:.2f}"
                        for pair in GAPS))
    print(f"\nwritten to {output}")


if __name__ == "__main__":
    main()
