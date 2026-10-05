"""
Summary of the runs of a data root (the production or a pilot), from the per-run files written by hpc/run_task.py:

    python -m tools.run_summary [data root]          (default: MSCOFD_DATA_ROOT)

1. per run: finished tasks and their status, iterations, solver clock time, peak memory, mesh variants, hosts; the
   numbers behind the --time and --mem of config/slurm.yaml
2. results/openfoam_attributes.csv: one row per case and run, in the format of tools/openfoam_attributes.py collect
   (resolution: "base" for the base_lu run, otherwise the run name), the input of tools/mesh_study.py:
       python -m tools.mesh_study <data root>/results/openfoam_attributes.csv --extra 0.25x base_fo
3. reverse flow at the outlet (attribute outlet_reverse_flow) on the finest run of every case, with the blockage
   ratio and the distance from the obstacle to the outlet of the manifest

Standard library only.
"""

import csv
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.openfoam_attributes import FIELDS  # noqa: E402

RUNS = ['0.25x', 'base_fo', 'base_lu', '4x', '16x']
RESOLUTION = {'base_lu': 'base'}     # name of the run in the attribute table (tools/mesh_study.py)
REVERSE_FLOW = 1e-3                  # outlet_reverse_flow above which the outlet condition matters


def read_runs(root):
    """{run: {case: (provenance, attributes)}} of the finished runs"""
    runs = {}
    for run in RUNS:
        folder = Path(root) / 'results' / run
        for path in sorted(folder.glob('*_provenance.json')):
            case = path.name[:-len('_provenance.json')]
            attributes = folder / f"{case}_attributes.json"
            runs.setdefault(run, {})[case] = (json.loads(path.read_text()),
                                              json.loads(attributes.read_text()) if attributes.exists() else {})
    return runs


def quantiles(values):
    values = sorted(values)
    if not values:
        return 'n/a'
    return (f"median {statistics.median(values):.4g}, p90 {values[int(0.9 * (len(values) - 1))]:.4g}, "
            f"max {values[-1]:.4g}")


def attribute_rows(runs):
    """rows of the attribute table (tools/openfoam_attributes.py format)"""
    rows = []
    for run, cases in runs.items():
        for case, (provenance, attributes) in sorted(cases.items()):
            row = {'case_name': case, 'resolution': RESOLUTION.get(run, run),
                   'status': 'ok' if attributes and provenance.get('status') == 'converged' else 'not converged'}
            row.update({key: value for key, value in attributes.items() if key in FIELDS})
            row['cells'] = attributes.get('cells', provenance.get('cells', ''))
            rows.append(row)
    return rows


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ['MSCOFD_DATA_ROOT'])
    runs = read_runs(root)

    print(f"1. runs in {root}")
    for run, cases in runs.items():
        records = [provenance for provenance, _ in cases.values()]
        statuses = {}
        for record in records:
            statuses[record['status']] = statuses.get(record['status'], 0) + 1
        variants = {}
        for record in records:
            variant = record.get('mesh_variant') or '0'
            variants[variant] = variants.get(variant, 0) + 1
        hosts = sorted({record.get('host', '') for record in records})
        print(f"  {run:8s} {len(records):5d} finished {statuses}")
        print(f"           iterations   {quantiles([int(r['iterations']) for r in records if r.get('iterations')])}")
        print(f"           solver hours {quantiles([float(r['clock_time_s']) / 3600 for r in records if r.get('clock_time_s')])}")
        print(f"           memory MB    {quantiles([r['peak_memory_kb'] / 1024 for r in records if r.get('peak_memory_kb')])}")
        print(f"           mesh variants {variants}; {len(hosts)} hosts")

    rows = attribute_rows(runs)
    output = root / 'results' / 'openfoam_attributes.csv'
    with open(output, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n2. attribute table of {len(rows)} runs: {output}")

    manifest_path = root / 'geometry' / 'manifest.csv'
    manifest = {r['case_name']: r for r in csv.DictReader(open(manifest_path))} if manifest_path.exists() else {}
    finest = {}
    for run in RUNS:
        for case, (_, attributes) in runs.get(run, {}).items():
            if 'outlet_reverse_flow' in attributes:
                finest[case] = (run, attributes)
    flagged = sorted(((a['outlet_reverse_flow'], case, run, a) for case, (run, a) in finest.items()
                      if a['outlet_reverse_flow'] > 0), reverse=True)
    print(f"\n3. reverse flow at the outlet, finest finished run of {len(finest)} cases: "
          f"{sum(f[0] > REVERSE_FLOW for f in flagged)} above {REVERSE_FLOW:g}, {len(flagged)} above 0")
    for value, case, run, a in flagged:
        m = manifest.get(case, {})
        print(f"  {case:28s} {run:7s} outlet_reverse_flow {value:.2e}  blockage {float(m.get('blockage_ratio', 'nan')):.2f}"
              f"  recirculation {a.get('recirculation_length_mm', float('nan')):6.1f} mm of {float(m.get('gap_outlet', 'nan')):6.1f} mm")


if __name__ == "__main__":
    main()
