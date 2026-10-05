"""
Initialization check (docs/validation.md): the same meshes solved from the converged 1 mm solution (mapFields, as in
the production) and from rest (runs <run>_scratch of a validation data root with its own dataset.yaml):

    python validation/compare_init.py <data root> [run ...]        (default runs: 4x 16x)

For every case that has both runs: the iterations of both, the difference of the arrays (u_x, u_y, p: RMS of the
difference over the RMS of the mapped run, fluid points) and of every attribute (|scratch - mapped| / |mapped|, lift
relative to the drag).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.results import cases_of, load_attributes, load_sample  # noqa: E402

CHANNELS = {'u_x': 0, 'u_y': 1, 'p': 2}
SKIP = ('time', 'cells', 'faces')


def compare(root, run):
    scratch = f"{run}_scratch"
    cases = cases_of(root, run, scratch)
    print(f"{run} from the 1 mm solution against {scratch} from rest: {len(cases)} cases")
    print(f"  {'case':28s} {'iterations':>13s} " + ' '.join(f"{c + ' %':>9s}" for c in CHANNELS) + f" {'attribute max %':>16s}")
    worst = {}
    for case in cases:
        mapped, fresh = load_sample(root, run, case), load_sample(root, scratch, case)
        fluid = mapped[4] != 1
        field = []
        for name, channel in CHANNELS.items():
            a, b = fresh[channel][fluid], mapped[channel][fluid]
            field.append(100 * np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))
        attributes_mapped, attributes_fresh = load_attributes(root, run, case), load_attributes(root, scratch, case)
        differences = {}
        for name, value in attributes_mapped.items():
            if name in SKIP or not isinstance(value, (int, float)) or name not in attributes_fresh:
                continue
            scale = abs(attributes_mapped['Fx']) if name.startswith('Fy') else abs(value)
            if scale > 1e-30:
                differences[name] = 100 * abs(attributes_fresh[name] - value) / scale
                worst[name] = max(worst.get(name, 0.0), differences[name])
        iterations = [json.loads((Path(root) / 'results' / r / f"{case}_provenance.json").read_text())['iterations']
                      for r in (run, scratch)]
        top = max(differences, key=differences.get) if differences else '-'
        print(f"  {case:28s} {iterations[0]:>6s}/{iterations[1]:>6s} " + ' '.join(f"{v:9.4f}" for v in field) +
              f" {differences.get(top, float('nan')):9.4f} ({top})")
    if worst:
        print("  largest difference over the cases, per attribute (%): " +
              ', '.join(f"{name} {value:.4f}" for name, value in sorted(worst.items(), key=lambda x: -x[1])[:8]))


if __name__ == "__main__":
    root = sys.argv[1]
    for run in sys.argv[2:] or ['4x', '16x']:
        compare(root, run)
