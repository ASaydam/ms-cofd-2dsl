"""
Reads the per-case results of the production (written by hpc/run_task.py) from the data root:

    results/<run>/<case>.npz               sample (6, nx, ny) float32 and the exported time
    results/<run>/<case>_attributes.json   attributes computed with OpenFOAM (tools/openfoam_attributes.py)
    results/<run>/<case>_provenance.json   status, iterations, timings, versions, ...

The analysis tools (tools/upsample_mse.py, tools/flow_attributes.py) use these readers, so they work during the
production as well as on the final dataset.
"""

import json
from pathlib import Path

import numpy as np


def results_folder(root):
    return Path(root) / 'results'


def cases_of(root, *runs):
    """names of the cases that have results for every given run, sorted"""
    names = None
    for run in runs:
        found = {path.stem for path in (results_folder(root) / run).glob('*.npz')}
        names = found if names is None else names & found
    return sorted(names or [])


def load_sample(root, run, case):
    """the 6 channel sample of a run of a case, as float64"""
    with np.load(results_folder(root) / run / f"{case}.npz") as data:
        return data['sample'].astype(float)


def load_attributes(root, run, case):
    """the OpenFOAM attributes of a run of a case, an empty dictionary when there are none"""
    path = results_folder(root) / run / f"{case}_attributes.json"
    return json.loads(path.read_text()) if path.exists() else {}


def pairs(root, coarse_run='base_lu', fine_run='16x', cases=None):
    """(case, coarse sample, fine sample) of every case that has both runs"""
    for case in cases or cases_of(root, coarse_run, fine_run):
        yield case, load_sample(root, coarse_run, case), load_sample(root, fine_run, case)
