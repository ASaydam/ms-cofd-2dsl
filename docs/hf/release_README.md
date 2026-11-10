---
license: cc-by-4.0
pretty_name: MS-COFD-2DSL
size_categories:
- 1K<n<10K
tags:
- cfd
- openfoam
- fluid-dynamics
- multi-fidelity
- laminar-flow
- simulation
- super-resolution
- surrogate-modeling
configs:
- config_name: metadata
  data_files: metadata.parquet
  default: true
- config_name: label_uncertainty
  data_files: label_uncertainty.parquet
- config_name: mesh_study
  data_files: mesh_study.parquet
- config_name: surface
  data_files: surface.parquet
- config_name: dropped_cases
  data_files: dropped_cases.csv
---

# MS-COFD-2DSL: Multi-Scale Central Object Flow Dataset, 2D Steady Laminar

MS-COFD-2DSL holds 4,500 two-dimensional, steady, laminar channel flows past one obstacle. Each flow was solved to
convergence with OpenFOAM at five fidelities, from a 2 mm to a 0.25 mm mesh. This repository holds the fields of
every run on regular grids, a table of the settings and attributes of every case and run, the mesh study of every
attribute and the numerical uncertainty of every label. The complete OpenFOAM case of every run is a separate
dataset: [asaydam/ms-cofd-2dsl-raw](https://huggingface.co/datasets/asaydam/ms-cofd-2dsl-raw).

The data card [`DATACARD.md`](DATACARD.md) in this repository describes the channels, grids, attributes, caveats,
provenance and validation in full. Read its caveats before you use an attribute as a target.

## Contents

```
<run>/<split>.h5            the fields of the cases of a split, on the grid of the run
metadata.parquet            one row per case (259 columns): parameters, geometry, split, and per run its settings,
metadata.csv                iterations, residuals, cost and attributes
mesh_study.parquet          values of every level, order, extrapolation and grid convergence index per case and
                            attribute
surface.parquet             pressure and wall shear on every obstacle face, every case and run
label_uncertainty.parquet   numerical uncertainty of the 19 attributes of every case
dropped_cases.csv           the 18 cases whose 1 mm second-order run did not converge, replaced by reserve geometries
dataset_info.json           counts, runs, grids, normalisation statistics, revisions
DATACARD.md                 the data card
checksums.md5               MD5 of every other file
```

| Run | Mesh | Convection | Array grid (nx × ny) | Size |
|---|---|---|---|---|
| `0.25x` | 2 mm | second order | 180 × 60 | 1.2 GB |
| `base_fo` | 1 mm | first order | 360 × 120 | 4.7 GB |
| `base_lu` | 1 mm | second order | 360 × 120 | 4.7 GB |
| `4x` | 0.5 mm | second order | 720 × 240 | 18.7 GB |
| `16x` | 0.25 mm | second order | 1440 × 480 | 74.7 GB |

The tables add about 0.2 GB, about 104 GB in all. The 0.25 mm run is the label. Every run was iterated until the
initial residuals of U and p were below 10⁻⁶.

| Split | Cases | Content |
|---|---|---|
| `train` | 3,085 | 80 % of the in-distribution cases of every size and family |
| `val` | 385 | 10 % |
| `test` | 386 | 10 % |
| `ood_blockage` | 224 | blockage ratio above the 95th percentile |
| `ood_rotation` | 216 | absolute rotation above the 95th percentile |
| `ood_position` | 204 | position shift furthest from the middle of its range |

The three `ood_` splits are out-of-distribution test sets, held out of train, val and test.

## Download

The whole repository is about 104 GB. A run or a split can be downloaded alone:

```python
from huggingface_hub import snapshot_download
snapshot_download("asaydam/ms-cofd-2dsl", repo_type="dataset", local_dir="ms-cofd-2dsl",
                  allow_patterns=["*.parquet", "*.csv", "*.json", "*.md", "*.md5", "base_lu/*", "16x/test.h5"])
```

```bash
cd ms-cofd-2dsl
md5sum -c --ignore-missing checksums.md5      # checks the downloaded files
```

## Reading the data

```python
import h5py
import pandas as pd

metadata = pd.read_parquet("ms-cofd-2dsl/metadata.parquet")
test = metadata[metadata["split"] == "test"]           # in the order of the rows of every <run>/test.h5
with h5py.File("ms-cofd-2dsl/base_lu/test.h5") as f:
    assert list(f["case_name"].asstr()[:]) == list(test["case_name"])
    print(dict(f.attrs))                               # channel names, grid, mean and std over the train split
    sample = f["fields"][0]                            # (6, 360, 120) float32, one sample per chunk
drag = test["16x.Fx"]                                  # attributes of a run: <run>.<attribute>
```

- **Channels:** 0 `u_x` and 1 `u_y` (m/s), 2 `p` (kinematic pressure, m²/s², 0 at the outlet), 3 `wall_distance`
  (to the closer channel wall, in grid points), 4 `cell_type` (1 wall or obstacle, 2 inlet, 3 outlet, 4 fluid),
  5 `obstacle_sdf` (signed distance to the obstacle outline in grid points, negative inside).
- **Grid:** point (i, j) lies at x = i · 360 / (nx − 1) mm and y = j · 120 / (ny − 1) mm, with the inlet at i = 0.
- **Rows:** row i is the same case in all five runs of a split.
- **Attributes** (drag, lift, recirculation, wake, dissipation and more) are computed on the simulation mesh by
  OpenFOAM. Their uncertainty is in `label_uncertainty.parquet`.

## Code

- The generation pipeline, [ms-cofd-2dsl](https://github.com/ASaydam/ms-cofd-2dsl) (tag `data-v1.0.0`), makes every
  file of this repository from the archived cases.
- The benchmark, [ms-cofd-2dsl-bench](https://github.com/ASaydam/ms-cofd-2dsl-bench), defines tasks and metrics, and
  holds readers, baselines and reference networks.

## Citation and licence

A dataset paper is in preparation. Until it is out, please cite the dataset:

```bibtex
@misc{saydam2026mscofd2dsl,
  author    = {Alper Saydam and Barış Akgün},
  title     = {ms-cofd-2dsl (Revision b312def)},
  year      = {2026},
  publisher = {Hugging Face},
  doi       = {10.57967/hf/11554},
  url       = {https://huggingface.co/datasets/asaydam/ms-cofd-2dsl}
}
```

The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): you may share and adapt it
for any purpose, including commercially, if you give credit by citing it and indicate any changes.
