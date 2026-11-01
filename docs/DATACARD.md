# MS-COFD-2DSL v1.0.0: data card

**Multi-Scale Central Object Flow Dataset, 2D Steady Laminar.** 4,500 two-dimensional, steady, laminar channel flows
past one obstacle, each solved to convergence at five fidelities with OpenFOAM.

| | |
|---|---|
| Version | 1.0.0. Arrays built at pipeline commit f71b088; metadata rebuilt at 0b6c60a (2026-10-19), see Provenance |
| Licence | CC BY 4.0 |
| Citation, links | the repository page of this dataset gives the citation, the DOI, the code repositories and the location of the archived cases |
| Code | the generation pipeline `ms-cofd-2dsl`, tag `data-v1.0.0`; the benchmark `ms-cofd-2dsl-bench` (tasks, metrics, label uncertainty) |
| Authors | Alper Saydam and Barış Akgün, KUIS AI Center, Koç University, Istanbul |
| Integrity | `md5sum -c checksums.md5` in this folder checks every file |

## Contents

```
0.25x/ base_fo/ base_lu/ 4x/ 16x/   each with train.h5, val.h5, test.h5,
                                    ood_blockage.h5, ood_rotation.h5, ood_position.h5
metadata.parquet  metadata.csv      one row per case (259 columns)
mesh_study.parquet                  values of every level, order, extrapolation, grid convergence index per case
                                    and attribute
surface.parquet                     pressure and wall shear on every obstacle face, every case and run
label_uncertainty.parquet           uncertainty of the 19 attributes of every case (see Reading the data)
dropped_cases.csv                   the 18 cases whose 1 mm second-order run did not converge, replaced from
                                    reserve geometries
dataset_info.json                   counts, runs, grids, normalisation statistics, revisions
DATACARD.md                         this file
checksums.md5                       MD5 of every other file
```

**The runs of every case:**

| Run | Mesh | Convection | Started from | Array grid (nx × ny) |
|---|---|---|---|---|
| `0.25x` | 2 mm | second order | rest (27 runs: `base_lu`, see Caveats) | 180 × 60 |
| `base_fo` | 1 mm | first order | rest | 360 × 120 |
| `base_lu` | 1 mm | second order | rest | 360 × 120 |
| `4x` | 0.5 mm | second order | `base_lu` | 720 × 240 |
| `16x` | 0.25 mm | second order | `base_lu` | 1440 × 480 |

The 0.25 mm run is the label. Every run was iterated until the initial residuals of U and p were below 10⁻⁶.

**Splits:** train, val and test are drawn from the same distribution. The three `ood_` sets are out-of-distribution
test sets: cases from the upper tails of blockage, rotation and position, held out of train, val and test.

| Split | Cases | Content |
|---|---|---|
| train | 3,085 | 80 % of the in-distribution cases of every size and family |
| val | 385 | 10 % |
| test | 386 | 10 % |
| ood_blockage | 224 | blockage ratio above the 95th percentile |
| ood_rotation | 216 | absolute rotation above the 95th percentile |
| ood_position | 204 | position shift furthest from the middle of its range |

## Reading the data

- **Arrays:** `<run>/<split>.h5` holds `fields` (N, 6, nx, ny) float32, one sample per chunk; `case_name`; and `time`,
  the iteration of the exported solution.
  - Row i is the same case in all five runs of a split, in the order of `metadata.parquet` (`dataset_index`).
  - The file attributes hold the channel names, the grid, and the mean and standard deviation of every channel
    over the training split of that run.
- **Channels:**
  - 0 `u_x` and 1 `u_y` (m/s);
  - 2 `p`, kinematic pressure in m²/s², 0 at the outlet;
  - 3 `wall_distance`, distance to the closer channel wall in grid points, min(j, ny − 1 − j), 0 on both walls;
  - 4 `cell_type`: 1 wall or obstacle, 2 inlet, 3 outlet, 4 fluid;
  - 5 `obstacle_sdf`: signed distance to the obstacle outline in grid points, 0 on the obstacle's border points
    (stored as −0.0) and negative inside.

  The flow channels are 0 inside the obstacle.
- **Grid:** point (i, j) lies at x = i · 360 / (nx − 1) mm and y = j · 120 / (ny − 1) mm, with the inlet at i = 0.
  The grids of different levels share only their end points.
- **Masks:** `cell_type == 1` covers the obstacle and both channel walls. `obstacle_sdf <= 0` covers the obstacle
  alone (`< 0` would leave out its border points).
- **Attributes:** `<run>.<attribute>` in the metadata are computed on the simulation mesh, by a Python script from the
  OpenFOAM solution and its function-object output.
  - Pressures and stresses are kinematic.
  - `dissipation` is the pseudo-dissipation ν∫|∇u|² dA. It differs from the viscous dissipation 2ν∫S:S dA by
    boundary terms.
  - `wake_min_u_x` is the smallest u_x on a line across the channel 20 mm behind the obstacle, within the obstacle's
    y extent.
  - Forces are for the 10 mm depth of the mesh (× 100 for per metre); dissipation is per metre of depth.
  - Quantities recomputed from the arrays differ by the loss of the resampling, which is largest for gradients:
    dissipation is 1.7 % low at 0.25 mm and 13 % low at 2 mm (median).
- **Uncertainty of the labels:** per case and attribute, in `label_uncertainty.parquet` (the same file as
  `references/label_uncertainty.parquet` of the benchmark repository).
  - `source = openfoam` is the uncertainty of the 0.25 mm solution (Eça & Hoekstra, four second-order levels).
  - `source = array` describes how the array values converge.
  - `method` gives the basis of each estimate: `power`, `low_order` or `high_order` for a fitted estimate,
    `anomalous` for the fallback of three times the largest difference between the levels (no convergence the fits
    can describe), and `equal` where all four values are equal (U = 0 by convention).

## Caveats

- **Not mesh-independent:** peak wall shear and peak vorticity (singular at sharp corners), and front pressure
  (the stagnation point often sits on a corner; no convergence in 2,947 of 4,500 cases). They are included but should
  not be used as targets.
- **Small quantities:** lift is small and of either sign for nearly symmetric obstacles. The pressure and friction
  parts of the drag shift in opposite directions with the mesh. Judge all three relative to the drag.
- **Inlet corners:** they are singular. Gradient attributes leave out the first 5 mm of the channel, and the same is
  advisable for error measures.
- **2 mm runs started from the 1 mm solution:** 27 2 mm runs stalled with SIMPLEC from rest (residuals 10⁻³ to
  3·10⁻²).
  - They were solved again from their converged 1 mm solution, with SIMPLE and under-relaxation (template
    `second-order-relaxed`), and converged.
  - Solved again from rest with the same settings, 4 of them agree within 0.007 % (fields) and 0.001 %
    (attributes).
  - They are marked by `0.25x.templates` and `0.25x.init_from = base_lu`.
- **Raised iteration caps:** in `<run>.end_time_override`.
  - 2 mm: 37 runs (10 to 50,000; the 27 above at 25,000).
  - 1 mm second order: 6 runs (25,000).
  - 0.5 mm: 15 runs (30,000).
  - 0.25 mm: 74 runs (25,000 to 60,000).
  - All of them converged to the same 10⁻⁶ criterion.
- **Remeshed:** meshes that failed checkMesh were made again with `Mesh.Smoothing = 5`: 3 at 2 mm, 3 at 1 mm, 5 at
  0.5 mm, 14 at 0.25 mm (`<run>.mesh_variant`).
- **Dropped cases:** 18 cases whose 1 mm second-order run did not converge were dropped with all their runs and
  replaced by reserve geometries (indices 301 to 330 of their class). They are listed in `dropped_cases.csv`.
- **Timings:** solver and task times are from two CPU models, Intel Xeon E5-2640 and E5-2670 v2 (`<run>.cpu_model`),
  and are not normalised between them.

## Provenance

- **Records:** every run records its status, iterations, final residuals, cap, mesh variant, cell count, template,
  starting fields, times, peak memory, host, CPU, container digest and code revision. All runs used one container
  image.
- **Starting fields:** `<run>.init_from` is never empty: `rest` or `base_lu`.
  - The run records hold it from code revision ec727c7 on.
  - For the 252 earlier 0.25 mm runs, the value `base_lu` comes from the dataset audit: their timings show the
    mapFields step (244 runs), or their archives hold `log.mapFields` (8 continued runs).
  - The metadata was rebuilt at 0b6c60a to fill this column. The arrays and all other files are those built at
    f71b088.
- **Code revision:** `<run>.code_revision` is the revision deployed when the run finished. A run that spanned a
  deploy ran the code before it. 136 of the 252 runs above carry ec727c7, although they started under an earlier
  revision.
- **Geometry:** every geometry is regenerated exactly from the global seed (2026) and the case name with the
  pipeline.
- **Archives:** the complete OpenFOAM case of every run (mesh, the last two time steps, logs, attribute outputs) is
  archived. The repository page of the dataset gives its location.

## Validation (summary; details in the paper and the pipeline's docs/validation.md)

- **Audit:** an audit of every run before the release found no error.
- **Reproduction:** five restored archives, processed again, reproduce the released arrays and attributes bit for
  bit. Three cases solved again from rest at 2 mm and 1 mm needed the same iterations and gave identical fields and
  attributes.
- **DFG 2D-1 benchmark** (the steady laminar flow past a cylinder in a channel of Schäfer, Turek et al., 1996), with
  the production settings: drag, recirculation length and pressure difference within
  the reference intervals (the pressure difference by extrapolation); lift 9 % high on uniform meshes (0.2 % of the
  drag).
- **Starting fields:** they change the iterations, not the solution. At 0.25 mm, solutions from rest and from the
  1 mm solution agree within 0.10 % (fields) and 0.15 % (attributes).
- **Outlet:** no reverse flow reaches the outlet in any 0.25 mm run. The sensitivity of the solutions to the outlet
  position was not tested.
