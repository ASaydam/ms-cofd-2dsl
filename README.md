# MS-COFD: Multi-Scale Central Object Flow Dataset

Code that generates the **Multi-Scale Central Object Flow Dataset (MS-COFD)**: 2D laminar channel flow around a
single object placed in the middle of the channel, with five object shapes in three sizes, 2000 cases, each
simulated on a base and a 16x refined mesh. In the code and in this README the central object is called the
obstacle.

The pipeline is split into stages that read the previous stage's files from disk, so every stage can run on the
machine it needs (geometry and meshing on a laptop, OpenFOAM on a cluster, extraction with ParaView) and can be
resumed or rerun on its own:

```
config.yaml
    │
    ▼
 geometry  ──►  outputs/geo-files/<resolution>/*.geo   + outputs/manifest.csv
    │
    ▼
 mesh      ──►  outputs/mesh-files/<resolution>/*.msh  (gmsh)
    │
    ▼
 foam-setup ─►  outputs/cases/<resolution>/<case>/     (OpenFOAM case folders)
    │
    ▼
 run        ──  openfoam/scripts/  (SLURM or local, gmshToFoam → checkMesh → simpleFoam)
    │
    ▼
 extract    ──  extraction/extract_fields.py  (ParaView, fields resampled to a regular grid, CSV)
    │
    ▼
 build-arrays ► outputs/arrays/<resolution>/<resolution>_tlast_6ch.npy
    │
    ▼
 build-dataset ► outputs/ms-cofd.h5  (layout of mscofd2000_v4.h5)
```

## The dataset

| | |
|---|---|
| Domain | 260 mm × 120 mm channel, 2D (one cell thick, 10 mm) |
| Obstacles | rectangle, diamond, left triangle (pointing to the inlet), right triangle (pointing to the outlet), circle |
| Sizes | normal (1000 cases, 200 per shape), smaller and larger (500 cases each, 100 per shape) |
| Meshes | gmsh, quad-dominant (`Mesh.Algorithm = 8`), characteristic length 1 mm (base) and 0.25 mm (16x) |
| Flow | steady, laminar, incompressible (`simpleFoam`), inlet U = 0.1 m/s, ν = 1e-4 m²/s, no-slip walls and obstacle, p = 0 at the outlet |
| Reynolds number | U·h/ν ≈ 8 – 80 with h the obstacle height (1 mm of obstacle height = Re 1) |
| Arrays | 260 × 120 (base) and 1040 × 480 (16x) grid points, 6 channels (see below) |

Case names are `<size>_<geometry>_<n>`, e.g. `larger_left_triangle_42`.

## Quick start

```bash
pip install -r requirements.txt

python main.py geometry            # 2000 cases × (base, 16x) .geo files, outputs/manifest.csv
python main.py mesh --workers 4    # .msh files (on a laptop: base ~4 s, 16x ~2.5 min each)
python main.py foam-setup          # OpenFOAM case folders

# on the cluster (settings in openfoam/scripts/slurm_settings.sh)
bash openfoam/scripts/submit_all.sh outputs/cases/base
bash openfoam/scripts/submit_all.sh outputs/cases/16x --partition=ai --account=ai --qos=ai --time=24:00:00
python -m extraction.check_runs outputs/cases/base          # converged / not converged / unfinished
bash openfoam/scripts/resubmit.sh continue 5000 outputs/cases/base outputs/cases/base/not_converged.txt

# extraction: python of an environment with the paraview package (or pvpython), in a shell without OpenFOAM
python extraction/extract_fields.py outputs/cases/base outputs/extracts/base --dims 260 120
python extraction/extract_fields.py outputs/cases/16x outputs/extracts/16x --dims 1040 480

python main.py build-arrays base
python main.py build-arrays 16x
python main.py build-dataset       # outputs/ms-cofd.h5 (needs h5py and scipy)
```

Every stage can work on a part of the dataset, e.g.
`python main.py geometry --sizes smaller --geometries circle --start 1 --end 10`.
Existing outputs are skipped (use `--overwrite` to create them again), so an interrupted stage can simply be run
again. Without SLURM, `bash openfoam/scripts/run_all_local.sh outputs/cases/base` runs the cases one by one.

## Stages

### Geometry — `geometries/`

`create_geometries.py` creates the obstacle corners, `sampling.py` decides size and position, `write_geo.py`
writes the `.geo` files. Every case has its own random generator, seeded with `"<seed>-<size>-<geometry>-<n>"`
(`config.yaml: seed`), so

* the same seed always gives byte-identical files, on any machine,
* a single case can be regenerated without generating the others,
* the base, 4x and 16x files of a case share the same geometry; they differ only in the `size` line.

A `.geo` file is `geo-base-files/geo_base_1.geo` (meshing settings and `size`) + the obstacle coordinates
(`x1…x4`, `y1…y4`, and `z11…z42` for the arc mid-points of circles, in mm) + `geo_base_2_<shape>.geo` (outlines,
extrusion, boundary names `inlet`, `outlet`, `fixedWalls`, `frontAndBack`).

`outputs/manifest.csv` has one row per case: seed, sampled width and height, all coordinates and derived values
(bounding box, centroid, area, blockage ratio = bounding box height / 120 mm). Circles are described by the
polygon through their corners and arc mid-points.

#### Sampling

Each obstacle is first sized, then centred in the domain, then each corner is moved by a random offset
(`create_variance`: a random point in a circle of radius `r · (1 − variance)` with `r = width / 5`, or `width / 10`
for diamonds), and the coordinates are truncated to whole mm.

| Size | Scheme | Rectangle, diamond (w × h) | Triangles (w × h) | Circle (w × h) | Corner variance |
|---|---|---|---|---|---|
| normal | `legacy`: integers, uniform | 45–55 × 20–30 | 40–50 × 40–50 | 20–30 × 20–30 | 0.15, circle 0.30 |
| smaller | `jitter`: nominal · (1 + U(±0.1)) × nominal · (1 + U(±0.2)) | 32.5 × 16.25 (diamond 16.5) | 27.625 × 27.625 | 16.25 × 16.25 | 0.15, circle 0.30 |
| larger | `jitter` | 67.5 × 33.75 | 57.375 × 57.375 | 33.75 × 33.75 | 0.15, circle 0.30 |

Smaller and larger nominal sizes are 35 % below / above the middle of the normal ranges.

* **Triangles** are made from a random rectangle: the pointy edge is the middle of one side (left side for left
  triangles, right side for right triangles). A triangle is drawn again until the pointy edge lies between the
  other two corners in y.
* **Circles** are a random rectangle whose sides are splines through an arc mid-point.
  * normal: each side bulges randomly inwards or outwards, by 5 mm (top, bottom) or width / 20 (left, right).
  * smaller / larger: the mid-point is the middle of the side plus a random offset (`create_variance(side / 10, 0.10)`),
    rounded to 0.1 mm.
* Every obstacle has to pass `check_geometry` (inside the domain, counter-clockwise, not crossing itself); all
  2000 released obstacles pass it and it did not reject any of 75 000 generated ones.

### Meshing — `meshing/create_mesh.py`

Runs `gmsh -3 <geo> -o <msh> -format msh2 -n <threads>` once per file (gmsh from `PATH`, the `GMSH_BIN`
environment variable, or the `gmsh` pip package). Writes `outputs/mesh-files/mesh_log_<resolution>.csv` with the
status, time, and node and element counts of every mesh.

### OpenFOAM — `openfoam/`

`create_foam.py` copies `case-templates/<template set>/<resolution>` and the mesh (as `mesh.msh`) into one folder
per case (OpenFOAM v1706, `simpleFoam`, laminar, SIMPLE consistent, 1 non-orthogonal corrector, first-order upwind
for momentum). There are two template sets (`python main.py foam-setup --templates …`, default in `config.yaml`):

| | `released` | `converged` |
|---|---|---|
| used for | the released dataset (`mscofd2000_v4.h5`) | new runs |
| stops at initial residuals | U 1e-3, p 1e-2 | U 1e-6, p 1e-6 |
| pressure solver, relaxation | PCG, p 0.3 / U 0.7 | GAMG, U 0.9 (SIMPLEC) |
| endTime / writeInterval | base 2500 / 10, 16x 3000 / 100 | base 10000 / 100, 16x 6000 / 100 (cap), last 2 written steps kept |

**The released settings stop the 16x runs long before steady state**: the residual criterion is met after ~170
iterations on the 16x mesh, when the wake and the wall boundary layers have not developed yet (`larger_circle_10`:
no recirculation zone instead of 33 mm, pressure drop 50 % too high, 0.03 m/s RMS difference to the converged
solution). The released base runs are close to converged (0.006 m/s RMS). The `converged` settings change only how
the solver gets there, not the discretisation, so they converge to the same solution.

With `--init-from base`, every 16x case starts from the latest solution of its base case (`init_from` file,
`mapFields` in `run_case.sh`), which is much closer to the 16x solution than the uniform `0` folder:

```bash
python main.py foam-setup --templates converged --init-from base     # base and 16x cases
bash openfoam/scripts/submit_pairs.sh outputs/cases/base outputs/cases/16x
```

`submit_pairs.sh` submits every base case and its 16x case; the 16x job waits for its base job
(`--dependency=afterok`). `larger_circle_10`: the 16x case converged in 2738 iterations (1.9 hours on one core),
the same case with the released solver settings from the uniform start had not converged after 11 hours.

`scripts/run_case.sh` runs `gmshToFoam`, `changeDictionary` (sets `frontAndBack` to `empty` and `fixedWalls` to
`wall`, see `system/changeDictionaryDict`), `checkMesh`, `mapFields` (only with an `init_from` file) and
`simpleFoam`. `submit_all.sh` submits the cases of one resolution (skips cases that have started), and
`resubmit.sh` continues (`continue <endTime>`) or cleans and reruns (`rerun`) the cases of a list.
`extraction/check_runs.py` writes `run_status.csv` (status, iterations, run time, last initial residuals) and the
lists `not_converged.txt` / `not_finished.txt` for `resubmit.sh`. With the `converged` settings, a case that stops
at endTime (status `end_time`) did not reach 1e-6; stalled residuals can mean that the flow has no steady solution.
The run time of a continued case is the sum of its runs. `run_case.sh` adds a line per run to `run_info.csv` in the
case folder (host, CPU model, CPUs of the node and how many were allocated when the run started), and
`run_status.csv` reports the host, CPU model and seconds per iteration of every case: run times depend on the CPU type
and on how many jobs share a node (16x pilot on one CPU type: 2.6–4.7 s per iteration), so the iteration count is the
hardware-independent cost, and timings for comparisons are measured separately on one node type.

### Extraction — `extraction/`

`extract_fields.py` resamples every case to a regular grid over the whole domain
(`ResampleToImage`, `UseInputBounds = 1`) and writes the last written time step (the converged solution) of
every case to `<case>_last_time.csv`, with `U`, `p` and ParaView's `vtkValidPointMask` (0 inside the obstacle,
which is outside of the mesh); the exported times are written to `extraction_log.csv`. Run it with the `python` of an
environment that has the `paraview` package (e.g. conda-forge `paraview`, as the released extracts) or with
`pvpython`, in a shell where OpenFOAM is **not** loaded: OpenFOAM's environment puts its own ParaView on `PATH`
and `LD_LIBRARY_PATH`, which breaks the ParaView session (`cannot import name 'OpenFOAMReader'`). The
environment must not have the pip `vtk` package as well (it comes with e.g. `pyvista`): its `vtkmodules` are
loaded instead of ParaView's own VTK (`vtkInitializationHelper.Initialize ... arguments do not match`). A clean
environment: `conda create -n ms-cofd-paraview -c conda-forge python=3.9 paraview=5.10.1`.

`build_arrays.py` (`python main.py build-arrays <resolution>`) turns them into one array per resolution,
`<resolution>_tlast_6ch.npy` of shape (cases, 6, nx, ny) with x along the flow (inlet at x = 0), and `cases.csv`
(case name of every row, ordered by the name of the CSV file). The channels are the ones of the MS-COFD file:

| Channel | Name | Paper | Content |
|---|---|---|---|
| 0 | `u_x` | | x velocity (m/s) |
| 1 | `u_y` | | y velocity (m/s) |
| 2 | `p` | | kinematic pressure (m²/s²) |
| 3 | `wall_distance` | Wall Distance | distance to the closer channel wall, in grid points (`min(j, ny − j)` for row j) |
| 4 | `cell_type` | Cell Type Mask | 1 wall or obstacle, 2 inlet, 3 outlet, 4 fluid |
| 5 | `obstacle_sdf` | Signed Obstacle Distance | Euclidean distance to the closest obstacle border point, in grid points, negative inside the obstacle |

The geometry channels come from the obstacle points of the same export (`vtkValidPointMask`; for the released
extracts, which have no mask, the points with `u_x = 0`, which gives the released `cell_type` channel of every
released base case). The arrays are float32: the exported fields have 5 significant digits (OpenFOAM writes 6), so
float32 keeps all of them and halves the size; `--dtype float64` gives the type of `mscofd2000_v4.h5`.

### Dataset file — `dataset/`

`build_dataset.py` (`python main.py build-dataset`) writes `outputs/ms-cofd.h5` in the layout of the released
`mscofd2000_v4.h5`, for the cases found in both resolutions:

| Key | Shape | Content |
|---|---|---|
| `base_tlast_6ch` | (N, 6, 260, 120) | coarse solution and geometry (F_c) |
| `16x_tlast_6ch` | (N, 6, 1040, 480) | fine solution and geometry (F_f) |
| `resampled_16x_tlast_6ch` | (N, 6, 260, 120) | `16x_tlast_6ch` reduced by non-overlapping 4 × 4 **max pooling** of every channel (F_rf), as in the released file: every value is the largest of 16 fine values, `wall_distance` is in fine-cell units, and `cell_type` only has 1 (obstacle) and 4 (the wall, inlet and outlet labels are pooled away) |
| `metadata/*` | (N,) | the 34 per-sample descriptors of the released file (obstacle size and position, flow statistics, coarse-vs-fine gap) and `case_name` |

The descriptors, channel names, units and key descriptions come from `dataset/build_mscofd_v4.py`, the script
that created `mscofd2000_v4.h5` from v3 (it fixed the channel order of samples 1000–1999, which v3 stored as
`cell_type, wall_distance`). `dataset/verify_geometry_channels.py` checks the channel roles of a file.

## How the released data maps to this code

The released dataset was created before this repository, with two earlier codebases whose random generators
were not seeded, so its cases cannot be regenerated one by one. The released `.geo` and `.msh` files are the
reference; this code generates new cases from the **same distributions** and runs them through the same meshing,
solver and extraction settings.

| Released | Here |
|---|---|
| normal cases `1.geo` … `1000.geo` (1–200 rectangle, 201–400 diamond, 401–600 left triangle, 601–800 right triangle, 801–1000 circle) | `normal_<geometry>_<n>`, see `tools/build_manifest.py` |
| smaller / larger cases `<size>_<geometry>_<n>` | same names |
| 16x `.geo` files | base file with `size = 0.00025` (400 smaller-size 16x `.geo` files were not kept, their `.msh` files were) |
| `mscofd2000_v4.h5`, rows 0–999 | `normal_<geometry>_<n>` in the order of `1.geo` … `1000.geo` |
| `mscofd2000_v4.h5`, rows 1000–1999 | smaller / larger cases sorted by the name of their extract (`larger_circle_100`, `larger_circle_10`, `larger_circle_11`, …) |
| array channels `BC`, `SDF-1`, `SDF-2` | `cell_type`, `wall_distance`, `obstacle_sdf` |

`mscofd2000_v4.h5` was released without case names; `python -m tools.add_case_names mscofd2000_v4.h5 --write` added
`metadata/case_name` to it (in place, nothing else changed; the file's md5 before the change is stored in its
`case_names_added` attribute). The row order above was checked sample by sample first: the obstacle of the expected
case, rasterized from its `.geo` coordinates, is the best match of all cases of its size and geometry for every one
of the 2000 samples (intersection over union: median 1.000, minimum 0.813). The checked order is in
`data/released_case_names.csv`. `python -m tools.released_cases <folder>` selects released cases (tallest and median obstacle of every size and
geometry) and copies their released meshes, to run them again with other settings; `python -m tools.gap_report`
compares the re-run cases with the released file (coarse-vs-fine gap, change of the 16x fields, recirculation).
`python -m tools.openfoam_attributes run <cases>` (with OpenFOAM loaded) computes flow attributes of the latest
time step of every case with OpenFOAM's function objects, on the simulation mesh: drag and lift (pressure and
friction parts), pressure and wall shear stress on the obstacle surface, pressure drop, vorticity, velocity gradient
and velocity along lines across the channel. The obstacle is part of the `fixedWalls` patch of the meshes, so
`openfoam/scripts/openfoam_attributes.sh` moves its faces to their own patch in a copy of the case (the case itself is
not changed). `python -m tools.openfoam_attributes collect <cases>` (numpy, no OpenFOAM) turns the results into
`openfoam_attributes.csv`: forces, stagnation and base pressure, peak wall shear, pressure drop, peak vorticity,
dissipation, recirculation length and area, flow split above / below the obstacle and wake deficit.
`python -m tools.flow_attributes <dataset .h5> --openfoam openfoam_attributes.csv` computes the same attributes from
the arrays and compares (1) base with 16x case by case, with the OpenFOAM values as reference, plus field errors, and
(2) the array values with the OpenFOAM values, i.e. how much of an attribute is lost on the resampled grid that a
network sees. For a converged `larger_circle_10` the array values of recirculation length and area, flow split,
pressure drop and wake deficit are within 0.1–3.3 % of OpenFOAM's; dissipation is 2–8 % and peak vorticity 27–36 %
lower on the arrays (resampling smooths the shear layers).

`python -m tools.upsample_mse <dataset .h5> [--released mscofd2000_v4.h5]` upsamples the base fields to the 16x grid
(nearest, bilinear, bicubic, and bicubic after filling the obstacle, whose points hold 0 in the arrays) and reports
per case and field the MSE over the fluid points of the 16x grid, the RMS difference relative to the 16x field and
the share of the squared difference within 2 mm of the obstacle; these are the interpolation baselines for
coarse-to-fine prediction (bilinear and bicubic with a = -0.75 match `torch.nn.functional.interpolate` with
`align_corners=True` to 1e-13; the tool uses a = -0.5).

Mesh convergence study: `python -m tools.released_cases <folder>/mesh-files --resolutions --geo 4x` writes the
released base `.geo` files of the selected cases with the 4x mesh size (0.5 mm; no 4x meshes were released for the
dataset), `main.py mesh` meshes them, `foam-setup --resolutions 4x --init-from base` sets them up with the
`converged/4x` template and `openfoam/scripts/submit_pairs.sh <base cases> <4x cases>` submits them. With the
attributes collected for `--resolutions base 4x 16x`, `python -m tools.mesh_study openfoam_attributes.csv` reports
per case and attribute the convergence type, the observed order, the Richardson-extrapolated value and the grid
convergence index of the 16x mesh (Celik et al. 2008). The written 4x `.geo` files are byte-identical to the
released 4x `.geo` files of the normal cases, and gmsh 4.9.5 reproduces the released base meshes (same elements,
node coordinates within 2e-15 m between the Windows and Linux builds).
`python -m tools.compare_with_released <arrays folder or .h5 file> --released mscofd2000_v4.h5`
compares new results with the released ones case by case. Running released meshes through this pipeline reproduces
the released base, 16x and resampled arrays and all 34 descriptors (checked for `larger_circle_10` and
`larger_circle_100`: equal to float32 rounding, same solver iterations).

`data/released_manifest.csv` is the parameter table of the 2000 released cases, written by
`python -m tools.build_manifest` from the released base `.geo` files. It has the same columns as
`outputs/manifest.csv`: case name, source file, size, geometry, case number, corner coordinates `x1…x4`, `y1…y4`
and circle arc mid-points `z11…z41` (x) / `z12…z42` (y) in mm, bounding box width and height, centroid, area and
blockage ratio (the seed and sampled width / height were not recorded for the released cases). `python -m tools.compare_distributions` compares generated and
released obstacles (two-sample Kolmogorov–Smirnov test per coordinate and derived value):

* 207 tests, 2 with p < 0.01 (≈ 2 expected by chance), no group with a systematic difference.
* One small difference remains: the widest **normal triangles** are a little rarer in the released data
  (95th percentile of the bounding box width 54.5–56 mm released, 57–57.5 mm generated). The original normal
  pipeline meshed every geometry with `blockMesh` and drew it again when `checkMesh` failed; this step is not
  reproduced here (it needs OpenFOAM, and a corner-angle criterion did not explain the difference).

### Kept on purpose, so the distributions match the released cases

* smaller / larger: diamonds and circles are centred with the size of a separately drawn rectangle, which moves
  them a few mm away from the centre line (meshForge did this).
* `smaller` diamond nominal height is 16.5 mm (16.25 would be 35 % below normal).
* normal circles: the left arc uses the random sign of the top arc (the original code never used the fourth sign).
* Wall Distance is 0 on the lower wall row and 1 on the upper one (`min(j, ny − j)`).

### Fixed with respect to the original code

* no seed → one seeded random generator per case.
* base and 16x geometries were drawn independently by meshForge's `main.py` → drawn once, written per resolution.
* `create_right_triangle` retried with `create_left_triangle` → retries with itself (never triggered in the
  released data, all released right triangles point right).
* all five shapes were created (and random numbers drawn) for every case → only the selected shape.
* `gmshToFoam` boundary types were set with `sed` on fixed line numbers → `changeDictionary` by patch name.
* job scripts: `RET=$?` returned the exit code of `grep`, not of the solver; `folder=(basename …)` created an array.
* extraction: the notebooks exported the first written iteration as "zero_time" (not the 0 folder, which they
  intended) only to find the obstacle → only the last time step is exported, the obstacle comes from ParaView's
  `vtkValidPointMask`;
* extraction: the notebooks' `SamplingBounds` in mm were ignored by ParaView → `UseInputBounds` is set explicitly;
  the log parsing (`lines[-7]`) failed on continued runs → `check_runs.py`; the obstacle is now found with
  ParaView's `vtkValidPointMask` (the released extracts used `Ux == 0`, which gives the same mask).
* meshing needed the POSIX-only `sh` package and a bash loop → one `subprocess` call per mesh, works on Windows.

## Environment of the released dataset

| Tool | Version |
|---|---|
| gmsh | 4.9.5 |
| OpenFOAM | v1706 (ESI), KUACC cluster, SLURM |
| ParaView | 5.10.1 (conda-forge `paraview` package), `OpenFOAMReader` and `ResampleToImage` |
| Python | 3.8 / 3.9 (normal cases), numpy |

Meshing with another gmsh version gives slightly different meshes from the same `.geo` file, so the released
`.msh` files are the reference for the released cases.

## Tests

```bash
python -m pytest tests
```

Tests that compare with the released files (`../mesh data/geo_files`, `../meshForge-set/geo-files`) are skipped when
those folders are not there.

## Citation and license

If you use this code or MS-COFD, please cite it as described in `CITATION.cff`.
License: to be decided before publication (see `LICENSE`).
