# MS-COFD-2DSL

**Multi-Scale Central Object Flow Dataset – 2D Steady Laminar**: the pipeline that generates the dataset, from the
random obstacle geometries to the arrays, attributes and metadata.

Every case is a two-dimensional laminar channel flow around one obstacle, solved to steady state with OpenFOAM on
four meshes (2 mm, 1 mm, 0.5 mm and 0.25 mm cells), and on the 1 mm mesh with two convection schemes. The dataset
is meant for learning and benchmarking flow prediction across resolutions (coarse-to-fine correction,
super-resolution, surrogate modelling), with the mesh convergence of every case documented.

## The dataset

| | |
|---|---|
| Cases | 4500: 5 obstacle families × 3 size classes × 300 |
| Families | rectangle, diamond, left triangle (pointing upstream), right triangle (pointing downstream), circle (four splines) |
| Size classes | normal, smaller (0.65×), larger (1.3×) |
| Domain | channel 360 mm × 120 mm, obstacle nominally centred 130 mm from the inlet |
| Variety | width ±10 %, height ±20 %, random corner offsets, rotation ±45° about the centroid, shift −30…+10 mm along and ±15 mm across the channel; blockage ratio (obstacle height / channel height) at most 0.40, where the flow stays steady |
| Flow | incompressible, laminar, steady; ν = 1·10⁻⁴ m²/s; uniform inlet velocity 0.1 m/s |
| Boundary conditions | inlet: fixed velocity, zero-gradient pressure; outlet: zero-gradient velocity, p = 0; walls and obstacle: no-slip |
| Runs per case | `0.25x` 2 mm · `base_fo` 1 mm, first-order upwind · `base_lu` 1 mm · `4x` 0.5 mm · `16x` 0.25 mm (all others second-order upwind) |
| Convergence | every run until the initial residuals of U and p are below 1·10⁻⁶ (SIMPLEC); no run of a case starts before its 1 mm second-order run converged; a case where that run stops at its cap (it did not converge) is dropped with all its runs and replaced by the next reserve geometry of its size and family (index 301 on, same seed), so every class keeps 300 cases |
| Arrays | 6 channels per run on a regular grid: 180 × 60 (2 mm), 360 × 120 (1 mm), 720 × 240 (0.5 mm), 1440 × 480 (0.25 mm), float32 |
| Attributes | computed on the simulation mesh, by a Python script from the OpenFOAM solution: drag and lift (pressure and friction parts), stagnation and base pressure, mean and peak wall shear, mean and peak vorticity, pressure drop, recirculation length and area (the reverse flow connected to the obstacle), flow split, wake deficit, dissipation, reverse flow at the outlet |
| Surface distributions | pressure and wall shear along the obstacle surface, every run |
| Mesh study | order of convergence, extrapolated value and grid convergence index of every attribute (1, 0.5, 0.25 mm), and the error of every level against the extrapolated value |
| Metadata | sampled parameters, geometry, convergence, iterations, timings, peak memory, hardware, software versions of every run |

### Benchmark targets

- **Attributes that converge with the mesh**: drag and lift with their pressure and friction parts, mean wall shear,
  mean vorticity, recirculation length and area, wake deficit, dissipation, pressure drop between cross-sections,
  flow split. Each comes with its mesh-study uncertainty (grid convergence index of the 0.25 mm value).
- **Surface distributions**: pressure and wall shear coefficients along the obstacle.
- **Peak values** (peak wall shear, peak vorticity) are included but not mesh-independent: at sharp corners the exact solution is
  singular, so they keep growing with refinement and are not mesh-independent targets.
- **Stagnation (front) pressure** is included but not mesh-independent either: the stagnation point often sits on a corner, and
  the value did not converge with the mesh in about two thirds of the cases.

### Array channels

| Channel | Name | Content |
|---|---|---|
| 0 | `u_x` | x velocity (m/s) |
| 1 | `u_y` | y velocity (m/s) |
| 2 | `p` | kinematic pressure (m²/s²) |
| 3 | `wall_distance` | Wall Distance: distance to the closer channel wall, in grid points, min(j, ny − 1 − j) |
| 4 | `cell_type` | Cell Type Mask: 1 wall or obstacle, 2 inlet, 3 outlet, 4 fluid |
| 5 | `obstacle_sdf` | Signed Obstacle Distance: distance to the obstacle border in grid points, negative inside |

Arrays have the shape (6, nx, ny): x along the flow (inlet at index 0), y across the channel. The grid points
include both ends of the domain. Points inside the obstacle are 0 in the flow channels.

## How a case is made

1. **Geometry** (`geometries/`): the obstacle is drawn from its own random generator, seeded with the global seed
   and the case name. An obstacle that does not keep 5 mm from the walls, 60 mm from the inlet and 150 mm from the
   outlet, blocks more than 40 % of the channel height, or crosses itself, is drawn again. Written as gmsh `.geo` files (one per resolution, identical apart
   from the mesh size) and a manifest.
2. **Mesh** (gmsh): unstructured quad-dominant 2D mesh, extruded one cell in z, converted with `gmshToFoam`.
3. **Solver** (OpenFOAM `simpleFoam`, `openfoam/case-templates/`): the 4x and 16x runs start from the converged
   `base_lu` solution (`mapFields`).
4. **Arrays** (`extraction/`): the converged fields are resampled to the regular grid with VTK
   (`vtkOpenFOAMReader`, `vtkResampleToImage`), and the geometry channels are made from the same resampling.
5. **Attributes** (`tools/openfoam_attributes.py`, `openfoam/scripts/openfoam_attributes.sh`): OpenFOAM function
   objects on a copy of the case, with the obstacle as its own patch.
6. **Archive**: the whole case (mesh, both kept time steps, logs, attribute outputs) is compressed and stored
   off the cluster; the arrays, attributes and a provenance record of every run stay with the results.

## Software

Everything runs in one container, built from `environment/Dockerfile` by GitHub Actions and published as
`ghcr.io/asaydam/ms-cofd-2dsl`:

| Software | Version |
|---|---|
| OpenFOAM (ESI) | v2512 |
| gmsh | 4.15.2 |
| VTK | 9.7.0 |
| Python | 3.14 (numpy, scipy, h5py, pyarrow, pyyaml, matplotlib) |
| rclone, zstd | for the archives |

The exact package list of every image is in `environment/versions/` (`conda-explicit.txt`, `pip-freeze.txt`,
`summary.txt`, `image-digest.txt`). Every run records the image digest and the code revision it ran with.

```bash
docker pull ghcr.io/asaydam/ms-cofd-2dsl@sha256:<digest>                       # Docker
apptainer pull ms-cofd-2dsl.sif docker://ghcr.io/asaydam/ms-cofd-2dsl@sha256:<digest>   # HPC clusters
```

Inside the container, `source /opt/mscofd/activate.sh` loads OpenFOAM and the Python environment.

## Running the pipeline

### Geometry (anywhere with Python, numpy and pyyaml)

```bash
python main.py geometry                                  # all 4500 cases, .geo files and the manifest
python main.py geometry --sizes larger --geometries circle --end 5
```

The output goes to the data root: `MSCOFD_DATA_ROOT`, or `outputs/` in the repository. All settings are in
`config/dataset.yaml`.

### A single case by hand (in the container)

```bash
export MSCOFD_DATA_ROOT=/path/to/data
python main.py geometry --sizes normal --geometries rectangle --end 1
python -m hpc.run_task base_lu normal_rectangle_0001     # mesh, solve, arrays, attributes, archive
python -m hpc.run_task 16x normal_rectangle_0001         # starts from the base_lu solution
```

### Production on a SLURM cluster (`hpc/`)

The production runs one SLURM job per task (one run of one case); nothing runs on the login node.

| Script | Where | What |
|---|---|---|
| `hpc/deploy.sh` | your computer | copies a committed revision of the repository to the cluster (`REVISION` holds the commit) |
| `hpc/pull_image.sh` | cluster | pulls the container by digest with Apptainer and makes it current |
| `hpc/env.sh` | cluster | paths, container, caches and rclone configuration, sourced by every script |
| `hpc/topup.py` | cluster | submits the tasks that are ready, up to the queue limit of `config/slurm.yaml`; pauses when the disk usage or the upload backlog is too large; submits interrupted tasks again |
| `hpc/job_task.sh` | cluster (SLURM) | one task: `hpc/run_task.py` in the container, then `topup.py` |
| `hpc/status.py` | cluster | tasks per run and state, queue, compute used, disk usage, failures |
| `hpc/upload.py` | cluster | uploads finished archives with rclone and checks their MD5 before deleting them |

Typical session:

```bash
bash hpc/deploy.sh                                       # on your computer
# on the cluster
source repo/hpc/env.sh
bash repo/hpc/pull_image.sh sha256:<digest>
apptainer exec "$MSCOFD_IMAGE" bash -c "source /opt/mscofd/activate.sh && cd $MSCOFD_REPO && python main.py geometry"
# rclone remote in $MSCOFD_HOME/config/rclone.conf; for Google Drive with your own OAuth client ID (rclone's shared
# client ID is being retired), authorised on a computer with a browser: rclone authorize "drive" <client id> <secret>
echo "remote:ms-cofd-2dsl/archives" > "$MSCOFD_HOME/config/remote"   # rclone target of the archives
python3 repo/hpc/topup.py                                # starts the production; every job keeps it going
python3 repo/hpc/status.py
```

Tasks are ordered so that the finished part of the dataset stays balanced over the sizes and families at any
time. Every step of a task is skipped when its result exists, so a task interrupted by preemption or the time
limit continues where it stopped.

## Reproducibility

- **Geometry**: fully determined by `seed` in `config/dataset.yaml` and the case name (`random.Random` seeded with
  `"<seed>-<size>-<geometry>-<index>"`), independent of the order in which cases are generated.
- **Meshes**: gmsh version and settings are fixed in the container, one thread per mesh.
- **Solutions**: converged to 1·10⁻⁶; the solver settings (`fvSolution`) do not change the converged solution,
  only the discretisation (`fvSchemes`) does.
- **Provenance**: every run writes `<case>_provenance.json` with its status, iterations, residuals, timings, peak
  memory, host and CPU, the container digest and the code revision.
- **Decisions**: the reasons behind the settings are in `docs/decisions.md`.

## Repository layout

```
config/          dataset.yaml (all dataset settings), slurm.yaml (cluster settings)
geometries/      obstacle sampling and .geo files (geo-base-files/: gmsh templates)
meshing/         gmsh calls for meshing many files at once
openfoam/        case templates (first-order, second-order), case setup, run and attribute scripts
extraction/      VTK resampling and the 6 channel arrays, run checks
tools/           OpenFOAM attributes, array attributes, mesh convergence study, interpolation baselines
dataset/         readers of the per-case results
hpc/             production on a SLURM cluster
environment/     Dockerfile, conda environment, version records
docs/            decision log
tests/           pytest (run in the container by the CI)
```

## License and citation

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`DATA_LICENSE`). Please cite as described in `CITATION.cff`.
