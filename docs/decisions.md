# Decision log

Every choice that shapes the dataset, with its date and reason. The evidence behind the early decisions is the
MS-COFD (v4) pilot: 30 released geometries re-run to convergence at base / 4x / 16x with first- and second-order
convection (KUACC, 30 Sep – 2 Oct 2026; backed up to Google Drive, `ms-cofd-backup/2026-10-04`).

| Date | Decision | Reason |
|---|---|---|
| 2026-09-30 | Every run converges to initial residuals of 1e-6 (residualControl); endTime is only a cap | The v4 16x runs stopped at U 1e-3 after ~170 iterations, far from steady state; converged runs shrink the base-to-16x field gap 260–440× (bicubic MSE, 30 cases) |
| 2026-10-01 | A third resolution, 4x (0.5 mm), for every case | Per-case mesh study (order, Richardson extrapolation, GCI): the uncertainty of every 16x label. 4x costs ~0.3 core-h (~8 % of the total) |
| 2026-10-04 | Second-order upwind convection (linearUpwind) for base, 4x and 16x | Pilot mesh study: first-order upwind sets the error (order ~1); with linearUpwind the 1 mm error of the recirculation length drops from 16 % to 1.9 % and the 16x error to 0.3 %, at slightly lower cost (16x 3.5 vs 4.3 core-h) |
| 2026-10-04 | A first-order base run as well (base_fo) | ~2 core-min per case. Keeps the low-fidelity solver of v4 for comparison, and lets a paper compare "switch the scheme" with "train a network" |
| 2026-10-04 | 4500 new seeded geometries (300 per size × geometry); the 2000 released geometries are not reused | Fully reproducible from code and seed; the v4 comparison rests on the pilot |
| 2026-10-04 | One continuous size scheme for all sizes: nominal × scale (normal 1, smaller 0.65, larger 1.35) × (1 ± 10 % width, ± 20 % height) | The released data used integer sizes for normal and jitter for smaller / larger. Nominal sizes are the middle of the old normal ranges; triangles use 42.5 mm, the size from which the v4 smaller / larger triangles were scaled (27.625 = 0.65 × 42.5) |
| 2026-10-04 | Rotation ±45° about the centroid; shifts x −30…+10 mm, y ±15 mm | More variety. ±45° keeps every family's identity (a right triangle still points downstream); the shifts are biased upstream to leave room for the wake |
| 2026-10-04 | Minimum gaps: 5 mm to the walls, 60 mm to the inlet, 150 mm to the outlet; invalid obstacles are drawn again | Valid meshes and room for the wake; rejection is rare (≤ 8.3 % of draws, larger rectangles and right triangles), so the distributions are barely changed |
| 2026-10-04 | Channel 360 × 120 mm (was 260 × 120); the obstacle stays 130 mm from the inlet | In the pilot the reverse flow reached the outlet in 6 of 30 cases (blockage ≥ 0.49); +100 mm downstream. Costs ~38 % more cells |
| 2026-10-04 | Fixed inlet velocity 0.1 m/s (Re 16–80 by obstacle height in v4, up to ~90 now) | Steady flow assured; varying it risks unsteady cases inside the 14-day allocation |
| 2026-10-04 | Fixed, seeded splits (train / val / test) and out-of-distribution test sets from the tails of rotation, position and blockage | Comparable benchmark results |
| 2026-10-04 | Locally refined 1 mm meshes for the test and OOD cases only | The baseline a learned coarse-to-fine correction has to beat, at small cost |
| 2026-10-04 | Container: OpenFOAM v2512 (ESI) + micromamba environment (gmsh 4.15.2, VTK 9.7.0, Python 3.14), built by GitHub Actions, run with Apptainer | KUACC runs CentOS 7 (glibc 2.17) with only OpenFOAM v1706; the container gives current software and an image anyone can run. Image public on ghcr.io, the repository private until publication |
| 2026-10-04 | VTK (vtkOpenFOAMReader + vtkResampleToImage) instead of ParaView for the arrays | The same filters ParaView used for v4, headless and without a GUI build; validated against the ParaView 5.10.1 arrays of the pilot |
| 2026-10-04 | Meshes are made inside each task; every task archives its own output and moves it to Google Drive | The user quota on KUACC is 1 TiB and must never be exceeded; the raw output is ~3.4 TB. Submissions pause above 600 GiB used or 200 GiB not uploaded |
| 2026-10-04 | Keep everything in the backup (case folders with both kept time steps, meshes, logs, attribute work folders) | Any attribute can be recomputed later without re-running the solver |
| 2026-10-04 | One SLURM job per task (one run of one case), at most 280 queued (QoS limit 300) | Per-task timing and memory for the cost study, single tasks can be rerun; the queue never floods the scheduler |
| 2026-10-04 | Code reaches the cluster as a git archive of a committed revision (REVISION file), not a clone | No credentials on the cluster; every result records the exact commit |
| 2026-10-04 | Pressure drop also on cross-sections 10 mm from the inlet and the outlet (dp_sections) | The area average on the inlet patch includes the singular inlet corners (pilot mesh study: it does not converge) |
| 2026-10-04 | Licences: data CC BY 4.0, code MIT; name MS-COFD-2DSL | User's choice |
