# Validation

Every check of the pipeline before and during the production, with its numbers and pass criterion. The preliminary
study is the one of `docs/decisions.md`: 30 obstacle geometries in a 260 × 120 mm channel, second-order convection,
1, 0.5 and 0.25 mm meshes, solved with OpenFOAM v1706.

## 1. OpenFOAM of the container (v2512) against v1706

**What**: 5 geometries of the preliminary study (larger_circle_87, larger_diamond_52, smaller_rectangle_13,
smaller_left_triangle_71, normal_right_triangle_44), solved again on the same meshes (1, 0.5 and 0.25 mm) with the
container's OpenFOAM and case templates (`validation/solver_check.sh`). Attributes of both versions computed with
the same tools (`tools/openfoam_attributes.py`); compared with `validation/compare_solvers.py`.

**Pass criterion**: attributes within 0.5 %, fields within 0.1 % (RMS).

**Result (2026-10-04): passed.**

| | |
|---|---|
| Meshes | identical (owner and neighbour lists) in all 15 runs |
| Iterations to convergence | identical in 14 runs, one more in v1706 in one (1411 instead of 1410) |
| U and p, cell by cell | identical to the written precision (6 digits): largest RMS difference 0.0003 %, largest pointwise difference 0.0006 % |
| Attributes (19) | largest difference 0.0004 % (peak vorticity), median 0.0000 % |

Also on identical flow fields: the attributes computed with the v2512 tools equal those computed with v1706 for all
30 × 3 runs of the preliminary study.

## 2. Locks on the cluster file system

**What**: KUACC's `/scratch` is BeeGFS with `tuneUseGlobalFileLocks = false` and `tuneUseGlobalAppendLocks = false`,
so `flock` and appends only protect processes on the same node. The pipeline uses a lock made with `mkdir`
(`hpc/ledger.py`, `DirectoryLock`). Test: two processes on two nodes (be10, be11) take the same lock 20 times each
and hold it for 0.3 s.

**Pass criterion**: no overlap of the hold intervals.

**Result (2026-10-04): passed**, 40 holds, no overlap. (Found in the pilot: with `flock`, `hpc/topup.py` calls of
jobs ending on different nodes ran at the same time and submitted six tasks two to four times.)

## 3. Pilots of the 360 x 120 mm domain (2026-10-04/05)

Three pilots with the production pipeline, each on the obstacles at the extremes of blockage, rotation and position
(plus the obstacles closest to the blockage limit), run on the cluster with the temporary allocation.

| Pilot | Geometry | 1 mm, second order (base_lu) | Finding |
|---|---|---|---|
| 1 | 45 obstacles, larger class 1.35x, no blockage limit | 37 of 45 converged; all 30 below a blockage of 0.47 converged, 8 of 15 from 0.47 to 0.76 did not (residuals stuck at 1e-3 to 1e-2) | No steady solution above a gap Reynolds number of ~105; these obstacles also caused reverse flow at the outlet (4 cases). Blockage limit introduced |
| 2 | 24 obstacles at a limit of 0.45, larger class 1.3x | 18 of 24 converged; the 6 others at a blockage of 0.448 to 0.450, mostly triangles | Limit lowered to 0.40; no reverse flow at the outlet |
| 3 | 30 obstacles at the 0.40 limit | all converged except 4 short, blunt left triangles (normal size, blockage 0.397 to 0.400) | Such cases are flagged and replaced from reserve geometries |

Also found and fixed in the pilots: a folded 0.25 mm mesh (remeshing with gmsh option variants), two nodes where the
container cannot start (excluded), duplicate jobs from locks that do not hold across nodes (mkdir locks). In every
finished pilot case whose 1 mm run converged, the 0.5 and 0.25 mm runs converged too (pilot 1, by 5 Oct 04:30:
36 at 0.5 mm, 12 at 0.25 mm). Resources: see docs/decisions.md.

## 4. Stop before the time limit (2026-10-17)

Data root `limit-test` (validation/limit_test, job names `m2t_...`): larger_diamond_0231 on 1 mm and 0.5 mm with
15-minute jobs and a stop margin of 300 s. The 0.5 mm run was stopped twice after 10:00 by its job, submitted again by
the same job's topup.py within a second, and converged in the third job, after 1,653 iterations in 3 solver runs
(production: 1,653 in one). Against the production run: attributes equal to 2.3e-7 (relative, base pressure), arrays
to an RMS difference of 2.4e-10 of the field RMS, same mesh (171,736 cells, variant 0).

## 5. Dataset audit (2026-10-17)

**What**: `validation/audit_dataset.py --remote` on the production root, every run of every case:
- members and states;
- provenance (residuals, templates, starting fields, caps);
- archives (one live row per run; the size and MD5 of every archive on the remote);
- arrays (shape, finite values, obstacle points, geometry channels, obstacle area and centroid against the manifest,
  volume flux of the cross-sections);
- attributes, and the attributes of every level against 0.25 mm;
- mesh study, splits, `.geo` files written again from the seed, and leftovers.

**Pass criterion**: no error.

**Result: passed** (final run, revision a7bf2e8, after the cleanup of the archive index and the seeds,
decisions 2026-10-17).

| | |
|---|---|
| Errors | 0 |
| Warnings | 8: 16x runs of 5–6 Oct continued in a second job, whose provenance does not record the start (init_from was recorded from ec727c7 on). Their archives contain `log.mapFields` and the `mapped` marker, so every one started from its base_lu solution; the other 244 runs of that period show their mapFields step in the provenance |
| Archives | 22,500 live, every size and MD5 equal on the remote, 1.56 TB |
| Arrays | volume flux of the cross-sections within 0.69 % (2 mm) and 0.022 % (0.25 mm) of the inlet flux; obstacle area within 0.67 % (2 mm) and 0.02 % (0.25 mm) of the manifest (median) |
| Geometry | every `.geo` file written again from the seed is identical |
| Levels | no case with outliers in 3 or more attributes (a mixed-up case); 19 single-attribute outliers, all physical: the first-order 1 mm wake velocity of smaller circles and larger diamonds (5–7 %), the base pressure of 2 cases |

Report: `audit/2026-10-17-final` of the data root. The first run (same day, before the fixes of a4ce01b) had 252
errors, all of them the missing init_from of the runs above, and warnings from limits that were too tight.

## 6. Dataset files v1.0.0 (2026-10-17/18)

**What**: `dataset/build_dataset.py` at f71b088, into `dataset/v1.0.0` of the data root, then read back.

**Pass criterion**: the splits equal those of the audit, the same case in row i of every run, every checksum correct.

**Result: passed.**
- 4,500 cases (18 reserve cases in use, 18 dropped).
- Splits: train 3,085, val 385, test 386, ood_blockage 224, ood_position 204, ood_rotation 216.
- The splits equal the audit's `splits.csv` for all cases.
- In every split, the case_name of the 5 runs is identical and in the order of `metadata.parquet`.
- Every MD5 of `checksums.md5` is correct. It lists the 36 data files (104,026,680,594 bytes); with `checksums.md5`
  itself the folder holds 37 files and 104,026,682,453 bytes (`du -sb` gives 42 bytes more: directory entries).
- MD5 of `checksums.md5` itself: 817adefb132205f30bf633710f554dc2.

**Metadata rebuild (2026-10-19), before any publication.**
- **Why:** `<run>.init_from` was empty for runs that started from rest and for the 252 0.25 mm runs from before
  ec727c7, which did not record their start.
- **What:** `dataset/build_dataset.py --metadata-only` at 0b6c60a (job 3198334) wrote new `metadata.parquet`,
  `metadata.csv` and `dataset_info.json`. `DATACARD.md` (docs/DATACARD.md at 0398659) was added. The arrays and the
  other tables are unchanged.
- **Checked:**
  - Old and new metadata compared cell by cell (audit/compare_metadata_2026-10-19.py): 4,500 rows and 259 columns
    on both sides, same rows and order.
  - Exactly 13,725 cells differ, all in the `init_from` columns and all from empty to filled:
    - 0.25x: 4,473 → rest;
    - base_fo and base_lu: 4,500 → rest each;
    - 16x: 252 → base_lu;
    - 4x: unchanged.
  - The only type change: base_fo / base_lu `init_from` from null to string.
  - `dataset_info.json` only gained `metadata_rebuilt` and `metadata_builder_revision`.
- **Release after the swap:**
  - 37 data files (104,026,744,970 bytes) and `checksums.md5`, 38 files and 104,026,746,875 bytes;
  - `md5sum -c` passes for all 37; the folder is read-only again;
  - Drive matches (38 files, 0 differences).
- **New MD5 of `checksums.md5`:** 209ebc4f85e13235bbc3e6b7f2a63243. The four files they replaced are kept in
  `dataset/v1.0.0-metadata-f71b088/`.

## 7. Reproduction check (2026-10-18)

**What**: `validation/check_reproduction.py` (8f6e78d).

*archives*: the archives of 5 runs restored from the remote. Their arrays were made again with VTK and their
attributes with OpenFOAM:
- 16x larger_circle_0279, continued over 4 solver runs;
- 4x normal_right_triangle_0009, continued to 30,000 iterations;
- base_lu normal_left_triangle_0020;
- 0.25x normal_left_triangle_0016, relaxed template, started from base_lu;
- base_fo normal_diamond_0001.

*solved*: 3 cases solved again from rest with the production settings, at 0.25x and base_lu:
- normal_rectangle_0001;
- normal_right_triangle_0247, blockage 0.3998;
- normal_circle_0001.

**Pass criterion**: arrays and attributes of the archives identical; fields of the cases solved again within 1e-5.

**Result: passed, bit for bit.**
- All 5 archives give back arrays and attributes identical to the released ones (largest difference 0).
- All 6 runs solved again need the same number of iterations (173 to 1,207).
- Their fields and attributes are identical to the released ones (largest difference 0).

## 8. Loose stopping criterion (2026-10-16/17)

**What**: data root `loose-audit` (job names `m2l_...`, set `validation-loose-audit` in every provenance).

What the stopping criterion of common practice does to the labels and to the coarse-to-fine gap:
- residualControl p 1e-2, U 1e-3;
- template `second-order-loose`, which differs from `second-order` only there.

Cases: 105 test cases, 7 per size and family, drawn from 7 bins of the iterations of their converged 0.25 mm run,
plus 8 ood_blockage cases (`validation/select_loose_audit.py`, which records each case's inclusion probability).

Runs:
- 16x_loose, from the 1 mm solution, as the production starts;
- 16x_loose_scratch and base_lu_loose_scratch, from rest.

They are compared with the converged production runs (`validation/compare_loose.py`, weighted by 1 / inclusion
probability).

**Result** (test cases, median; 90th percentile of |·| in brackets):

| Run | Stops after | Converged run needs | RMS change of u_x, % of U_in | Drag | Recirculation length |
|---|---|---|---|---|---|
| base_lu_loose_scratch | 233 | 571 | 0.55 | +0.01 % (0.0) | 0.00 % (0.1) |
| 16x_loose | 8 | 3,353 | 0.32 | +1.33 % (2.3) | −1.53 % (2.9) |
| 16x_loose_scratch | 355 | 3,353 | 18.67 | +0.50 % (7.8) | −32.6 % (67.2) |

Coarse-to-fine gap (relative RMS of u_x, 1 mm upsampled bicubically to 0.25 mm):
- converged: 0.47 %;
- 16x_loose against the converged 1 mm: 0.36 %;
- both levels loose from rest: 17.1 %.

The loose criterion takes the label away from the converged solution, in a direction set by the start. From the 1 mm
solution the label stays close to that solution, and the gap looks smaller than it is. From rest the recirculation is
too short, and the gap is overstated 36 times.

## 9. DFG 2D-1 benchmark (2026-10-18)

**What**: Schäfer & Turek (1996), steady flow past a cylinder in a channel at Re = 20, solved with the settings of the
production. Data root `dfg-benchmark`, job names `m2g_...`, code in `validation/dfg/` (`run_dfg.sh`, `dfg.py compare`).

- Unchanged from the production: fvSchemes and fvSolution of the template `second-order` (SIMPLEC, linearUpwind,
  residualControl 1e-6) and the gmsh settings (Delaunay quads by recombination, one cell layer in z).
- New: the geometry (channel 2.2 × 0.41 m, cylinder D = 0.1 m at (0.2, 0.2) as an exact circle and a patch of its own),
  the viscosity (1e-3 m²/s) and the inlet.
- Inlet: a fixedValue list, the parabola u = 4 U_m y (H − y) / H² with U_m = 0.3 m/s evaluated at the inlet face centres
  (the values an exprFixedValue condition would give).
- Three uniform meshes, D/20, D/40 and D/80. The two finer runs start from the coarse solution (mapFields), as the
  production starts its fine runs.

Quantities:
- cD and cL from the forces function object, per unit depth;
- Δp = p(0.15, 0.2) − p(0.25, 0.2), from the pressure of the cylinder faces, interpolated in the angle;
- La from the zero of u_x on the line y = 0.2 behind the cylinder;
- mesh study with `tools/mesh_study.py` (`study`).

**Result**: every run stopped at residuals ≤ 1e-6 (largest u_y 9.99e-7).

| Mesh | Cells | Iterations | Solver time | cD | cL | Δp | La (m) |
|---|---|---|---|---|---|---|---|
| D/20 | 35,965 | 410 | 54 s | 5.5848 | −0.0013 | 0.1138 | 0.08428 |
| D/40 | 143,907 | 715 | 8 min | 5.5857 | 0.0100 | 0.1157 | 0.08433 |
| D/80 | 575,745 | 1,977 | 1.7 h | 5.5827 | 0.0116 | 0.1166 | 0.08446 |
| Extrapolated | | | | – | 0.0118 | 0.1175 | – |
| Reference interval | | | | 5.57–5.59 | 0.0104–0.0110 | 0.1172–0.1176 | 0.0842–0.0852 |

- cD and La: the finest mesh is inside the interval. Their changes between the meshes are not monotonic and smaller
  than the intervals (0.05 % and 0.2 %), so they are not extrapolated.
- Δp: the finest mesh is 0.8 % low. The extrapolated value (observed order 1.0, GCI 1.0 %) is inside.
- cL: outside. The finest mesh gives 0.0116, 9 % above the reference value 0.010619, and the extrapolation (order
  2.8) moves further away. The lift is 0.2 % of the drag here: the difference of 0.001 in cL is 0.02 % of cD. Reported
  as it is, nothing tuned.

Iteration error of cL (job 3197523): the forces were evaluated again on the time step kept before the last one
(`simpleFoam -postProcess -time 1900` on D/80, `-time 700` on D/40).

| Mesh | Steps | Change of cL | u_y residual | cL pressure part | cL viscous part |
|---|---|---|---|---|---|
| D/40 | 700 → 715 | +0.004 % | 1.12e-6 → 9.95e-7 | −0.0035 | 0.0135 |
| D/80 | 1900 → 1977 | −0.019 % | 1.13e-6 → 9.99e-7 | 0.0029 | 0.0087 |

cD does not change in its seventh digit on either mesh. If the D/80 run converges at the rate of its residual (0.888
per 77 iterations), the change still to come is about 1.7e-5 in cL, 0.15 % of its value, far below the 9 % to the
reference. The lift is mostly viscous here, and its split between pressure and friction changes with the mesh (the
pressure part changes sign between D/40 and D/80). The cL deviation comes from the mesh, not from the iteration.

## 10. Starting fields (2026-10-08 and 2026-10-18)

**What**: the same meshes solved from the starting fields of the production and from rest, compared with
`validation/compare_init.py`: RMS of the field difference over the RMS of the field (fluid points), and
|difference| / |value| of every attribute (lift relative to the drag).
- **2 mm** (data root `init-check-2mm`): 27 2 mm runs converged only when started from their 1 mm solution with
  SIMPLE and under-relaxation (template `second-order-relaxed`, decisions 2026-10-06/07). 4 of them were solved again
  with the same template from rest (`0.25x_scratch`, cap 50,000):
  - normal_left_triangle_0271, which was stuck at 3.4e-2 with SIMPLEC;
  - larger_left_triangle_0296, the highest blockage (0.399);
  - larger_circle_0279, the only circle;
  - normal_left_triangle_0107, which converged after only 575 iterations.
- **0.5 and 0.25 mm** (data root `init-check`): 6 cases, solved from the 1 mm solution as in the production and from
  rest (`4x_scratch`, `16x_scratch`). 4 are nearly symmetric, centred obstacles near the blockage limit and 2 are
  ordinary: larger_rectangle_0061, larger_diamond_0069, larger_circle_0293, normal_rectangle_0166,
  normal_left_triangle_0001 and smaller_right_triangle_0001.

**Pass criterion**: differences of the order of the iteration error at the 10⁻⁶ stop, far below the label
uncertainty.

**Result: passed.**

| Level | Cases | Fields, largest RMS difference | Attributes, largest difference | Iterations, mapped vs from rest |
|---|---|---|---|---|
| 2 mm (2026-10-08) | 4 | 0.0067 % (u_y) | 0.0008 % (base pressure) | 575–6,157 vs 6,909–14,576 |
| 0.5 mm (job 3197680) | 6 | 0.021 % (u_y) | 0.033 % (base pressure) | 760–2,385 vs 1,217–4,782 |
| 0.25 mm (jobs 3197680, 3197710) | 6 | 0.10 % (u_y) | 0.15 % (base pressure) | 2,466–8,142 vs 3,966–15,885 |

- **Iteration error:** the differences grow on the finer meshes, as the iteration error left at the 10⁻⁶ stop does.
  - All twelve 0.25 mm runs stopped on u_y.
  - The two cases with the largest differences are the two whose run from rest stopped with a higher u_x residual
    (7–9 × 10⁻⁷, against about 3 × 10⁻⁷).
- **Against the label uncertainty:** the largest attribute difference of each 0.25 mm case is 0.3–8 % of the label
  uncertainty of that case and attribute (section 11).
- **The sixth 0.25 mm run from rest** (normal_left_triangle_0001) had been cancelled at a time limit on 2026-10-06.
  It was continued to convergence on 2026-10-18 (job 3197682, 15,885 iterations).
- **Printouts:** `init-check/compare_init_4x_16x-3197680.txt` and `init-check/compare_init_16x-3197710.txt`.

The starting fields change the iterations, not the solution.

## 11. Discretisation error and label uncertainty (2026-10-18)

**What**:
- The uncertainty U of every 0.25 mm label, with `tools/uncertainty.py`: the least-squares procedure of Eça &
  Hoekstra (2014) over the four second-order levels.
- The error of every cheaper level against the 0.25 mm label, on the same scales, with `validation/level_errors.py`
  (`python validation/level_errors.py <dataset> <label_uncertainty.parquet>`).

**Result** (median over the 4,500 cases, %; the full table with the 90th percentiles is printed by the script):

| Attribute | 2 mm | 1 mm, first order | 1 mm | 0.5 mm | U of the label |
|---|---|---|---|---|---|
| drag | 1.10 | 3.93 | 0.76 | 0.33 | 1.45 |
| recirculation length | 2.80 | 11.10 | 1.62 | 0.64 | 2.41 |
| dissipation | 4.68 | 4.56 | 1.69 | 0.53 | 0.49 |
| mean vorticity | 0.65 | 2.31 | 0.33 | 0.12 | 0.34 |
| lift (of the drag) | 0.66 | 1.34 | 0.46 | 0.23 | 0.97 |

- **The second-order levels converge towards the label.**
- **The first-order 1 mm run differs where the flow separates:** recirculation length 11 % and drag 3.9 %.
- **Several cheap levels are already within the label uncertainty.** The 1 mm drag error is about half the U of the
  drag label.
- **The three-level GCI** of `mesh_study.parquet` is undefined for 34 % of the drag labels (convergence not
  monotonic). The meshes are independent unstructured meshes, not nested refinements, which is why the label
  uncertainty uses the Eça–Hoekstra procedure.

## 12. Arrays against the solver mesh (2026-10-18)

**What**:
- Every attribute that can be computed from the fields was computed from the arrays of every level on its own grid,
  with `tools/flow_attributes.py --coarse <run> --fine 16x --output <folder>` (40b637f; jobs 3197503–06, all 4,500
  cases). It was compared with the OpenFOAM value on the simulation mesh by `validation/array_vs_openfoam.py`.
- This replaces a comparison with ParaView's Resample To Image, which wraps the same vtkResampleToImage as
  `extraction/resample_vtk.py`.
- Output: `audit/array-check-2026-10-18`.

**Result** (median |array − OpenFOAM| / |OpenFOAM| over the cases, %):

| Attribute | 2 mm | 1 mm FO | 1 mm | 0.5 mm | 0.25 mm |
|---|---|---|---|---|---|
| recirculation length | 2.45 | 1.41 | 1.27 | 0.62 | 0.30 |
| recirculation area | 4.99 | 1.67 | 1.57 | 0.47 | 0.14 |
| Δp inlet – outlet | 1.52 | 1.41 | 1.46 | 1.40 | 1.36 |
| dissipation | 13.01 | 7.17 | 6.80 | 3.44 | 1.74 |
| mean vorticity | 4.35 | 2.25 | 2.14 | 1.05 | 0.52 |
| peak vorticity | 35.7 | 31.3 | 29.2 | 24.6 | 21.5 |

- **Gradient quantities lose the most,** and the loss halves with every refinement. It is a systematic underestimate.
- **The inlet – outlet Δp of the arrays is about 1.4 % low at every level.** The end columns of the grid differ from
  the patch faces, which include the singular inlet corners.
- **Applied to the array values of the four levels** (`tools/uncertainty.py --arrays`), the uncertainty procedure
  gives a median U of the 0.25 mm array values of 2.94 % for dissipation and 0.90 % for mean vorticity. It is about
  the same as on the mesh for recirculation, flow split and Δp.
- **This measures how well the arrays represent the attributes.** It is not the uncertainty of a comparison of two
  arrays on the same grid, where the resampling loss is common to both.
