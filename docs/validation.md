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
- `checksums.md5` lists all 36 files, and every MD5 is correct: 104,026,682,495 bytes.
- MD5 of `checksums.md5` itself: 817adefb132205f30bf633710f554dc2.

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

## Planned

- VTK resampling (`extraction/resample_vtk.py`) against ParaView's Resample To Image on the same OpenFOAM results.
- DFG 2D-1 benchmark (Schäfer & Turek 1996, confined cylinder, Re = 20) with the production settings on three meshes:
  drag coefficient 5.57–5.59, lift coefficient 0.0104–0.0110, pressure difference 0.1172–0.1176, recirculation
  length 0.0842–0.0852.
- Initialization check (running): 6 cases (4 nearly symmetric and centred obstacles near the blockage limit, 2
  ordinary ones) solved on the 0.5 and 0.25 mm meshes from the 1 mm solution (as in the production) and from rest;
  arrays and attributes compared with `validation/compare_init.py`.
- 2 mm initialization check (running, data root `init-check-2mm`): 27 2 mm runs converged only when started from
  their 1 mm solution with SIMPLE and under-relaxation (template `second-order-relaxed`, decisions 2026-10-06/07).
  4 of them are solved again with the same template from rest (`0.25x_scratch`, cap 50,000):
  - normal_left_triangle_0271, which was stuck at 3.4e-2 with SIMPLEC;
  - larger_left_triangle_0296, the highest blockage (0.399);
  - larger_circle_0279, the only circle;
  - normal_left_triangle_0107, which converged after only 575 iterations.

  Both are compared with `python validation/compare_init.py <root> 0.25x`. Equal fields would show that the solution
  does not depend on the starting point.

  Result (2026-10-08): all 4 converged from rest, after 6,909 to 14,576 iterations (575 to 6,157 from the 1 mm
  solution). RMS difference of the fields at most 0.0010 % (u_x), 0.0067 % (u_y), 0.0005 % (p); attributes at most
  0.0008 % (base pressure). The solution does not depend on the starting point.
