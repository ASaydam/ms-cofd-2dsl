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
