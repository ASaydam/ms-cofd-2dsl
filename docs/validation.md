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

## Planned

- VTK resampling (`extraction/resample_vtk.py`) against ParaView's Resample To Image on the same OpenFOAM results.
- DFG 2D-1 benchmark (Schäfer & Turek 1996, confined cylinder, Re = 20) with the production settings on three meshes:
  drag coefficient 5.57–5.59, lift coefficient 0.0104–0.0110, pressure difference 0.1172–0.1176, recirculation
  length 0.0842–0.0852.
- Pilot of the 360 × 120 mm domain: 45 geometries (the largest blockage, rotation and downstream shift of every size
  and family), all five runs; convergence, cost, memory, reverse flow at the outlet, the whole chain to the archive.
