# Optional upstream conversion bridge

`t3d.native_voxelize.mesh_to_voxels_native` calls the upstream O-Voxel CPU
converter with explicit CPU vertices/faces, grid dimensions and AABB. It is
reviewed against TRELLIS.2 commit `75fbf0183001ed9876c8dbb35de6b68552ee08bd`:
[Python wrapper](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/o-voxel/o_voxel/convert/flexible_dual_grid.py),
[native converter](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/o-voxel/src/convert/flexible_dual_grid.cpp).

Source inspection shows the wrapper subtracts the AABB minimum before calling
the CPU implementation. That implementation solves in physical coordinates and
returns these physical positions. They are not the local [0,1] offsets used by
this project's schema. `from_native_grid` adapts them as
`local = physical / native_voxel_size - coords`, with float32 voxel-size semantics
matching the wrapper, then canonicalizes all rows together. It rejects materially
out-of-cell positions and records/clamps only declared float32 roundoff. Two CPU
unit tests cover nonzero origin/anisotropic cells, unit mismatch and roundoff.

This is a bridge implementation, not measured upstream execution. The optional
`o_voxel` package requires a CUDA-enabled build even though this converter runs
on CPU. It is not installed locally and is not included in active Kaggle version
16. A separate pinned native build and source-reconstruction benchmark must
validate behavior/performance before changing preprocessing defaults. In
particular, this does not establish arbitrary topology, resolve multiple edge
crossings, repair the two real fixtures, or meet the <10-second conversion goal.
The current custom reference converter remains the default.
