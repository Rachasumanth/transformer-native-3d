# CuMesh cleanup and UV preparation

`t3d.mesh_postprocess.clean_and_unwrap` takes CUDA vertices/faces and a target
face count, returning `PreparedMesh` with geometry, UVs, normals, a cleaned-mesh
vertex map, reference BVH and per-stage counts. The vertex map indexes cleaned
geometry, not source voxel rows: changed geometry must use BVH reprojection for
later material baking and rig transfer.

The standard branch uses hole perimeter 0.03, an initial 3× target simplification,
duplicate/nonmanifold/component cleanup (minimum component area 1e-5), hole filling,
final target simplification and the repeated cleanup pass, then unified winding.
UV charts use a 90-degree cone threshold, one global iteration and smoothing 1.
CuMesh runs geometry operations on CUDA but its atlas packing uses CPU Xatlas.

The adapter follows the public upstream APIs:
[CuMesh](https://github.com/JeffreyXiang/CuMesh) and
[TRELLIS.2 postprocessing](https://github.com/microsoft/TRELLIS.2/blob/main/o-voxel/o_voxel/postprocess.py).
The Kaggle integration probe pins CuMesh commit
`12289e1062f0603f2f0d0771b02e1395d247f26f`, builds its CUDA extension for T4
(sm_75) with two compile jobs, then processes a cube extracted from raw voxels.
Build logs and probe reports are retained as notebook outputs. Version 15's
source-matched standard-branch probe passed on Tesla T4: 300 input faces became
186 faces, with 148 UV vertices, finite UVs/normals and one closed component after
position welding. Strict remeshing and subsequent changes remain unverified.

The strict remeshing branch is implemented but pending execution: pass
`remesh=True` with `aabb` and `grid_size`. It uses one-cell narrow-band dual
contouring, 0.9 projection to the reference BVH, and final simplification.
BVH-projected sparse PBR baking and GLB packaging now have separate implementations
awaiting end-to-end GPU verification. Generated rig decoding/attachment remains
unimplemented. Cleanup may remove small components and change topology
as specified by the plan; this is not proof of arbitrary-source reconstruction
fidelity. Production performance and broad real-asset quality remain unverified.

The next notebook submission uses `tools/build_cuda_dependency.py` to retain
built wheels and runtime/ABI/hash receipts under `dependency-wheels` in its
outputs. Wheels target T4 sm_75 and should only be reused with a compatible
runtime; the receipt records the build environment. The current version 15
predates this artifact-retention change.

Future runs may attach the private `transformer-native-3d-cuda-wheels` dataset
with `<dependency>/receipt.json` plus its named wheel. Reuse checks source
URL/commit, Python major/minor, exact PyTorch/CUDA versions, C++ ABI and sm_75,
then hashes the payload before installation. Missing/incompatible receipts cause
a source build; corrupted payloads fail. This path has local eligibility tests
but has not yet installed/imported a real cached wheel in Kaggle.

`python -m tools.prepare_cuda_cache OUTPUT/dependency-wheels CACHE_DIR` prepares
flat dataset files (`CuMesh.receipt.json`, etc., plus their named wheels), checks
pinned source identities and hashes, and refuses mismatched runtime receipts.
Use `--allow-partial` only to retain successfully built dependencies from an
incomplete run. Both nested notebook receipts and flat dataset receipts are
supported by the reuse helper. Real dataset creation/reuse is still pending.
