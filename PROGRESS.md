# Implementation evidence

The authoritative specification is Part 2 of the supplied DOCX, extracted into
`plan.md` without rewriting its requirements. `research.md` preserves both parts
and the bibliography. Inline equations are preserved as original PNG images.
Regenerate with `python tools/extract_blueprint.py`.

## Current status

1. Dataset preprocessing: mesh/skeleton scale normalization implemented with an
   invertible shared transform. Exact axial quarter-turn augmentation and CPU
   sharp-edge conditioning sampling are implemented. Dataset acquisition, VLM
   filtering, pose network, and integration into a dataset pipeline remain pending.
2. Dual-track primitives: validated Track A tensor schema implemented, including
   local dual vertices, boolean edge flags, optional split weights, six-channel
   PBR, six-channel skeleton fields, and four-channel skin embeddings. Producing
   dual vertices from supplied Hermite data via a bounded QEF is implemented.
   A CPU reference mesh/grid intersection converter with boundary planes is now
   implemented. CPU source-surface PBR factor/texture sampling is implemented;
   skeleton-field construction, deterministic joint recovery from generated
   fields, source skin-weight transfer, and weighted aggregation of supplied
   joint embeddings are implemented. A three-asset
   SkinAE optimization pilot exists; cross-asset quality remains weak. Core
   glTF animation sampling and multi-pose skin evaluation are implemented;
   source-clip retargeting onto the v13 export is implemented, exported
   animation tracks round-trip, posed error is SkinAE-dominated, and a top-2
   influence limit lowers the measured v13 articulated error by 39%;
   Track B now has a CPU reference BPT tokenizer/detokenizer. Dataset-scale
   topology/entropy evaluation and autoregressive training remain pending.
3. Serialization/container: Morton and transposed Morton uint64 keys, decoding,
   and order/inverse mappings implemented. Experimental raw profile 3 stores
   voxel attributes and supplied BPT tokens with a SHA-256 trailer; profile 1
   remains readable. Hilbert variants are implemented. Latent profile 4 stores
   shared-support latents, checkpoint identities, joint hierarchy, and a
   checksum; profile 2 remains readable. Separate skeleton support and streaming
   remain pending.
4. Compression: PyTorch sparse resampling shortcuts, reference learned residual
   paths, occupancy heads and reference ConvNeXt sparse blocks implemented;
   version 3 passed CUDA validation. Composed shape/material VAE architecture and
   tensor losses passed CUDA validation and a single-fixture optimization probe. Optional synthetic/teacher-connectivity rendering supervision has CUDA execution
   evidence, but the one-train/one-validation-asset pilot still reconstructs poorly.
   Production training, research-scale SkinAE, and trained generative models
   remain pending.
5. Generative models: rectified-flow DiT foundations implemented with local
   serialized attention, 3D RoPE, adaLN-Zero and conditioning. Occupancy/rig
   stages with separate skeleton support, trained stage checkpoints, dataset
   training and DPO remain pending. A point-cloud-conditioned Track B causal model now exists; its dedicated Kaggle T4 optimization probe passed. Three-stage latent orchestration is
   implemented and has passed source-matched Kaggle CUDA execution tests.
6. CPU reference quad extraction and component orientation repair implemented.
   PyTorch extraction passed CUDA parity and synthetic sheet timing checks.
   Standard CuMesh cleanup/UV preparation passed the version-15 cube probe.
   PBR baking and raw-container GLB export have synthetic-cube GPU integration
   evidence. Strict remeshing produced an extra component with reversed winding;
   broad workload quality and custom hash kernel remain pending. A
   source-matched Kaggle T4 run now exercises the full synthetic skinned raw
   export path; it does not establish real-asset quality. The direct
   low-level GLB skin packer now round-trips three source rigs and passes the
   Khronos validator; this does not complete generated-rig export.
7. Animated single-asset deformation QA now has correspondence-based evidence
   and a validated top-2 influence-rewriting path; rendered quality, benchmark
   suites and target attainment remain pending. Unit tests do not
   establish research quality targets.

Run current checks: `.venv/bin/python -m unittest discover -s tests -v`.

Source performance and quality figures remain research targets, not measured
results for this implementation. A two-asset pilot dataset and its small-run
checkpoints are present; the curated research-scale corpus and production-trained
weights are not.

## Part 2 requirements audit

This audit covers all seven numbered steps in Part 2 of the DOCX, represented
verbatim (including equation images) in `plan.md`. The implementation below is
partial against that full scope; the subsections that follow provide narrower
evidence and limitations.

| Plan step | Current evidence | Status against the plan |
| --- | --- | --- |
| 1. Dataset curation, filtering, normalization | Normalization, quarter-turn augmentation, SES sampling, source/license/hash manifests, static and rigged GLB imports, plus asset-disjoint pilot datasets. | Partial. No 300K–500K corpus, multi-view VLM filtering, pose-canonicalization network, or broad repository ingestion. |
| 2. Dual-track primitives | Track A tensor schema, CPU QEF/triangle-cell voxelization, source PBR sampling, articulation preprocessing and deterministic field-to-skeleton recovery; experimental Track B BPT encoder/decoder. Pinned upstream O-Voxel ran on three GLBs at 16³/32³ on Kaggle T4. | Partial. Neither custom nor upstream converter resolves real-asset nonmanifold topology; paired sample shows no consistent native geometry advantage. Production arbitrary-topology fidelity is unproven. Untouched-source BPT rejects fifteen of sixteen screened Khronos/Poly Haven assets at 128 bins; explicit filtering lets two more real meshes encode, with one nonmanifold edge remaining on Food Lime. The corpus is still not representative. A small SkinAE checkpoint exists; the AR model has one-asset CUDA optimization evidence and exact training-asset generation, while held-out BPT conditioning fails. |
| 3. Serialization and binary container | Z/TZ/H/TH mappings and inverse indices; checksummed raw profile 3 and latent profile 4, with backward readers for profiles 1 and 2. Latents include decoder identities and optional joint hierarchy. | Partial. No `.vxz` compatibility, streaming guarantees, GPU ordering kernel, or demonstrated zero-copy attention integration. |
| 4. SC-VAE and SkinAE | Four-level sparse resampling shortcuts, occupancy heads, reference sparse blocks, composed shape/material VAE, tensor losses, small CUDA optimization/resume probes, and three-fold asset-held-out SkinAE evaluation. | Partial. No optimized Triton sparse convolution, production-scale 512³–1536³ training, thin-feature/sharp-edge fidelity, complete rendered depth/normal/color and LPIPS/SSIM supervision, or research-scale SkinAE. Optional rendering supervision has narrow CUDA evidence but did not improve the held-out pilot. The source set is small and no deformation benchmark establishes model quality. |
| 5. Transformer and training protocol | CUDA-tested rectified-flow DiT and three-stage occupancy/geometry/material orchestration; source-matched v12 bundle has 88 passing tests on dual Tesla T4, no skips. | Partial foundation only. No conditioning encoder, image/text-aligned corpus, separate shape/skeleton occupancy branches/adapters, trained generators, 4B model or prescribed distributed training, Track B corpus training/DPO, or quality evidence. A point-conditioned causal AR baseline was added; its one-asset CUDA probe passed; generation reproduces its training asset but fails held-out conditioning. |
| 6. CUDA decoding and production GLB pipeline | CPU extraction/orientation; CUDA extraction parity and sub-100 ms synthetic sheet timings; standard CuMesh cleanup/UVs and raw export have synthetic-cube CUDA evidence. Low-level skinned GLB packing round-trips RiggedSimple, CesiumMan and Fox; each output passes Khronos glTF Validator with 0 errors, warnings or infos. Synthetic exact/noisy fields recover a joint hierarchy. `export_raw` field-to-rig/SkinAE decoding passed on T4, and a synthetic skinned cube passed extraction, CuMesh, PBR baking, GLB reload and Khronos validation. V9/v10 export the real CesiumMan base-color map through direct and `.t3d` CLI paths. V13 verifies the map at 128³ with a 2048² atlas. Animation retargeting writes LINEAR TRS clips into packed GLBs and round-trips through the importer on three source rigs and five clips; limiting SkinAE weights to two influences lowers measured v13 retargeting error by 39%. | Partial. No custom hash kernel, generated model prediction tests or real generated-asset rig quality, broad real-asset latency/quality results, viewer deformation or Khronos validation of the animated output, rendered material-fidelity evaluation, or production validation across a representative asset set. Standard v13 geometry has zero boundary but 16 nonmanifold edges; strict mode has one boundary and nine nonmanifold edges. |
| 7. Validation and QA | Point-to-triangle surface metric utility, edge-incidence diagnostics, serialization locality fixture, small asset pilots, explicit quantized-face cleanup metrics, fixture-level CR/PTME/AVD metrics, five sampled frames per clip for three rigged assets, three-fold SkinAE source holdout, and source-to-output linear PBR comparisons after UV atlas repacking at 4,096 points across five seeds. | Partial and currently failing. No Dora-Bench/Sketchfab-90 tiered targets, representative-corpus PTME/entropy or broad artist-mesh BPT quality, comparable subsequence fragmentation baseline, HDRI relighting metrics, UniRig/Anymate deformation-quality suite, or target attainment. The 32³ audit finds substantial color loss; v13 128³/2048² cuts mean base-color RMSE to 0.122 / 0.095 / 0.101, but this remains an attribute-space metric, not rendered fidelity. Animation sampling verifies execution and stable triangle incidence. A CesiumMan-to-v13 retargeting audit measures posed deformation: source-transferred oracle weights match the posed source within 0.00052 mean canonical distance (F-score 1.0), while exported SkinAE weights give 0.0179 (F-score 0.60), so weight error dominates articulated deformation. Limiting those weights to two influences cuts the measured error to 0.0109 and raises F-score to 0.78 without invalid animation round-trips; top-1 reaches 0.0098/0.84 but produces a degenerate reloaded triangle. Exported animation tracks reload within 3.3e-6 of in-memory vertices on that export and within 5.4e-7 on same-topology round-trip rigs across three assets and five clips. This is not a viewer or benchmark result. Early predicted pilot reconstruction had F-score 0 at tolerance 0.01; later small runs remain poor (v20/v24 held-out original-GLB mean sampled F-scores across three seeds about 0.1495/0.1355). These small audits do not meet the research benchmarks. |

Version 12 bundle SHA256 is `b82f78558a53d5fb3d99cee776185b6d6ad1ea9c1261e1c6ff633728a9dafc4e`.
`artifacts/kaggle-v12/gpu-report.json` matches that source hash and records two
Tesla T4 GPUs with CUDA 12.8. `artifacts/kaggle-v12/unit-tests.log` records all
88 tests passing in 10.454 seconds without skips. This verifies execution of the
current tests on Kaggle, not completion of Step 5's training protocol or the
end-to-end plan.

## Sharp-edge sampling evidence

`t3d/sampling.py` implements interior-edge dihedral detection, unique sharp
vertices, deterministic FPS, edge interpolation, and separate area-uniform surface
samples with unit face normals. It follows equations 4–7 of the
[Dora paper](https://arxiv.org/html/2412.17808v1#S3.SS2.SSS1).
No detected sharp edges yields an empty salient array, per equation 7.
Boundary and non-manifold edges are excluded from the two-face dihedral test.
Face winding must be consistent; degenerate triangles are rejected.
Sharp vertex normals use a reproducible incident face, avoiding smoothing across
the crease. Interpolation distributes samples evenly by edge count, with integer
remainders assigned in sorted edge order. FPS is a CPU reference with O(N*K)
runtime and O(N) workspace; production dataset throughput is unmeasured.

Verification: 10 unit tests pass, including crease locations, smooth-surface
behavior, exact sample counts, unit normals, seed reproducibility, FPS selection,
area-proportional sampling and barycentric centroids, and rotation inverses.

## Track A schema and ordering evidence

`t3d/schema.py` validates sparse coordinate uniqueness and bounds, shape and range
of geometry/PBR channels, and finite optional rig attributes. PBR channel order is
RGB, metallic, roughness, alpha. Skeleton/skin fields are accepted independently
to permit intermediate pipeline stages. It copies inputs and marks owned arrays
read-only; callers should not deliberately reset NumPy write flags.
World vertices use `aabb_min + (coords + local_dual_vertex) * voxel_size`.
Canonical sorting applies one permutation to every present voxel attribute.

`t3d/serialization.py` uses x/y/z bit lanes 0/1/2 and exchanges x/y for TZ;
batch prefixes occupy bits above the spatial code. Depth is 1–21; the schema's
uint16 coordinate storage allows axis sizes through 65536. These explicit local
conventions are not a claim of upstream `.vxz` compatibility.

Verification: all 17 tests pass. New tests exhaustively check an 8³ grid
bijection, known Morton bit patterns, the maximum uint64 batched key, inverse
permutations, duplicate rejection, empty grids, invalid tensor rejection, input
ownership, world positions, and preservation of all attribute rows during sort.

## QEF fitting evidence

`t3d/qef.py:solve_qef` consumes local intersection positions and normals and
solves a weighted plane-fitting objective with optional boundary planes and a
mass-point regularizer. It enumerates all 27 cube active sets, solving free axes
with least squares; this handles rank deficiency without simply clipping an
unconstrained fit. Normal magnitude and sign do not change the plane objective.
The explicit local objective is documented in the module; exact numerical parity
with TRELLIS.2's native solver has not been established.

Defaults are face=1, boundary=1, regularization=0.1, verified against the
[upstream wrapper](https://github.com/microsoft/TRELLIS.2/blob/main/o-voxel/o_voxel/convert/flexible_dual_grid.py).
The extracted source's `0.119` joins the value `0.1` with citation 19; the source
extraction remains unchanged. Boundary Hermite planes must currently be supplied
by the caller when using the standalone solver; `mesh_to_voxels` now discovers
mesh boundary edges and builds these constraints automatically.

All 22 tests pass, including analytic plane/crease/corner fits, constrained
optimality in a case where clipping fails, boundary weights, normal rescaling,
input order, and malformed-input rejection. This CPU solver has not been timed
at production asset scale and does not satisfy the full mesh conversion gate.

## Direct mesh conversion evidence

`t3d/voxelize.py:mesh_to_voxels` clips triangles directly against candidate voxel
boxes, fits local dual vertices with the QEF solver, and detects three primal
edge crossings per cell. The x-edge is at the cell's upper y/z coordinates,
with corresponding cyclic definitions for y/z, so positive transverse neighbors
share each marked edge. Original mesh boundary edges contribute perpendicular
in-surface plane constraints. The converter returns a canonical `VoxelAsset`;
it does not construct an SDF, infer inside/outside, or discard components.

All 27 tests pass. Converter tests cover an analytic open sheet and its edge
flags, a subcell triangle without primal-edge crossings, two disconnected
internal surfaces, reversed winding, anisotropic bounds, and bounds rejection.

Remaining limitations: candidate traversal scans each triangle bounding box;
large oblique triangles are expensive. Grid-coincident coplanar edges have no
unique crossing and are left unmarked. Boolean flags collapse multiple edge
crossings and one dual vertex fits all patches in a voxel. A subcell patch can
be retained as an active voxel yet produce no quad on extraction. Robust
grid-degeneracy handling, reconstruction QA, high-resolution throughput, and
upstream native-solver parity remain unverified. These tests do not establish
the blueprint's arbitrary-topology reconstruction or <10-second conversion gate.

## Mesh extraction and round-trip evidence

`t3d/extraction.py:voxels_to_mesh` maps marked primal edges to four neighboring
dual vertices, skips incomplete quads, and triangulates them. Weight products
choose the diagonal when supplied; otherwise it maximizes the minimum interior
triangle angle. The angle criterion follows the blueprint description and differs
from the current upstream wrapper's normal-dot heuristic. Vertex indices retain
voxel attribute correspondence. Raw winding is provisional because boolean flags
do not encode crossing direction; `orient_faces` now performs a separate repair.

All 32 tests pass. Extraction checks all three axes, missing neighbors, weighted
diagonals, empty grids, and end-to-end open-sheet and cube conversion. The sheet
retains its plane and total area; the cube has two incident faces per edge and
Euler characteristic 2. These limited analytic cases do not prove arbitrary
topology or material reconstruction quality.

Run `python -m tools.roundtrip_demo` to regenerate `artifacts/roundtrip/cube.obj`
and `cube.json`. The 8³ example produces 152 active voxels and 300 triangles,
zero boundary/nonmanifold edges by incidence count, and original [0.2,0.8]³
bounds. Timings in the JSON measure this small CPU example only; the <100 ms
CUDA extraction gate remains unverified.

## Orientation repair evidence

`t3d/topology.py:orient_faces` propagates face-flip constraints across shared
edges, preserving all vertices, face identities, and disconnected components.
It rejects duplicate faces, edges with more than two incident faces, and
contradictory orientation constraints instead of silently deleting geometry.
Open components retain their first face's winding. Optional `outward=True`
orients each closed component to positive signed volume, computed relative to
a local origin for translation stability. This is an independent-shell policy;
nested cavity semantics and self-intersections are not analyzed. Vertex-only
nonmanifold junctions are not repaired by this edge-based procedure.

All 37 tests pass, including mixed winding, seed preservation on open sheets,
disconnected shells, large translations, idempotence, input rejection, and a
non-orientable triangulated strip. The round-trip cube test verifies closedness
and volume 0.216. The regenerated 8³ demo repaired 150 of 300 face windings;
`artifacts/roundtrip/cube.obj` now has consistent outward winding for this cube,
and the JSON records the component audit and separate orientation timing.

## Raw container evidence

`t3d/container.py` implements in-memory encoding/decoding and path read/write
with the blueprint's 128-byte version-2 header, 64-byte-aligned coordinate/raw
attribute/BPT blocks, and explicit raw profile metadata. All 41 tests pass,
including all-field round trips, geometry-only and empty grids, supplied token
boundaries, malformed headers/slices, truncation, trailing bytes, and float16
overflow. The demo writes and reloads `cube.t3d`; maximum world-vertex error
from float16 storage is about 2.04e-5 for this fixture. Latent storage, skeleton
hierarchy, streaming, integrity checks, and `.vxz` compatibility remain pending.

## Four-curve serialization evidence

Z/TZ/H/TH order/inverse mappings and Hilbert decoding are implemented in
`t3d/serialization.py`. Hilbert delegates to the CPU `hilbertcurve` 2.x library
([upstream API](https://github.com/galtay/hilbertcurve)); transposition exchanges
x/y. Containers now record curve IDs 0–3 and validate the declared order.
`VoxelAsset.canonical` accepts `curve='h'` with optional transposition.

All 44 tests pass using `.venv/bin/python -m unittest discover -s tests -q`.
The local venv uses system NumPy plus hilbertcurve 2.0.5; the dependency is also
declared in pyproject.toml. New checks exhaustively test 8³ Hilbert bijection and
face adjacency, empty inputs, high-bit batch prefixes, all four inverse mappings,
and container round trips for every curve. `artifacts/serialization/locality.json`
records adjacent-distance measurements on the complete 8³ grid; those results
do not establish sparse asset locality or Transformer performance. GPU ordering
and integration into attention layers remain pending.

## Source material sampling evidence

`t3d/materials.py` projects dual vertices onto the nearest source triangle and
interpolates per-corner UVs with the closest-point barycentrics. It supports
linear base-color factors, sRGB RGB textures decoded before bilinear filtering,
linear alpha, metallic/roughness factors and linear G/B texture channels,
repeat/clamp addressing, and OPAQUE/MASK/BLEND alpha modes. Channel conventions
were checked against the [Khronos material schema](https://github.com/KhronosGroup/glTF/blob/main/specification/2.0/schema/material.pbrMetallicRoughness.schema.json).
Textures must already be decoded float RGB/RGBA arrays in [0,1]. Output voxel RGB
is explicitly linear. Per-corner UVs preserve seams; closest-triangle ties pick
the first source face deterministically.

All 50 tests pass in the project venv. New checks cover face/edge/vertex closest
points, material factors, alpha interpretation, periodic/clamped texture sampling,
color transfer, G/B channel assignment, face selection, validation, and material
container round trips. The cube demo now stores a documented synthetic source
material in cube.t3d; it is not a learned material prediction.

Limitations: closest-surface projection is O(voxels*faces), without a BVH. Texture
files, glTF material import, texture transforms, separate sampler axes, mirrored
repeat, mipmaps, normal maps and emissive extensions are not implemented. One PBR
value per voxel cannot preserve multiple coincident materials. Renderer-based
relighting fidelity and production throughput remain unverified.

## Articulation preprocessing evidence

`t3d/articulation.py` validates acyclic skeleton parent forests with arbitrary
joint ordering and -1 roots. It constructs the blueprint's six-channel
nearest-joint/parent displacement field at voxel centers. Roots point to
 themselves as parent; ties select the lowest joint index. Query chunking bounds
temporary nearest-joint memory, while runtime remains O(voxels*joints).
Four-channel skin embeddings are computed as weighted averages of caller-supplied
joint embeddings; this is the aggregation operation, not a learned SkinAE.
`transfer_skin_weights` interpolates source vertex weights at exact closest
triangle points, and `attach_articulation` adds the resulting fields to an asset.

A separate linear blend skinning utility applies supplied pose-times-inverse-bind
matrices for deformation checks. It does not implement forward kinematics.
All 56 tests pass, including parent cycles/invalid indices, roots, nearest ties,
chunk equivalence, center sampling, source weight transfer, embedding aggregation,
rig attribute storage, and analytic identity/translation/rotation deformation.

SkinAE training, learned joint embeddings and skin-weight decoding, confidence
fields, skeleton voxel support, joint clustering, bone containment, animation
benchmarks, and skeleton hierarchy container storage remain pending. The
blueprint's nearest-joint field is implemented explicitly; this does not claim
full reproduction of [AniGen](https://arxiv.org/html/2604.08746v2), which also uses
confidence-aware fields and separate skeleton support.

## Track B tokenizer evidence

`t3d/bpt.py` implements canonical 128-bin quantization, 8³ coarse block/16³ offset
factorization, highest-degree fan centers, shared block prefixes, separator-free
center/peripheral vocabularies, and a grammar-checking decoder. Exact local
conventions and compatibility limits are documented in `docs/bpt.md`; the
[official BPT repository](https://github.com/Tencent-Hunyuan/bpt) supplies the
research reference, not a checkpoint-compatibility guarantee for this code.

The existing BPT tests cover oriented face preservation, canonical output under
face/vertex permutation, branching nonmanifold fans, malformed token streams,
quantization collapse, and vocabulary bounds. Added a reproducible
`tools/benchmark_bpt_grid.py`; the updated 8192-face planar grid round-trip
preserves all oriented quantized triangles using 15,888 tokens (1.939
tokens/face; 21.55% of raw nine-coordinate-token sequences, 78.45% reduction).
The 2,080 center-token patches remain a BPT-specific statistic. Artifacts are
`artifacts/bpt/grid_tokens.npy` and `grid_report.json`.
The fixture has 2080 patches, so it does not establish the fragmentation target.
Dataset benchmarks, PTME/entropy, artist-mesh topology quality, tokenizer identity
metadata in the container, point-cloud conditioning, AR training and DPO remain
pending. No required research training/quality gate is marked complete.

## Kaggle GPU backend verified

User requested Kaggle notebook GPUs instead of local GPU execution. The private
notebook https://www.kaggle.com/code/sumu143/transformer-native-3d-gpu-bootstrap
completed version 1 with two Tesla T4 devices, PyTorch 2.11.0+cu128 and CUDA 12.8.
The bundled project tests passed. A 30-step synthetic MLP CUDA optimization probe
had finite gradients and reduced loss from 1.07596 to 0.64502. This establishes
remote GPU execution, not training or quality of the planned SC-VAE/DiT models.

The source bundle SHA256 is
`d3ea286b3cfe82e92d51a07049ef7e8acc3ed509301895f62f6c0db98c8308de`.
Downloaded evidence is in `artifacts/kaggle-bootstrap/gpu-report.json`.
`tools/build_kaggle_notebook.py` bundles only allowlisted project code; credentials
are excluded. See `kaggle/README.md` for regeneration, submission, monitoring and
output retrieval. Future GPU experiments should use this Kaggle workflow.

## Sparse compression implementation in progress

`t3d/sparse_compression.py` adds PyTorch CPU/CUDA sparse child packing,
space-to-channel group averaging, channel-to-space duplication, learned linear
residual resamplers, and child occupancy logits with teacher/predicted pruning.
Coordinates include a batch column; empty child octants contribute zeros.
Four stages implement 16× spatial reduction. Decoder outputs sort coordinates
so occupancy masks remain aligned across levels. The learned linear paths are
reference resamplers, not substitutes for the planned ConvNeXt sparse blocks.
ConvNeXt blocks, variational posterior heads, full SC-VAE architecture, reconstruction
losses and asset training remain pending.

Local checks discover 64 tests: 60 pass and 4 PyTorch tests skip because PyTorch
is provided by the Kaggle runtime. Remote validation of the new modules is in
progress; this section does not claim they have passed on CUDA yet.

Kaggle version 2 (bundle SHA256
`c8909eaed490c255e8e56dc1c814a02377e81b643cdb909fe5e395c6229a346e`)
was pushed successfully and last verified QUEUED. Do not restart it merely
because it is queued. A subsequent local correction sorts decoded coordinates
and strengthens the four-stage support test; it is not in version 2. Once that
run is terminal, retrieve its evidence, rebuild the notebook from current source,
and submit the corrected version. The generated notebook currently represents
version 2, while local sparse_compression.py/test_sparse_compression.py are newer.

## Sparse ConvNeXt reference and CUDA failure correction

Kaggle version 2 reached ERROR. Its downloaded `artifacts/kaggle-v2/unit-tests.log`
shows 3 errors in sparse child packing: integer matrix-vector multiplication is
unsupported on CUDA (`addmv_impl_cuda` for Long). This was replaced by integer
elementwise multiplication and reduction, including the new convolution key path.
The decoder coordinate-order correction is now also included in the bundle.

`t3d/sparse_blocks.py` adds a batch-aware 3x3x3 submanifold convolution and a
ConvNeXt-style residual block with one spatial convolution, LayerNorm and wide
pointwise MLP. Tests cover exact neighbor sums, absent cells/batch isolation,
center identity, coordinate permutation and parameter/input gradients.
Local discovery now runs 67 tests (60 pass, 7 skip without PyTorch).

Version 3 was successfully submitted with bundle SHA256
`5304686b55922204827da8d337ea6356a5c5aea9fb4314b99f52c3402a856545`.
CUDA validation remains pending until its current-state output is inspected.
See `docs/sparse-compression.md` for implementation semantics and remaining work.

## VAE composition and remote verification

Kaggle version 3 completed successfully. Downloaded `artifacts/kaggle-v3` contains
the source-matched GPU report and a test log showing all 67 tests pass, with no
skips. This verifies the corrected sparse shortcuts, four-stage support recovery,
occupancy-head gradients, submanifold convolution, and ConvNeXt block on CUDA.

`t3d/sc_vae.py` now composes separate shape/material variational models with
16× encoder/decoder hierarchy, posterior sampling, KL regularization, tensor
attribute reconstruction losses and teacher-mask occupancy supervision. Added
model tests check support alignment, shape backward gradients, posterior sampling,
material loss, parameter separation and decoder allocation limits. Locally there
are 69 tests: 60 pass and 9 skip without PyTorch.

A new `tools/train_compression_probe.py` runs ten updates of each VAE on the cube
fixture and writes component losses/checkpoints. This is a single-fixture execution
probe, not dataset training. Notebook version 4 was submitted with bundle SHA256
`264db579da284d9a4af9f70d7bd1241a77e8afbbda3e510ba576c7f71df5dd7b`.
Its model tests and probe must be verified from terminal status and downloaded
outputs before claiming success. Version 3 evidence covers the sparse building
blocks only; it predates sc_vae.py. Rendering supervision, optimized kernels,
dataset curation/training, checkpoint resume and research benchmarks remain open.

Version 4 subsequently completed successfully. The source-matched downloaded log
in `artifacts/kaggle-v4/unit-tests.log` proves all 69 tests passed with no skips.
`compression-probe.json` records ten finite-gradient CUDA updates on Tesla T4
for each VAE: shape total loss 1.80159 → 1.55788, material 1.01742 → 0.93712.
Model/optimizer checkpoints `shape-vae-probe.pt` and `material-vae-probe.pt` were
downloaded. Attribute, occupancy and KL terms are finite throughout both runs.
This establishes trainable model execution on one analytic fixture only. It does
not establish stochastic-posterior dataset convergence, held-out reconstruction,
rendering fidelity, or the blueprint's compression/quality targets.

## Dataset manifests and resumable training

`t3d/dataset.py` adds versioned prepared-asset manifests with provenance/license
strings, split selection, file hash verification, duplicate-content/ID/path
rejection, and sparse batch construction. `t3d/training.py` adds a training CLI
with stochastic posterior updates, held-out posterior-mean validation, atomic
checkpoints and model/optimizer/RNG/epoch/cursor restoration. Dataset fingerprints
must match on resume. See `docs/training.md` for exact scope and remaining work.
The real training corpus, curation/rendering pipeline and dataset-scale runs are
still pending; no synthetic fixtures are presented as a curated corpus.

Local discovery: 71 tests, 61 pass and 10 PyTorch tests skip. New tests cover
manifest hashes, duplicate-split leakage, relative paths, batching, and stochastic
checkpoint replay. Kaggle version 5 was submitted with source SHA256
`b32f262783cd42d5f2169e6b91bfd1c7b0cdbc8d7a92e1bb0124667d136eb29c`.
The resume test must pass remotely before claiming CUDA replay correctness.

Version 5 reached ERROR with one checkpoint write failure; its log is downloaded
under `artifacts/kaggle-v5`. PyTorch's filename-based zip writer rejected the hidden
temporary checkpoint path. The atomic save now serializes through an open file
handle, flushes/fsyncs it, and replaces the destination after serialization.
Version 6 was submitted with source SHA256
`f499cba6d89c26488abd85c33f7677998806abdb340f644cd30518141d51068e`.
Its checkpoint replay result remains to be inspected; do not use version 5's
failure as evidence that resume has passed.

Version 6 completed successfully. Its source-matched `artifacts/kaggle-v6`
report and log confirm all 71 tests passed on Kaggle with no skips, including
`test_stochastic_resume_matches_next_update`. That test saves after a stochastic
update, executes the next update, restores model/optimizer/CPU+CUDA RNG/cursor,
and reproduces the next loss terms and resulting parameters within the specified
tolerances. The wrong-manifest resume rejection also passed. This is tested on
the synthetic fixture, not evidence of dataset-scale training or resume across
different hardware/runtime configurations.

## Real GLB ingestion evidence

`t3d/import_glb.py` adds static GLB scene/material import and a CLI connecting
normalization, mesh conversion, source-texture PBR projection and raw-container
storage. Supported cases and explicit exclusions are in `docs/import.md`.
Added Trimesh/Pillow dependencies; the notebook builder installs them as well.
Tests cover transformed/mirrored instances, UV convention, textured preparation,
channel preservation and malformed GLB rejection. Local discovery passes 64 tests
with 10 PyTorch tests skipped (74 total). These new importer changes have not yet
been run in a new Kaggle notebook version; version 6 remains the last verified
remote bundle. Builder source is newer than the generated notebook; regenerate
before the next submission.

Downloaded the official Khronos Avocado asset and retained its upstream README
(CC0-1.0, Microsoft). The real textured asset converts from 682 source faces to
498 active voxels at 16³. `artifacts/import/avocado` holds source, prepared file,
hash report and a verified single-record manifest. Real input revealed integer
PBR factors causing in-place texture multiplication casting errors; factor arrays
now use float dtype and a regression check covers the integer-factor case.
This is an integration fixture, not the required curated training corpus.

## Two-real-asset Kaggle pilot

Imported Khronos WaterBottle at 16³: 4510 source triangles → 464 active voxels.
Its upstream README (retained locally) lists CC0-1.0, Microsoft, 2017. Combined
with Avocado (498 voxels), `tools/build_pilot_dataset.py` builds a verified manifest
with Avocado train and WaterBottle validation, preserving upstream notices and
source/prepared hashes. Both files pass hash/PBR validation through AssetDataset.
This is a two-asset integration pilot, not a VLM-curated corpus.

Created private dataset https://www.kaggle.com/datasets/sumu143/transformer-native-3d-pilot
and verified Kaggle dataset status `ready`. The notebook now attaches this source.
`tools/train_corpus_probe.py` performs ten stochastic updates per VAE on Avocado,
saves/restores checkpoints, and records held-out WaterBottle losses before/after.
It records all loss components and actual asset IDs/counts, without assuming
held-out improvement. Notebook version 7 is being submitted with bundle SHA256
`7452d6dece185eb3cc6dad9c15d6ac370b0c8da2294da59f0eac3edbbf2ebac9`.
Remote corpus results and the newly bundled importer tests remain unverified until
terminal status and outputs are inspected. Locally all 64 CPU tests pass; ten
PyTorch tests skip (74 total).

Version 7 completed successfully. Downloaded `artifacts/kaggle-v7` reports match
the submitted source digest and the exact local pilot manifest fingerprint.
All 74 tests pass remotely with no skips, covering the importer changes as well.
Avocado training (498 voxels) and WaterBottle validation (464 voxels) are distinct
records and content hashes. Each VAE performed ten stochastic CUDA updates and
saved/reloaded its model/optimizer/RNG checkpoint. Training total loss changed
shape 1.71813 → 1.50082 and material 1.13370 → 1.10694. Posterior-mean held-out
loss changed shape 1.52572 → 1.50188 and material 1.04884 → 1.04238.
`shape-pilot.pt`, `material-pilot.pt`, `corpus-probe.json`, GPU report and full test
log were downloaded. These are tensor-loss integration measurements at 16³ on
two samples, not rendering quality, corpus generalization, or research targets.

## Reconstruction decoding and measurement work

`t3d/reconstruction.py` decodes posterior means into validated raw voxel assets
with source-teacher or predicted occupancy. Optional material decoding derives
masks from the reconstructed geometry and aligns latent parents, avoiding source
material copying. Empty predicted support is handled explicitly. Rig latent
decoding is not silently approximated.
`t3d/metrics.py` provides explicitly defined area-sampled bidirectional surface
distance/F-score, topology incidence counts, and failure results for missing or
degenerate surfaces. These are integration metrics, not aliases for the named
research evaluation suites. See `docs/reconstruction.md`.

Added analytic surface-distance and decoder tests. Local discovery: 77 tests,
66 pass and 11 skip without PyTorch. Version 8 was pushed with bundle SHA256
`d8f94f690c1f221947a3511ccaf182b5e93875ed71bbcd6653720d4fce77ca8a`.
The pilot will save train/validation x teacher/predicted reconstructions as OBJ
and `.t3d`, and measure against extracted prepared-source meshes. Results remain
pending until the current run reaches terminal status and outputs are inspected.

Independent conversion QA now compares original normalized GLB triangles against
prepared-source extraction, recorded in `artifacts/import/conversion-metrics.json`.
At 16³ and 256 area samples, mean bidirectional distance is 0.000862 for Avocado
and 0.003550 for WaterBottle (unit canonical coordinates; not the named research
MD protocol). However, Avocado extraction has 4 boundary and 28 nonmanifold edges;
WaterBottle has 0 boundary and 112 nonmanifold edges by face incidence count.
These fail topology quality expectations despite small sampled surface distances.
Grid-edge degeneracy/multiple-crossing behavior needs investigation and repair;
neither prepared file should be treated as production-quality geometry. The pilot
training observations remain integration results on these imperfect preparations.

Version 8 completed successfully and downloaded source-matched evidence to
`artifacts/kaggle-v8`. All 77 tests pass with no skips. Four decoded `.t3d`/OBJ
pairs cover train/validation and teacher/predicted support. The metrics expose
poor model quality after ten pilot updates: predicted support extracts only
6 triangles for Avocado and 8 for WaterBottle; both have F-score 0 at tolerance
0.01. Mean bidirectional distances against prepared-source extraction are 0.48465
and 0.40582. Teacher support yields 242/204 triangles, distances 0.02508/0.02637,
and F-scores 0.17079/0.18047. Thus smaller training losses did not establish usable
geometry. `metrics.valid=True` means measurement succeeded, not a quality pass.
Next priorities are source-conversion topology robustness and decoder occupancy/
edge prediction diagnosis, followed by proper training and rendering supervision.
The research completion gates remain open.

## Source topology investigation and ambiguity gate

The original GLBs are open surfaces after exact positional welding: Avocado has
38 boundary edges, WaterBottle 264; neither has a source nonmanifold edge under
that audit. Therefore watertightness is not an appropriate universal acceptance
gate for the blueprint's open-surface requirements. Added crossing-parameter
diagnostics identify a concrete representation limitation: at 16³, 3 Avocado
and 84 WaterBottle primal edges have multiple distinct surface intersections,
which the existing boolean edge flag collapses. Adjacent triangles sharing the
same intersection are deduplicated by parameter tolerance rather than counted
as multiple crossings. `artifacts/import/crossing-audit.json` retains examples.

GLB preparation now rejects multiple crossings before voxel fitting/output by
default. Diagnostic `--allow-multiple-crossings` explicitly permits the lossy case
and records the override/audit in its report. It does not discard intersecting
surfaces or force open meshes watertight. Other grid degeneracies and multiple
patches per cell remain unresolved; this gate does not claim full topology repair.
The retained pilot dataset predates the gate and remains known-imperfect.

Local tests: 80 discovered, 69 pass, 11 skip without PyTorch. New tests distinguish
shared-triangle intersections from two separate sheets and verify strict GLB
import creates no output on ambiguity, while diagnostic import records it.
No GPU modules changed; these CPU changes are not in notebook version 8 yet.
Regenerate the notebook before the next GPU submission.

A 32³ follow-up is retained in `artifacts/import/crossing-audit-32.json`: Avocado
has 11 ambiguous edges and WaterBottle 208. The absolute counts increase as more
edges are sampled; doubling resolution alone did not eliminate ambiguity. This
experiment does not establish an appropriate production resolution. Future work
must check sufficiently resolved/local-refined representations and remaining
multi-patch/coplanar cases, rather than marking the gate solved by a blind grid
increase or deleting crossings.

## Latent and hierarchy container profile

`t3d/latent_container.py` implements experimental profile 2 with the blueprint's
128-byte header, 16× coarse coordinates, dimensioned shape/material/rig float16
streams, curve identifiers, exact decoder checkpoint SHA256 identities, optional
joint positions/parent forest, and BPT payload. It validates geometry, counts,
alignment, ordering, dimensions, finite precision, parent cycles and decoder
identity. Raw profile 1 remains separate. See `docs/container.md` for exact bytes.

Three new CPU tests cover all four curve variants, all streams/hierarchy/tokens,
shape-only/empty payloads, malformed headers/lengths, invalid coordinates,
precision overflow and identity mismatch. Local discovery: 83 tests, 72 pass and
11 skip without PyTorch. `reconstruction.encode_latents` exports actual VAE
posterior means, and the corpus pilot now saves/reloads profile-2 files tied to
SHA256 digests of its actual checkpoints. No rig embeddings are fabricated.

Version 9 was submitted with source SHA256
`19ad9c40b5ea7946edc0444b4c0ca87c8392f03ccbfffd56cde01b1ee49dd5b0`.
Actual GPU latent storage validation and all new remote tests remain pending.
This run also includes the prior CPU crossing-ambiguity gate. The retained pilot
inputs are explicitly known-imperfect preparations predating that gate; no claim
of topology repair follows from adding latent storage.

Version 9 completed successfully. Source-matched `artifacts/kaggle-v9` logs prove
all 83 tests pass on Kaggle with no skips, including the crossing gate and latent
container tests. Actual shape/material posterior means were saved for both pilot
assets and reloaded in Kaggle with expected checkpoint identities. The downloaded
files also load locally without PyTorch: each has one 8-channel shape and one
8-channel material token, reflecting this tiny 16³ fixture and 16× downsampling,
in a 448-byte profile-2 file. This size is not a production compression-quality
result. Joint hierarchy storage was verified with explicit test skeletons only;
no learned rig reconstruction is claimed. Matching model checkpoints are being
retrieved and must be retained with the files because their identities differ
from checkpoint byte streams of previous runs.
Matching version-9 `shape-pilot.pt` and `material-pilot.pt` are now downloaded;
their SHA256 digests were verified against both stored latent checkpoint identities.

## Step 5 rectified-flow transformer foundations

`t3d/flow.py` adds a scalar-time rectified-flow path/velocity objective, conditional
dropout, Euler/CFG sampling, a sparse DiT with adaLN-Zero, serialized local patch
self-attention, prompt-token cross-attention, per-head Gram-Schmidt 3D RoPE, and
optional aligned clean-shape conditioning for material flows. It accepts cached
integer order permutations for cycling Z/TZ/H/TH and handles one asset per forward.
The coarse occupancy model, conditioning encoders, coupled skeleton branches and
full three-stage generative training remain pending. See `docs/flow.md`.

Three new PyTorch tests cover an analytic linear trajectory/Euler solution,
rotation orthonormality/norm preservation with degenerate learned axes, and
conditional-model optimization/sampling/invalid-order rejection. Local discovery:
86 tests, 72 pass and 14 skip without PyTorch. Version 10 was submitted with bundle
SHA256 `f2c9df08642eee175e5c8658165106b567754f3bd878412442220c089ff85e99`.
CUDA execution of these additions remains pending until current outputs are inspected.
No generative quality claim follows from the implementation.

Version 10 completed successfully. Downloaded source-matched logs under
`artifacts/kaggle-v10` confirm all 86 tests pass without skips, including the
analytic flow trajectory/sampler, degenerate-axis RoPE rotation/norm checks, and
conditional sparse DiT optimization/sampling with finite gradients on CUDA.
This verifies backbone execution only. No real-asset flow training, conditioning
encoder alignment, or generated-asset quality has been established. Earlier
pilot VAE/topology failures remain active requirements for correction.

## Three-stage latent flow orchestration

`t3d/generation.py` adds a bounded coarse occupancy flow, sparse geometry/optional
co-located rig flow, and shape-conditioned material flow. It returns explicit
latent streams, handles empty occupancy, validates sparse support, and returns
separate training objectives. Integer Z/TZ/H/TH permutations are cached per stage
for reuse across Euler steps. Details and pending scope are in `docs/generation.md`.

New tests use analytic velocity oracles to verify stage support, clean-shape
conditioning, and empty occupancy, plus neural backward tests for all losses and
coordinate/grid budget rejection. Local discovery: 88 tests, 72 pass and 16 skip
without PyTorch. Version 11 was pushed with bundle SHA256
`d966e4938cdc115282b6325b11f5c65b12859bc234166e18b4c64306ac7e2bb3`.
A subsequent local fix permits parameter-free analytic stage oracles to generate
noise on the registered grid device; it is not yet in version 11. Inspect the
current run before submitting the corrected source. Generated quality, trained
conditioning and the full rig branch remain pending.

Version 11 reached ERROR with one analytic-stage test failure: a parameter-free
oracle left no parameters for noise-device lookup. The local fix uses the
registered grid device and a float32 fallback when stages have no parameters;
trainable stages retain their parameter dtype. The downloaded failure is retained
under `artifacts/kaggle-v11`. Version 12 was submitted with bundle SHA256
`b82f78558a53d5fb3d99cee776185b6d6ad1ea9c1261e1c6ff633728a9dafc4e`.
Its validation remains pending until terminal output inspection.

## Device-resident mesh extraction implementation

`t3d/extraction_torch.py` implements world-vertex placement, sparse neighbor
lookup, complete-quad retention and weighted/minimum-angle triangulation using
PyTorch tensors on the input device. `docs/cuda-extraction.md` defines its scope.
It uses sorted int64 coordinate keys and binary search; the custom CUDA hash
kernel and the <100 ms requirement are still pending. Winding and production
cleanup remain unresolved, including the prior real-asset topology failures.

Three new tests compare randomized CPU/device extraction, row order, sparse
missing neighbors, all-axis boundary rejection, weighted/angle/degenerate
splitting, empty grids and malformed coordinates. Kaggle submission uses bundle
SHA256 `767ed0e848cede3f57ecb3c268570d367e669311100d73eebe55a1e1ead9ce7a`;
remote execution is pending. This addition does not claim Step 6 completion.

Version 13 completed successfully. Downloaded `artifacts/kaggle-v13` reports match
bundle SHA256 `767ed0e848cede3f57ecb3c268570d367e669311100d73eebe55a1e1ead9ce7a`
and record dual Tesla T4 allocation, Torch 2.11.0+cu128/CUDA 12.8, and all 91
tests passing in 10.762 seconds without skips. The three extraction tests ran on
CUDA and matched CPU world vertices and exact triangle indices. A warning about
read-only NumPy staging arrays occurred in the test adapter; extraction does not
mutate these arrays. No extraction timing or production-resolution memory result
was measured; the planned <100 ms gate is unverified.

## Extraction performance measurement

Added `tools/benchmark_extraction.py` to the private Kaggle notebook. It measures
both split modes at 512³, 1024³ and 1536³ on analytic sheets with R² active voxels,
checks count/bounds/area, and records synchronized wall/CUDA-event times and peak
PyTorch allocated memory. Its explicit measurement scope is documented in
`docs/cuda-extraction.md`. Version 14 bundle SHA256 is
`77a6a1c74dd63408608a54a2f6147c74e9eb7447616e8fe7cfaa9490afb6d486`.
Results are pending source-matched output inspection.

Version 14 completed and downloaded outputs match the submitted bundle SHA256.
The GPU report confirms dual Tesla T4 allocation and the full 91-test suite passes
without skips (12.800 seconds). The extraction benchmark itself uses one T4.
All six analytic count/bounds/area checks passed. Median full-call wall times for
weighted/minimum-angle modes were 5.77/10.15 ms (512³, 262,144 voxels),
19.82/32.00 ms (1024³, 1,048,576 voxels), and 36.27/69.99 ms (1536³,
2,359,296 voxels and 4,712,450 triangles). Peak PyTorch allocation at 1536³ was
981,286,912/2,303,284,224 bytes respectively. Exact repetitions, CUDA-event times
and measurement scope are retained in `artifacts/kaggle-v14/extraction-benchmark.json`.
The <100 ms condition is observed on these sheets only. Broad asset workloads,
nonplanar/sliver geometry, source reconstruction fidelity and production export
remain unverified. This evidence does not complete Step 6 or repair prior topology
and VAE failures. See `docs/cuda-extraction.md` for the measurement table.

## CuMesh cleanup and UV adapter

Added `t3d/mesh_postprocess.py` for the standard Step 6 cleanup/decimation,
reference BVH and cone-clustered UV pipeline using the real CuMesh API. Output
includes UV geometry, normals, cleaned-vertex mapping, reference BVH and stage
counts. `docs/mesh-postprocess.md` records API sources and limits. At the time
this adapter was added, strict remeshing, PBR baking, GLB packaging and generated
rig attachment were pending. Later synthetic-cube GPU probes cover portions of
the pipeline, and the low-level skin packer now has separate real-source
round-trip evidence; predicted-rig integration and broad production validation
remain pending.

`tools/probe_cumesh.py` builds pinned CuMesh commit
`12289e1062f0603f2f0d0771b02e1395d247f26f` on Kaggle and checks an extracted
cube's cleanup, UV/normals, valid indices and closedness after welding UV seams.
Kaggle version 15 source bundle SHA256 is
`8c4f4fe4eca142fa27f9479bd09b163f8ef154f8602520f02be887b43c588e69`.
Compilation and probe results are pending terminal source-matched output inspection.

## Baked PBR GLB packaging

Added `t3d/glb.py:pack_glb` to package prepared mesh arrays and an explicitly
supplied baked six-channel PBR texture as self-contained binary glTF. It handles
linear-to-sRGB RGB conversion, linear alpha, G roughness/B metallic packing,
explicit alpha modes/cutoff, included normals, UV V flip and z-up to glTF y-up
conversion. `docs/glb.md` records input conventions and remaining integration.
Two focused local tests passed for GLB structure/embedded maps, coordinate/UV
round trips, PBR channel/color values, alpha and invalid inputs. A trimesh reload
warning uses a NumPy fallback because local SciPy is absent. A subsequent normal
attribute assertion was added and verified in the later four-test GLB suite. The initial test
expected 128 from a rounded linear value lying slightly below sRGB byte midpoint;
using linear 0.25 with unambiguous expected byte 137 corrects the test fixture.

## Low-level rigged GLB packing

Extended `t3d.glb.pack_glb` with optional dense skin weights and a validated
`Skeleton`; `attach_skin` writes up to 16 influences per vertex, a joint-node
hierarchy and inverse-bind matrices. It also sets vertex/index buffer-view
targets required by the Khronos validator. Four focused GLB tests pass.

`tools.audit_rig_export` repacks the actual RiggedSimple, CesiumMan and Fox
source rigs and reloads each result with the project importer. Maximum vertex
errors are 2.87e-8, 9.21e-8 and 4.44e-6; joint errors are zero; maximum weight
errors are 0, 2.51e-8 and 9.09e-12. Bind-palette identity deltas are below
1.9e-6. The emitted files are retained under
`artifacts/rigged-inputs/glb-export-roundtrip/`; independent Khronos validator
results in `artifacts/rigged-inputs/glb-export-validator.json` show zero errors,
warnings or infos for all three files. This tests skin packing and bind-pose
round-trip only: flat placeholder PBR is used, and no original materials,
animation clips, predicted joint clustering, SkinAE decoding, post-remesh weight
transfer, or connection to `t3d.export.export_raw` is implemented. Step 6 remains
partial.
The packer explicitly includes normals. GPU baking and the full production export
chain are tracked separately; passing packer tests does not verify those stages.

## Sparse GPU texture-baking implementation

Added `t3d/texture_baking.py` with nvdiffrast UV rasterization, BVH barycentric
reprojection, FlexGEMM trilinear six-channel PBR sampling plus a support channel,
and Telea atlas-gap inpainting. `PreparedMesh` now preserves the reference
geometry corresponding to its BVH. The function returns linear PBR and the valid
UV mask for the GLB packer. `docs/texture-baking.md` records semantics, upstream
API sources and pending probes. Syntax compilation succeeds locally; execution
requires CUDA and optional libraries absent locally and is NOT yet verified.
This source is newer than still-running version 15. It must be tested in the
next source-matched notebook after inspecting the current CuMesh build result.

## End-to-end export probe prepared

Added `tools/probe_export.py` to build pinned FlexGEMM/nvdiffrast dependencies,
verify the sparse interpolation coordinate convention independently, bake a
constant PBR cube through the real GPU operations, save `export-cube.glb`, and
reload it for geometry/transparency checks. Dependency revisions are recorded in
the script. It is syntax-checked locally but not yet submitted: version 15 is
still authoritatively RUNNING and must reach a terminal state before replacement.

Inspected the pinned FlexGEMM CUDA kernel: interpolation samples are at voxel
centers `coord+0.5`, with `floor(query-0.5)` neighbor selection. The affine oracle
expects `(query-.5)/3` for features `coord/3`; world queries need no added offset.
Corrected the baking docstring and documented this distinction. This prevents
an incorrect test oracle and does not establish GPU execution yet.

## Export-probe notebook wiring and dependency artifacts

Wired `tools/probe_export.py` into the next notebook generation, after the CuMesh
probe. `tools/build_cuda_dependency.py` now builds pinned dependency wheels in
`/kaggle/working/dependency-wheels`, installs them and retains per-wheel SHA256,
commit, Python/PyTorch/CUDA/ABI and sm_75 build receipts. Successful builds can
therefore be retained with the notebook outputs instead of losing the compiled
artifacts. These changes affect future submissions only; current version 15 is
still RUNNING and has not been replaced. All four changed build/probe Python
files pass local syntax compilation; CUDA build/runtime validation is pending.

## Strict remeshing branch

Extended `clean_and_unwrap` with the plan's optional narrow-band dual-contouring
branch: `remesh=True` requires an explicit AABB/grid, builds at the maximum grid
resolution with band 1 and BVH projection 0.9, then simplifies to the face target.
The standard branch is preserved. Added finite-float32 position validation before
CuMesh conversion. The next CuMesh probe now exercises both branches and saves
separate geometry/UV artifacts and topology audits. Syntax compilation succeeds;
GPU execution is pending. Version 15 is still RUNNING and contains the prior
standard branch only; it has not been replaced. No claim of strict watertightness
is made before the planned welded-position topology check passes.

Version 15 completed successfully. Downloaded GPU and CuMesh probe reports match
bundle SHA256 `8c4f4fe4eca142fa27f9479bd09b163f8ef154f8602520f02be887b43c588e69`.
Pinned CuMesh compiled/installed on Kaggle's Torch 2.11.0+cu128 runtime, and the
standard cube cleanup/UV pipeline ran on Tesla T4. Input 152 vertices/300 faces
became 95 cleaned vertices/186 faces and 148 UV-seam vertices. Welded-position
QA finds one closed component, signed volume 0.2157349906, zero flips; UVs and
normals are finite. This is standard-branch analytic evidence only. Version 15
does not include the subsequent remeshing, reference-array retention or baking
additions. Those still require the expanded next run.

Version 16 submitted with bundle SHA256
`f5777014916193eb5411929f0139e10e03f98888e2db9ef5b99d3749c5f59f60`.
It includes two-branch CuMesh QA, retained pinned dependency wheels, all current
unit tests, sparse affine interpolation oracle, constant-PBR UV baking and saved
GLB reload validation. Its extended build budget is 3600 seconds. Do not replace
this run while active; inspect terminal source-matched reports and build logs.

## Raw-container export entry point

Added `t3d/export.py` with `export_raw` and `python -m t3d.export input.t3d output.glb`.
It connects validated raw Track A data to CUDA extraction, standard/strict CuMesh
preparation, PBR baking and GLB packaging. It records stage counts, texture
coverage, source/output hashes, device/options/timing in a sidecar. Source bytes
are read once for consistent parsing/hash provenance. Writes replace temporary
files after export succeeds; input/output/audit collisions are rejected.
Missing PBR and currently unsupported rig fields fail before GPU allocation.
Alpha defaults to BLEND when the source is translucent; y/z coordinate conventions
are explicit. Latent decoding and BPT retopology are not performed by this command.

Two local contract tests pass and CLI help executes. The full public export API
is newer than queued version 16 and has not yet run on CUDA. Version 16 exercises
the underlying stages via its existing probe; inspect those results before the
next submission tests this new entry point. See `docs/export.md` for usage and
scope. The original plan's data/training/quality requirements remain unfinished.

## Atlas padding regression and correction

Installed optional OpenCV in the project venv for CPU padding QA (OpenCV 5.0.0.93,
NumPy 2.5.4). A single-channel 0.25 experiment passed, but extending it to six
constant material values exposed a real gap-filling failure: unscaled float32
Telea output drifted enough to clip to 0/1 outside the covered UV region. The
old GPU probe checked only covered texels and would not catch this.

Extracted `t3d/texture_padding.py:pad_pbr`, switched Telea to the plan's byte-map
domain, and bounded extrapolated channels by covered source extrema. Byte-domain
filling alone still drifted up to 0.01569 on this fixture, so the bounds are
necessary for constant-material preservation. Covered texels retain their exact
source values. Two local tests now pass for six constant channels across gaps,
covered preservation, input immutability, full atlases and invalid empty support.
The future GPU export probe now checks whole-atlas error as well. These changes
are newer than queued version 16 and are not yet validated on Kaggle.

## Public export command validation prepared

The complete local suite after the padding/dependency changes discovers 97 tests:
78 pass and 19 skip for absent PyTorch, with zero failures. Detailed output is
retained in `artifacts/local-suite.log`. This is CPU regression evidence only.

Extended the future export GPU probe to serialize a raw PBR `.t3d`, invoke
`python -m t3d.export`, reload its GLB/audit sidecar, verify source/output hashes,
alpha mode and face counts, and require closed/outward topology after welding UV
seams. This deliberately exercises raw provisional extraction winding, whereas
the lower-level probe previously oriented its fixture on CPU before CuMesh.
Syntax checks pass. Version 16 has now started RUNNING and predates this command
probe/padding fix; it has not been replaced. No command-level GPU success is
claimed before terminal source-matched results from a later run are inspected.

## CPU source-surface query hierarchy

Added opt-in `TriangleBVH` and `closest_surface(method='bvh')` to prune source
triangles by AABB distance while reusing the exact existing triangle solver.
Default material/skin projection remains exhaustive pending broader seam/perf
validation. Two targeted tests pass, covering randomized reference parity,
pruning, source-order equidistant ties, empty queries and invalid inputs.

Real Avocado/WaterBottle normalized fixtures use 256 deterministic perturbed
vertex queries. Initial projected positions agreed within 6e-17 but some face
IDs differed: recomputing the leaf distances with a different NumPy reduction
perturbed near-tie comparisons. Returning the solver's original squared distances
resolved this on both fixtures: current face IDs, positions and distances match
exactly. Results are retained in `artifacts/proximity-benchmark.json`; the tool
now requires face-ID parity as well as geometric agreement.

Observed build+query time was ~0.118 s versus 0.119 s brute on Avocado, and
~0.657 s versus 0.774 s on WaterBottle. Visited pairs fell to 23,768/174,592 and
68,506/1,154,560 respectively. These are single small-batch observations, not
production performance proof. See `docs/proximity.md`. Version 16 is still
RUNNING and predates this code; it has not been replaced.

## Dependency wheel reuse checks

Extended the future CUDA dependency builder to read an explicitly named private
Kaggle cache mount (`/kaggle/input/transformer-native-3d-cuda-wheels`). It reuses
only a source-URL/commit-matched wheel with matching Python major/minor,
PyTorch build, CUDA version, C++ ABI and sm_75 target, then verifies its recorded
SHA256. Absent or incompatible receipts fall back to source compilation;
corrupt payloads/invalid filenames fail. Reused wheels and provenance receipts
are retained in the new outputs. No cache dataset is attached yet: version 16
must first finish and its actual wheel artifacts must be inspected and retained.

Two local tests pass for compatible reuse eligibility, runtime/source mismatch,
checksum failure, missing cache and invalid paths. Tests use inert byte fixtures
and do not install binaries. Actual cached-wheel import/execution on Kaggle
remains unverified. Current version 16 is still RUNNING and predates this helper
change; it has not been replaced.

## Spatial texture-baking oracle prepared

Strengthened the future export probe with an independent planar UV fixture and a
spatially varying dense PBR field. Expected RGB follows analytic world x/y and
constant z at each texel center, so constant-material masking cannot hide UV
orientation, interpolation or BVH-projection errors. A second bake forces one
triangle per raster chunk and must match the single-chunk bake exactly within
1e-6, explicitly exercising global face-ID offsets beyond triangle zero.

`bake_pbr` now accepts bounded `raster_chunk_size` (default remains the plan's
100,000 faces) to exercise chunk logic without building a huge test mesh. The
new probe asserts analytic gradient error below 1e-5 and stores both results.
Syntax compilation passes; actual CUDA execution is pending a subsequent run.
Current version 16 is still RUNNING and predates these checks; do not attribute
future checks to its pending output.

## CUDA cache dataset preparation

Added `tools/prepare_cuda_cache.py` to validate downloaded dependency receipts
against pinned source revisions and hashes, require consistent runtime records,
and prepare a private Kaggle dataset directory. It supports explicit partial
caches when a run built only some dependencies. Dataset files are flat (named
`<dependency>.receipt.json` alongside wheel files), avoiding directory-upload
archive ambiguity. The reuse helper accepts that layout and the original nested
notebook layout. Two eligibility tests pass, now including flat-layout checks.
The preparation CLI passes syntax/help checks. No dataset has been created yet;
version 16 remains RUNNING, so real wheel inspection and a reuse/import probe
are pending. This has not replaced or restarted the active notebook.

## Upstream native conversion convention audit and bridge

Inspected the actual O-Voxel wrapper/native sources rather than assuming the
plan's named API returns this project's local-offset schema. Reviewed upstream
commit: `75fbf0183001ed9876c8dbb35de6b68552ee08bd`. The wrapper subtracts the AABB
minimum; the native CPU QEF returns physical positions in that shifted frame.
Those outputs need `physical / voxel_size - coords` before schema storage.

Added optional `t3d/native_voxelize.py` with explicit unit adaptation and CPU
inputs to the upstream function. `from_native_grid` preserves float32 voxel-size
semantics, rejects significant out-of-cell values, records any roundoff clamping
and canonicalizes rows. Two CPU tests pass for anisotropic cells/nonzero origin,
wrong units and small float32 boundary error. `docs/native-voxelize.md` links the
reviewed source and documents this as source-derived behavior needing execution
confirmation, not an upstream reconstruction or performance success claim.

The upstream extension is not installed locally or in active version 16. A pinned
build plus real source/reconstructed topology comparison is still required; the
default custom converter and its known failures remain unchanged. Version 16 is
still RUNNING and has not been replaced.

Version 16 completed. Downloaded reports match bundle SHA256
`f5777014916193eb5411929f0139e10e03f98888e2db9ef5b99d3749c5f59f60`.
All 93 tests pass without skips (10.319 seconds). Real FlexGEMM affine sampling
passes, GPU UV/PBR baking produced 15,624 covered texels with max covered-material
error 2.384e-7, and a 15,108-byte, 186-face BLEND GLB was saved/reloaded. The
saved artifact is `artifacts/kaggle-v16/export-cube.glb`. This probe predates the
whole-atlas padding correction, gradient/chunk oracle and public CLI checks.

The remesh probe is NOT a quality success: while both branches produced closed
welded components with finite UVs/normals, the 8³ strict branch produced TWO
components (188+12 faces), with 12 faces needing outward flips in the audit.
The source cube has one component. The existing closedness gate was too weak.
Inspection of the pinned CuMesh code confirms its field is unsigned distance
minus a band offset, which can produce inner/outer level sets; broader topology
preservation and outward orientation need an explicit solution/quality gate.
Do not label strict remeshing complete based on this run.

All three dependency wheels and receipts were downloaded, their pinned identities
and hashes verified, and package license files confirmed inside the wheels.
`tools.prepare_cuda_cache` successfully prepared `artifacts/cuda-cache`. A private
Kaggle cache dataset upload is in progress; no cached-wheel execution claim yet.

The private CUDA-wheel cache dataset is now READY at
https://www.kaggle.com/datasets/sumu143/transformer-native-3d-cuda-wheels .
It contains all three verified version-16 wheels and flat receipts, preserving
embedded upstream licenses. Version 17 bundle SHA256 is
`23a5ce2feef1198bbe36b66c34dd1a6be0b7a93ec7a79cba9ba021352530a6de`.
The next notebook attaches this cache, installs OpenCV before tests, and includes
the gradient/chunk oracle, fixed atlas padding, public export command probe,
CPU projection/native-unit tests and explicit remesh component/orientation flags.
Cache import/reuse and the stronger export results remain pending.

Version 17 ended in ERROR after all 103 tests passed without skips (11.358 s).
Downloaded GPU report matches bundle SHA256
`23a5ce2feef1198bbe36b66c34dd1a6be0b7a93ec7a79cba9ba021352530a6de`.
All three dependency receipts confirm cached-wheel reuse; execution reached the
export probe without rebuilding libraries. The failure was the independent
plane fixture: CuMesh cuBVH asserts triangle count >8, while the fixture had two.
No gradient/chunk/CLI export success is claimed for this run.

Added `build_reference_bvh` to repeat reference triangles only when needed to
satisfy that backend minimum. Duplicating reference triangles does not change
the closest surface or output mesh. The returned exact BVH face array is retained
for barycentric face-ID indexing; the source remeshing input and exported faces
are not padded. The two-triangle gradient probe now exercises this bridge.
Syntax checks pass. Version 18 bundle SHA256 is
`512fcd434230053e1f2ab6d448e12b20dfc51d6f3a868422d9e64134ea2f4159`;
GPU verification is pending. The known remesh quality failure remains unresolved.

## Version 18 GPU export evidence

Version 18 completed with source SHA256
`512fcd434230053e1f2ab6d448e12b20dfc51d6f3a868422d9e64134ea2f4159`.
The source-matched Kaggle report records 103 tests passing without skips on
Tesla T4. The independent affine sampler and two-triangle BVH fixture passed;
the spatial gradient had zero measured maximum error and the forced raster
chunking matched. The constant six-channel full-atlas maximum error was
2.3841858e-7.

The public raw `.t3d` export command produced and reloaded
`artifacts/kaggle-v18/export-cli.glb`: 152 active voxels, 300 extracted faces,
200 output faces, BLEND alpha, verified input/output hashes, and one closed
outward component after position welding. Total measured command time was
2.1055 seconds for this 8³ synthetic cube and 128² texture. This demonstrates
the standard path on that fixture only; it is not a real-asset fidelity result.

The strict branch still fails topology preservation on the same fixture: it
produces two closed components and 12 faces requiring flips, while the source
has one component. This remains an unresolved quality failure. Large-scale
curation/training, rig export, research benchmarks and the full blueprint remain
incomplete. The saved test GLB and source-matched JSON/logs are retained under
`artifacts/kaggle-v18`.

## Longer reconstruction pilot

The real-asset VAE probe now runs 200 updates per shape/material model by default
and records posterior-mean train/held-out evaluations at steps 1, 10, 50, 100 and
200. It still saves/restores model/optimizer checkpoints, latent files and decoded
teacher/predicted support meshes with surface metrics. This provides evidence on
whether the previous ten-update reconstruction failure improves with training;
the two-asset 16³ corpus remains insufficient for generalization claims.

Kaggle version 19 was submitted with bundle SHA256
`1519d5a19e540294aab6e4e24edc2b6ac2f3224fe05575e14fe334cef437f824`.
Local syntax compilation passed. Training and quality results require inspection
of the source-matched terminal output; no improvement is claimed yet.

## 200-update real-asset reconstruction pilot

Kaggle version 19 completed with source bundle SHA256
`1519d5a19e540294aab6e4e24edc2b6ac2f3224fe05575e14fe334cef437f824`;
all 103 tests passed without skips. Shape and material models each completed 200
updates on the single 16³ Avocado training asset; WaterBottle was held out.
Source-matched reports, checkpoints, latent files and reconstruction meshes are
retained in `artifacts/kaggle-v19`.

Predicted-support F-score at tolerance 0.01 improved from zero in the earlier
10-update pilot to 0.3085 on train and 0.1447 on held-out. These compare against
meshes extracted from the already imperfect prepared `.t3d` inputs, using only
256 samples; they are not original-GLB fidelity or production benchmarks.
Topology remains poor: train/held-out predicted meshes have 321/288 boundary
edges and 35/28 nonmanifold edges. Held-out total shape loss worsened from
1.526 to 2.708 and material loss from 1.049 to 3.936; KL rose substantially.
Longer training alone therefore does not establish usable reconstruction or
generalization. The large curated corpus, better optimization/supervision and
research-quality topology remain unresolved.

## Step-50 reconstruction comparison

The two-asset Kaggle probe now saves both shape/material checkpoints at step 50
and reconstructs train and held-out meshes with teacher and predicted support,
using the same point-to-triangle metric and samples as the final checkpoints.
It retains both checkpoints and `.t3d`/OBJ reconstructions so geometry can be
inspected directly. This investigates the held-out loss minimum observed in v19;
it does not assume loss and mesh quality improve together.

Version 20 bundle SHA256 is
`4b605fe2f41150e0535c7ab6682d4aa2b1c9e9d35b125ef13f4fb6b5fd4e0388`.
Local syntax compilation passed. Source-matched Kaggle execution and comparison
are pending; the small corpus and source-topology problems remain limitations.

Version 20 completed with source SHA256
`4b605fe2f41150e0535c7ab6682d4aa2b1c9e9d35b125ef13f4fb6b5fd4e0388`;
all 103 tests passed without skips. At step 50, predicted-support F-score was
0.0216 train and 0.0141 held-out, with only 30 triangles in each mesh. At step
200 it was 0.3085 train and 0.1447 held-out, with 570 and 554 triangles.
Teacher-support scores also rose from 0.1641/0.1090 to 0.3662/0.2209.
Thus the lower held-out tensor loss at step 50 did not correspond to better
mesh reconstruction on these fixtures. Neither checkpoint yields acceptable
geometry: step 200 predicted train/held-out meshes have 321/288 boundary edges
and 35/28 nonmanifold edges. These are 256-sample comparisons against prepared
16³ `.t3d` source meshes, not original GLB fidelity or a generalization benchmark.
Checkpoints and OBJ/`.t3d` reconstructions are retained under `artifacts/kaggle-v20`.

## Edge-intersection loss imbalance ablation

Added optional per-axis positive weighting to the shape intersection BCE loss.
The default remains unweighted. The Kaggle ablation derives weights only from
the training asset (negative/positive count per axis, bounded to [1,20]); it does
not use held-out geometry to choose weights. The trainer saves/restores this
objective setting, with old checkpoints defaulting to unweighted behavior.
A CUDA test checks exact stochastic continuation of a weighted shape checkpoint.
Local syntax checks pass; actual execution and quality effects remain unverified.

Version 21 submitted with source bundle SHA256
`d4f71803937689f3c7f58e7ebdc91b4e59a12a546b9e69f2031d7e76b32ae20b`.
It uses the same two-asset pilot, seed, widths and 200-update protocol as v20,
with the edge loss change enabled. Step-50/final reconstructions and topology
metrics are retained for comparison. More positive flags may also create false
surfaces, so no improvement is presumed before inspecting the metrics.

## Weighted edge loss result and edge audit

Version 21 completed with source SHA256
`d4f71803937689f3c7f58e7ebdc91b4e59a12a546b9e69f2031d7e76b32ae20b`;
all 104 tests passed without skips, including weighted checkpoint resume. The
200-step predicted-support held-out F-score fell from 0.1447 (unweighted v20)
to 0.1249 (weighted v21), while nonmanifold edges rose from 28 to 82. Training
F-score likewise fell from 0.3085 to 0.2565. Teacher-support held-out F-score
rose from 0.2209 to 0.2558, but this does not establish better inference quality.
Both runs use the same small pilot and seed; the held-out target is an already
imperfect prepared 16³ asset. Unweighted remains the default.

Added `tools/audit_edge_predictions.py` to compare decoded edge flags on the
union of source and predicted coordinates, by axis, including missing and extra
support. On the held-out predicted-support mesh, weighted training increased
true positives from 117 to 135 but false positives from 361 to 507. Counts and
precision/recall for both checkpoints and support modes are saved in
`artifacts/kaggle-v20/edge-audit.json` and `artifacts/kaggle-v21/edge-audit.json`.
A focused test verifies union-coordinate TP/FP/FN handling, including zero
precision/recall denominators. These results motivate investigating edge
calibration and topology-aware supervision rather than adopting this weighting.

## Edge threshold calibration

`reconstruct_asset` now accepts `edge_threshold` independently of the existing
occupancy `threshold`; omitting it preserves the old behavior. Both thresholds
are validated. A CUDA test checks that changing edge threshold changes flags
without changing teacher support. The next two-asset pilot sweeps edge thresholds
0.35, 0.5, 0.65 and 0.8 with occupancy fixed at 0.5, reporting per-axis TP/FP/FN
and mesh metrics for train and held-out. Held-out results are exploratory and
must not be used to claim independent final evaluation after selecting a setting.

Version 22 source bundle SHA256 is
`6e4ffad31cd586bed8dd8b90cb5b5af78954cbadb3a27a06197fcc88ea8fb70e`.
It uses the weighted v21 training setup and the existing two-asset corpus; syntax
checks passed locally. CUDA tests and sweep results remain pending. Calibration
alone cannot establish topology fidelity or replace the required larger corpus.

Version 22 completed with matching source SHA256
`6e4ffad31cd586bed8dd8b90cb5b5af78954cbadb3a27a06197fcc88ea8fb70e`;
all 106 tests passed without skips. The weighted-model sweep fixed occupancy at
0.5. On held-out WaterBottle, raising edge threshold from 0.35 to 0.8 reduced
false-positive flags from 594 to 296 and nonmanifold edges from 120 to 14, but
also reduced true positives from 158 to 71, increased boundary edges from 270 to
364 and lowered F-score from 0.1287 to 0.0777. At 0.5 the corresponding values
were 507 false positives, 135 true positives, 82 nonmanifold edges, 306 boundary
edges and F-score 0.1249. The training asset shows a similar precision/coverage
tradeoff. Full results are in `artifacts/kaggle-v22/corpus-probe.json`.

This is an exploratory sweep on one held-out asset, not an independently tested
threshold selection. No setting resolves the underlying topology or reconstruction
quality, and no new default has been selected. Render-based supervision, a larger
curated dataset, source conversion quality and evaluation against original meshes
remain necessary to pursue the blueprint's targets.

## Differentiable render-loss foundation

Added `t3d/render_loss.py` with nvdiffrast multi-view silhouette, depth and
normal losses on meshes sharing connectivity. Positions are differentiable;
edge selection is discrete and receives no gradient. A Kaggle probe checks zero
loss for identical surfaces and finite nonzero gradients for a shifted/tilted
surface. This is not yet integrated into SC-VAE training and does not provide
material color/SSIM/LPIPS supervision. Syntax compilation passed locally.

Version 23 was submitted with bundle SHA256
`5a60a8b3156ef23a3fb7bed9546b56e8f4302a338de32cc398fef6cbf6fca1e0`.
The CUDA probe is pending; no reconstruction improvement is claimed. Surface
correspondence, visibility coverage, camera conventions, and training effectiveness
still require broader validation.

Version 23 completed with matching bundle SHA256
`5a60a8b3156ef23a3fb7bed9546b56e8f4302a338de32cc398fef6cbf6fca1e0`.
All 106 tests passed without skips, and the export probe completed its identical
surface and shifted/tilted surface render checks, including finite nonzero
vertex gradients on Tesla T4. This establishes a small synthetic execution case,
not render fidelity or training effectiveness.

## Opt-in render-supervised shape training

Connected the render loss to `VAETrainer` via optional `render_weight`, default
zero and persisted in checkpoints (old checkpoints load with zero). For a
single asset, source edge flags and teacher support establish reference triangles;
three orthographic views supervise silhouette, depth and normals. Vertex positions
receive gradients; discrete edge flags still use BCE and receive no render gradient.
The reference topology may itself be imperfect. This initial path supports only
one asset per batch and does not implement material render/SSIM/LPIPS losses.

Version 24 source SHA256 is
`82db5a79d4311c2970b45f8037db1158b8f0b1f259aff7ac1c10f95cca53b365`.
It submits the same 200-step two-asset pilot with unweighted edge BCE and render
weight 0.1, so v20 is the relevant tensor-only comparison. Export dependencies
are installed/probed before training. Syntax compilation passed locally; training,
checkpoint resume and quality results are pending CUDA inspection. No improvement
or production-scale validity is claimed.

## Render-supervised training pilot result

Version 24 completed with matching source SHA256
`82db5a79d4311c2970b45f8037db1158b8f0b1f259aff7ac1c10f95cca53b365`;
all 106 tests passed without skips. The shape trainer executed 200 updates with
render weight 0.1 and restored its checkpoint with the render setting intact.
Against tensor-only v20 at the same seed and two-asset protocol, predicted-support
train F-score was 0.3309 versus 0.3085, while held-out was 0.1049 versus 0.1447.
Held-out boundary/nonmanifold counts were 280/22 versus 288/28. Teacher-support
held-out F-score was 0.2052 versus 0.2209. Metrics use 256 samples against the
prepared 16³ meshes, not original source GLBs, and the held-out sample is one
asset. Lower held-out tensor/render total cannot be compared directly with v20's
loss because the objectives differ. No reliable generalization or quality gain
is established, and render supervision remains opt-in. The discrete edge logits
still receive no rendering gradient; source connectivity is used during rendering.
Reports and checkpoints are in `artifacts/kaggle-v24`.

## Original GLB surface comparison

`tools/audit_original_reconstruction.py` compares the retained v20 tensor-only
and v24 render-supervised outputs directly against the normalized original GLB
triangles, rather than only against extracted prepared voxels. It checks canonical
bounds, records source/output hashes and notebook bundle identities, and reports
three deterministic 256-sample seeds with unsquared bidirectional distances and
F-score at 0.01. Results are in `artifacts/import/original-reconstruction-audit.json`.

Across those seeds, mean held-out predicted-support F-score is 0.1495 for v20
and 0.1355 for v24; mean distance is 0.04241 and 0.04319 respectively. The
prepared WaterBottle reference scores 0.9786 and 0.00351 against its original
surface by this sampling metric. Training-asset predicted F-scores are 0.2939
(v20) and 0.3195 (v24). These numbers confirm that poor VAE reconstruction is
not solely an artifact of comparing to the prepared mesh, but they do not prove
a statistically reliable ranking of render supervision: there are only two
assets, three small samples, and one held-out asset. The prepared source also
has known nonmanifold defects that small surface distances do not reveal. This
is not the blueprint's official Mesh Distance or sharp-normal benchmark.

## Additional dataset candidates

Fetched two model candidates and retained their upstream README/license files.
Lantern (`a79458c4b02d695187a952f23a63b8bf278e7bc3d316a3c2a314f2d6974181f1`)
loads as 4,145 vertices/5,394 faces, but the 16³ converter detects 30 edges
with multiple distinct crossings and rejects the conversion. Toy Car's extended
materials are rejected by the static core importer. Neither candidate was added
to the training manifest or used to claim a larger valid corpus. The model files
are CC0, while each retained LICENSE.md states the applicable metadata terms.
`artifacts/import/candidate-audit.json` records source links, hashes and rejection
reasons. This confirms that importer support and conversion fidelity must improve
before these real assets can safely broaden the pilot.

## Pinned native converter comparison submitted

Created a private source-asset dataset with the retained Avocado, WaterBottle and
Lantern GLBs and source/license documentation. Added `tools/probe_native_converter.py`
and a separate private Kaggle notebook builder `tools/build_native_notebook.py`.
The notebook installs the previously verified cached CUDA wheels, builds upstream
O-Voxel at commit `75fbf0183001ed9876c8dbb35de6b68552ee08bd`, then runs its CPU
converter at 16³ and 32³ on all three original assets. It records conversion time,
source/extracted boundary and nonmanifold edge counts, sampled surface distances,
unit-conversion diagnostics and hashes, and saves geometry-only `.t3d` outputs.
This evaluates native behavior; it does not silently admit lossy assets to training.

Notebook source SHA256:
`e9fc8590b7a7664c17233a52383f122bcc488929d06d78dad625f9186fdad130`.
Local syntax checks passed. The notebook and upstream CUDA extension build have
not yet been validated on Kaggle. Results must be checked against this hash and
against the earlier custom converter audit; any topology defects remain failures.

## Occupancy versus edge-error diagnosis

Extended the edge audit to divide false-positive flags between reference-active
voxels and extra predicted voxels, and false-negative flags between retained
predicted support and missing predicted support. A focused test verifies both
partitions on a known coordinate example; the audit was rerun on v20 and v21.

For v20's held-out predicted reconstruction, 233 of 361 false-positive edge
flags lie outside source voxel support, and 217 of 385 false-negative flags lie
on missing predicted voxels. For weighted v21, the corresponding counts are
316/507 and 213/367. Both occupancy and flag prediction contribute materially;
optimizing edge classification alone cannot account for errors on absent or
extra voxels. These measurements are against one imperfect 16³ prepared target,
not proof of generalization. Detailed per-axis results are in the two run
`edge-audit.json` files. The native-converter audit remains running independently.

## Final support and ancestor audit

Added `tools/audit_support_pyramid.py` to compare decoded final support and its
integer 2× ancestors with source support. A focused synthetic test checks union
counts, ancestor merging and empty predictions. On the held-out v20 reconstruction,
the final 16³ support has 263 true positives, 243 extra voxels and 201 missing.
At 2× downsampling the counts are 81/32/15; at 4× they are 16/12/0; at 8× and
16× the ancestor sets match. The v21 and v24 held-out reconstructions show the
same ancestor counts through 2× and broadly similar final counts. These are
ancestors of surviving final voxels, not a direct recording of every decoder
intermediate or its logits: a parent later pruned to zero children is invisible.
Thus the audit identifies where surviving support differs but cannot attribute
individual errors to a particular occupancy head. The one-asset prepared target
is itself imperfect. Full results are retained in each run's
`support-pyramid-audit.json`.

## Direct decoder occupancy trace prepared

`SparseCompressionVAE.decode` now optionally records parent coordinates, logits,
selected masks and child coordinates at all four upsampling stages. The default
return/API is unchanged when no trace list is supplied. A new GPU test checks
teacher masks and a fully pruned first stage; actual CUDA execution remains pending.

`tools/audit_decoder_trace.py` computes per-stage false positives and separates
misses among reachable children from targets whose parent was already pruned.
Its focused CPU fixture passes. `tools/probe_decoder_trace.py` uses the actual
saved v20/v24 checkpoints on both pilot assets, recording free-running decisions
and teacher-context logits at each scale. This is stronger than inferring coarse
support from final descendants. A separate private notebook was prepared with
source SHA256 `68218722f79929e0a8bec1d449d49ef75f31a2ca601737e1e3e696614fddcb1b`;
its checkpoint dataset upload is in progress. It has not yet been submitted or
run. The independent native-converter job is still active and is not restarted.

The private trace-checkpoint dataset is READY. An attempt to submit the separate
trace notebook was rejected by Kaggle: `Maximum batch GPU session count of 2
reached.` The trace notebook was not started; the native-converter notebook is
still RUNNING. Do not cancel or replace active sessions to work around this. Once
a batch GPU slot is available, submit `kaggle-trace` and verify its source hash,
GPU tests and report. A transient session limit is not evidence that the goal is
blocked while meaningful work or an active job remains.

The prepared decoder trace now verifies both downloaded checkpoint SHA256s
against the exact v20/v24 source artifacts before deserializing them. Rebuilt
trace notebook source SHA256:
`f7fed88ab9e1c15e2cc539b41c1a8a513fe82eb47c12a5d4abb8efd4b15e9431`.
The CPU stage-count test passes; CUDA trace behavior is still unverified. Kaggle
previously rejected trace submission because its two batch GPU sessions were in
use, and the independent native converter remains active. Submit the rebuilt
notebook only when a session slot is available; do not attribute the earlier
`682187...` bundle hash to this revision.

## Current Kaggle handoff state (2026-11-03)

Rebuilt both notebook bundles from the present working tree. The authoritative
embedded source hashes are now trace `3c8b2d3b5898b4405dbe1e9fcf572aefaa3e3363ecaf23a5fff9bc1bcd024bd7`
and native converter `973eae7d4af24df3477bb2c00c614c8a426ddc61edc4307b95d18efd04726ebc`.
These supersede the earlier hashes above. The corresponding local notebook files
are `kaggle-trace/decoder-trace.ipynb` and `kaggle-native/native-audit.ipynb`.

Local output inspection finds no downloaded trace-run artifacts. The only native
audit output directory is `artifacts/kaggle-native-v1`, whose logs document the
first run's failure at asset discovery after a successful upstream build. The
native notebook has since been fixed to locate mounted GLBs recursively and to
retain the built wheel. The v2 native notebook and decoder-trace notebook have
not yet been submitted from this workspace. Prior statements above that a native
run was still active or that a trace bundle was merely awaiting an upload are
historical and must not be treated as current remote status; current Kaggle
status has not been checked in this handoff.

The earlier credential hold was an unnecessary agent-imposed restriction. The
user has authorized Kaggle execution, and the configured credential can be used
without copying secrets into notebooks or logs. Rotation of the screenshot's
token remains advisable, but does not itself prevent authorized work.

Revalidated remote state: native converter notebook reports ERROR; the decoder
trace status returned 404 (no existing session). Submitted decoder trace version
1 successfully with the current source bundle on Tesla T4, timeout 1200 seconds.
No existing running notebook was replaced. Its GPU results remain pending.

## Direct decoder trace verified on Kaggle

Trace notebook version 1 completed on Tesla T4. Downloaded
`artifacts/kaggle-trace-v1/decoder-trace-report.json` matches source SHA256
`3c8b2d3b5898b4405dbe1e9fcf572aefaa3e3363ecaf23a5fff9bc1bcd024bd7`;
both checkpoint identities match the pinned v20/v24 hashes. All four focused
tests passed in 2.131 seconds, including the fully pruned-parent trace case;
this is a focused suite, not the complete project suite.

For held-out v20, stage downsample=4 predicts 28 cells against 16 true cells
(TP16/FP12/FN0). Downsample=2 has TP81/FP32/FN15, all 15 misses reachable.
At the final stage TP263/FP243/FN201, only 40 missing cells are unreachable
because of prior pruning; 161 are reachable but rejected locally. With teacher
parents, the final head still misses 181 cells and adds 150. Thus restoring
parent support alone cannot solve the measured final occupancy errors. For v24,
final TP264/FP241/FN200 divides into 40 unreachable and 160 reachable misses;
teacher-context FN166/FP150 also remains poor. These are measurements against
one imperfect 16-cubed held-out target, not production quality evidence.

After downloading trace outputs, submitted the corrected native converter audit
as version 2 on Tesla T4 with timeout 3600 seconds and source SHA256
`973eae7d4af24df3477bb2c00c614c8a426ddc61edc4307b95d18efd04726ebc`.
Its conversion and topology results remain pending. Do not treat a successful
submission or extension build as successful conversion of any source asset.

## Reproducible notebook source archives

Found that earlier archive creation embedded wall-clock ZIP timestamps: a fresh
bundle hash could differ even with identical source files. Therefore the earlier
statement that changed hashes necessarily meant changed source was too strong.
Existing downloaded reports remain valid matches to their exact submitted
archives. Added `tools/notebook_bundle.py` and switched all three notebook
builders to sorted entries with fixed timestamps, permissions, compression and
platform metadata. Future source bundle hashes now remain stable across rebuilds
with unchanged files. Verified fixed metadata, order-independent archive bytes,
content-change sensitivity, and two complete captured builds of each generator
producing identical notebook/metadata bytes. Syntax compilation also passed.
The checks captured generated writes, so the current submitted native-v2 and
trace-v1 notebook artifacts and their evidence hashes were preserved.

Native converter v2 was re-polled after this work and remains RUNNING. Continue
observing this same session; do not replace it or infer conversion success from
its running state. The previous goal turn made progress by completing the direct
GPU trace and submitting this corrected native job.

## Independent native output audit prepared

Added `tools/audit_native_outputs.py`. It requires the exact submitted source
bundle identity and pinned upstream revision, all three original assets at both
16 and 32 resolutions, matching source/output SHA256s, and canonical output
bounds/grid sizes before measuring any geometry. It then independently extracts
all saved `.t3d` files and recomputes the sampled bidirectional surface metrics
against original normalized GLBs for seeds 0/1/2. A report is written only after
all six cases complete. Syntax/CLI checks passed, and temporary fixtures verified
rejection of stale source bundles, wrong upstream revisions and incomplete asset
sets. Actual six-case execution remains pending receipt of native-v2 outputs;
these preparation checks do not establish geometric quality. Kaggle still reports
the same native-v2 session RUNNING.

## Skeleton confidence targets

Reviewed the blueprint's cited AniGen source, section 3.2.2 and 3.3.1–3.3.3:
https://arxiv.org/html/2604.08746v2 . SkinAE pretraining uses the joint/vertex
embedding compatibility decoder; confidence-weighted joint regression belongs
to the skeleton-field branch. These are separate implementation requirements.
Added `skeleton_confidence` in `t3d/articulation.py`, implementing equation 3's
nearest/second-nearest squared-distance target separately from the existing
six displacement channels. Singleton confidence is explicitly one; coincident
nearest joints, including zero-distance duplicates, explicitly receive zero.
Three analytic tests passed, covering joint/midpoint/ratio values, translation
and scale invariance, chunk equivalence, duplicates, singleton and empty inputs.
This supplies CPU preprocessing targets, not a trained confidence predictor,
regression loss integration, SkinAE, or rig export. Native-v2's submitted bundle
is preserved; future notebook builds will include this new helper.

## SkinAE reference model verified on Kaggle

Added `t3d/skin_ae.py`: Fourier-encoded parent-relative vectors and parent-edge
features, a permutation-equivariant Transformer joint encoder with four-channel
outputs, skin-weighted vertex embeddings, 64-channel compatibility MLPs with a
positive learned per-vertex temperature, skin KL loss, and explicit freezing.
Architecture hyperparameters are local choices; no upstream checkpoint parity is
claimed. See `docs/skin-ae.md` for scope and the cited AniGen specification.

Submitted a separate private SkinAE notebook without replacing the active native
audit. Kaggle resolved its title to `sumu143/transformer-native-3d-skinae-check`;
the builder/metadata now use that actual ID. Version 1 completed on Tesla T4.
Downloaded `artifacts/kaggle-skin-v1/skin-report.json` matches source bundle SHA256
`98932ec91a5cebc6c877f5024565e16ab9e888dd4a2cd167f61f947237ebe4f9`.
All three focused tests passed in 1.774 seconds without skips, including finite
parameter gradients, arbitrary joint relabeling, one-joint decoding, normalized
weights, weighted-average encoding, freeze behavior, cyclic rig and bad-weight
rejection. No trained rig checkpoint or real-asset quality follows from this.
Rigged-data ingestion/training, skeleton regression, sparse skeleton support,
latent integration and rig export remain incomplete. The independent native-v2
converter session was re-polled and remains RUNNING.

## Rigged source candidates acquired

Downloaded the Khronos glTF-Sample-Assets RiggedSimple, CesiumMan and Fox GLBs
under `artifacts/rigged-inputs`, retaining each upstream README, source URLs and
SHA256s in per-asset SOURCE.json and combined candidate-audit.json. Binary JSON
inspection confirms one skin each with 2, 19 and 24 joints respectively, plus
JOINTS_0/WEIGHTS_0 attributes and animation data. None declares required glTF
extensions. This is acquisition/schema inspection only: no skin accessor,
node/bind transform or animation correctness is yet established by the importer.
The static importer continues rejecting rigs rather than dropping articulation.

RiggedSimple's README attributes the asset to Cesium under CC BY 4.0. CesiumMan
has the same asset license plus a Cesium legal-mark notice, retained separately
as Cesium-LegalMark.txt. Fox's model is CC0 while its rigging/animation and glTF
conversion have separate CC BY 4.0 attributions retained in its README. Do not
label the entire Fox rigged asset CC0. These three samples are not a curated
research corpus or validated SkinAE training split. The next ingestion step must
read and validate accessor layouts, skin indices/weights, hierarchy and bind
transforms explicitly. Native converter v2 remains RUNNING after revalidation.

## Rigged accessor decoding and real-source audit

Added `t3d/gltf_accessors.py`, a bounded embedded-buffer GLB reader supporting
scalar/vector and float32 MAT4 accessor storage, base offsets/strides, normalized
integer values and sparse overrides. It validates buffer spans and sparse index
ordering before reading, rejects unsupported accessor extensions and other matrix
layouts, and caps decoded elements. `skin_weights` checks paired contiguous
JOINTS_n/WEIGHTS_n sets and legal component types, accumulates influences, rejects
unweighted vertices, and returns normalized dense weights plus the maximum source
sum correction. This is an accessor/weight layer, not a complete glTF validator,
scene evaluator or rigged mesh importer. Reference: Khronos glTF 2.0 specification,
https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html .

Executed on all three retained GLBs: RiggedSimple has 160 vertices/2 joints;
CesiumMan 3273/19; Fox 1728/24. Maximum source weight-sum corrections are
2.98e-8, 8.94e-8 and 5.96e-8. Detailed results are retained in
`artifacts/rigged-inputs/accessor-audit.json`. Three synthetic tests pass for
interleaved normalized values and out-of-bounds rejection, sparse zero-base
reconstruction and invalid index-order rejection, signed normalization, and MAT4
component preservation. Hierarchy/world transforms, inverse-bind evaluation,
shared normalization and independent deformation checks remain necessary before
admitting these sources to SkinAE training. Native-v2 was re-polled and remains
RUNNING; its submitted notebook has not been regenerated or replaced.

## Single-skin scene ingestion and bind evaluation

Added `t3d/import_rig.py`, separate from static PBR voxel import. It resolves core
TRS/column-major node matrices, validates hierarchy cycles/multiple parents,
selects the default scene, maps skin joints to nearest joint ancestors through
intermediate non-joint nodes, reads inverse-bind matrices, and evaluates skinning
in world coordinates as sum(weight * joint_world * inverse_bind * position).
It supports one skin with core triangle primitives and explicitly rejects static
mesh companions, multiple skins, morph targets and required extensions. Animation
tracks and appearance are not evaluated; the original files retain them. Its
result describes only the node-default pose. Shared mesh/skeleton normalization
is invertible and preserves outlying joints.

Real-source execution on RiggedSimple/CesiumMan/Fox produced 160/3273/1728 vertices,
188/4672/576 faces, 2/19/24 joints and one skeleton root each. Recorded hashes,
weight corrections, normalized bounds and mesh roundtrip errors are in
`artifacts/rigged-inputs/rig-import-audit.json`; maximum roundtrip error is 1.43e-14.
Two focused tests pass for independent analytic weighted world transforms,
intermediate joint ancestry, avoiding double mesh transforms, shared joint
normalization inversion, quaternion rotation, column-major matrices and invalid
hierarchy rejection. External viewer/deformation agreement and animation sampling
are still unverified; do not infer those from normalization roundtrip checks.
No SkinAE real-asset training checkpoint has been produced yet. Native-v2 remains
RUNNING after another explicit status poll.

## Real-source SkinAE optimization pilot on Kaggle

Prepared `artifacts/skin-pilot` from the validated node-default-pose imports:
1024 area-uniform surface samples per asset (seed 17), with barycentric skin
weights and recorded face IDs/barycentric coordinates. RiggedSimple and CesiumMan
are training assets; Fox is asset-held-out validation. Per-file/source hashes,
normalization and split identities are retained, along with upstream README,
source and legal-mark records. This remains a three-asset integration dataset;
external deformation agreement and research-scale curation remain pending.

Added `tools/train_skin_probe.py` and extended the private SkinAE notebook.
Version 2 completed 400 alternating training updates (256 points/update,
AdamW lr=1e-3, seed 23) on Tesla T4. All three focused tests passed again in
2.019 seconds. Source SHA256 is
`904ef7b2006075b4800667b2d7a188a3abf788d3a009b8653d980f43a1f11191`;
downloaded manifest and checkpoint hashes also match their report. Full evidence
and model/optimizer/RNG checkpoint are in `artifacts/kaggle-skin-v2`.

RiggedSimple KL improved 0.33222 -> 0.01043, CesiumMan 2.35168 -> 0.03341.
Held-out Fox KL improved 2.60000 -> 1.64187 (uniform baseline 2.61128), but final
dominant-joint accuracy is only 0.59473 and mean absolute weight error 0.05428.
The checkpoint is a pilot, not a production SkinAE. `freeze()` was checked on the
in-memory trained model; the saved training checkpoint precedes freezing, so
callers must explicitly freeze after loading before any generator training.
No validation-based checkpoint selection was used; 400 updates were fixed.

Native converter v2 reached terminal ERROR. Output download sessions 32297 (all
artifacts/wheels) and 96862 (logs only) are active; inspect their final logs before
choosing a repair. Do not restart the terminal job without first diagnosing it.

Native-v2 downloads completed. Upstream O-Voxel built successfully and converted
Avocado at 16-cubed, saving `native-Avocado-16.t3d`, but report serialization then
failed: `TypeError: Object of type float32 is not JSON serializable`.
The roundoff tolerance diagnostic in `from_native_grid` was a NumPy float32;
converted it explicitly to Python float and confirmed the complete diagnostics
dictionary is JSON-serializable on a synthetic native-grid input. The retained
wheel is `artifacts/kaggle-native-v2/native-wheels/o_voxel-0.0.1-cp313-cp313-linux_x86_64.whl`,
SHA256 `b7ee8c199e613c01b543938715179f3d9261b63c4403b478ad74c8b2fa236f68`.
Next run should reuse this pinned wheel rather than repeating the 20-minute
build. No six-case conversion report exists yet, so the complete native quality
comparison remains unverified.

## Cached native converter rerun submitted

Created private Kaggle dataset `sumu143/transformer-native-3d-native-wheel`
containing the retained native-v2 wheel, pinned source/Eigen/runtime receipt and
source attribution note; dataset status is READY. Modified the native probe to
require this cache, check SHA256
`b7ee8c199e613c01b543938715179f3d9261b63c4403b478ad74c8b2fa236f68`
and Python 3.13 / PyTorch 2.11.0+cu128 before installation. It no longer repeats
the upstream compile. Added JSON serialization to the native-adapter regression
test; both focused adapter tests pass.

Submitted native converter notebook version 3 with timeout 1200 seconds and
source bundle SHA256 `63a2d8a9ae6840ec109f96e54f7a83f327eddd497d46d2dc5acd971dc0b2e584`.
The job is RUNNING after explicit polling. Do not assign this source identity to
v2 or use its partial Avocado file as proof that all six cases passed.

## SkinAE exact-input ambiguity check

While native-v3 remains RUNNING, checked the actual float32 parent-relative and
parent-edge inputs for exact duplicate joint feature rows in all three pilot
rigs. RiggedSimple and CesiumMan have none. Fox joints 0/1 share identical input
rows, which implies equal logits in this permutation-equivariant reference
encoder/decoder; however their sampled target weights yield zero additional KL
lower bound from this constraint. Thus this exact-input collision does not
explain the measured held-out KL. `artifacts/skin-pilot-input-audit.json` retains
the groups and the constrained-equal-logit KL calculation. This does not rule out
near collisions, Fourier aliasing, insufficient capacity or training/data limits;
no architecture change is justified by this check alone. Native-v3 was explicitly
re-polled and remains RUNNING; continue observing the same job.

## Salvaged native-v2 Avocado geometry result

Although v2 failed while serializing its report, its first `.t3d` file had already
been saved. Independently reloaded `native-Avocado-16.t3d`, verified its 16^3
grid, and measured it against the original Avocado GLB using the same metric,
sample count, tolerance and seeds as the custom prepared baseline. Across seeds
0/1/2, the native mean bidirectional distances are 0.001066/0.001029/0.001080,
versus custom 0.000862/0.000741/0.000697. Both methods yield 1014 extracted
triangles, 4 boundary edges, 28 nonmanifold edges, and F-score 1.0 on all seeds.
The upstream conversion therefore does not repair Avocado's known topology
defects or improve the sampled distance. This one-asset, one-resolution salvage
is not a substitute for the six-case rerun. Evidence is
`artifacts/kaggle-native-v2/avocado-salvage-audit.json`.

## Held-out nearest-joint baseline

Computed a geometry-only baseline on the same area-sampled validation points.
Fox nearest-joint assignment reaches 0.5195 dominant-joint accuracy and 0.04329
mean absolute weight error (one-hot weights), compared with the 400-update SkinAE's
0.5947 and 0.05428 respectively. SkinAE improves joint identity while degrading
continuous weight error. The weighted targets are inherently soft, so dominant
accuracy alone is incomplete. Per-asset baseline metrics are in
`artifacts/skin-pilot/nearest-joint-baseline.json`; KL uses an explicit 1e-8
probability floor and is diagnostic only. This confirms held-out SkinAE is not
reliable enough to freeze for generation. Kaggle native-v3 session remains
RUNNING; kernel lastRunTime is unchanged at 11:02 UTC and output downloads remain
empty at 11:09 UTC, within the configured 20-minute timeout.

## Full local suite after rig ingestion changes

Ran `.venv/bin/python -m unittest discover -s tests -v`: 120 tests discovered,
95 passed, 25 skipped because this environment has no PyTorch/CUDA. Native adapter,
GLB accessors, rig hierarchy/LBS analytic test, confidence targets, and CPU geometry
all passed. CUDA-dependent checks remain covered only by their source-matched
Kaggle runs. This local suite doesn't change the active Kaggle native-v3 state.

## Rig source bind-palette audit

Compared each node-default mesh-space palette `inverse(mesh_world) * joint_world * inverse_bind`
with identity. CesiumMan's maximum matrix delta is 3.45e-7 across 19 joints;
Fox's is 8.21e-6 across 24. RiggedSimple's two palettes both share the same
90-degree rotation (max delta 1.0), confirming its reported vertex discrepancy
is a rigid node-default pose transform rather than different per-joint errors.
The importer intentionally retains this evaluated pose. Full matrices and
per-asset residuals are in `artifacts/rigged-inputs/bind-pose-audit.json`.
This establishes importer consistency with its node transforms, not visual parity
against an independent glTF renderer. The native converter audit remains RUNNING.

## Bind-palette quality signal in rig manifests

Extended `ImportedRig` to report maximum mesh-space bind-palette deviation from
identity, and recorded it into the regenerated SkinAE pilot manifest. The rig
import tests still pass. Manifest values distinguish RiggedSimple's retained
uniform rotation (1.0) from near-identity CesiumMan (3.45e-7) and Fox (8.20e-6).
This is descriptive pose metadata; it does not reject a nonidentity default pose.
Existing v2 training results/checkpoint used the preceding manifest and remain
historical evidence for that exact manifest hash. Future training bundles must
use the regenerated pilot manifest and verify all data-file hashes.

At this progress checkpoint native converter v3 was still active; see the
following runtime-update section for its terminal status and v4 replacement.

## Native converter runtime update and v4 submission

Native-v3 reached terminal ERROR before conversion. Its logs show Kaggle had
updated to Python 3.13.16 / PyTorch 2.11.0+cu130 / CUDA 13.0, while the cached
O-Voxel wheel was built for Python 3.13.15 / PyTorch 2.11.0+cu128 / CUDA 12.8.
The explicit runtime guard correctly rejected installation. The run did build
CuMesh, FlexGEMM and nvdiffrast for the cu130 T4 runtime; each wheel SHA256 was
verified against its receipt and retained in `artifacts/native-cuda-cache-cu130`
and private Kaggle dataset `sumu143/transformer-native-3d-cuda-wheels-cu130`.

Updated the notebook to install only those runtime-matched cached dependency
wheels, then clone the pinned O-Voxel source and build its extension for the
current runtime. Submitted notebook v4 on Kaggle with source bundle SHA256
`167fc18bc9fce8fa68764b81f44e2c365339ef95163ae9a8115f93a2e5fc4a82`; it is
complete on a Tesla T4 with PyTorch 2.11.0+cu130 / CUDA 13.0. It built the pinned
O-Voxel extension for this runtime and produced all six requested outputs. The
wheel SHA256 is `c674b532ea1e5ce78ecf20100c55353fc61c35c82950580196b71c7dffc458b5`.
The notebook report and `.t3d` output hashes passed the independent audit in
`artifacts/kaggle-native-v4/independent-report.json`; per-run timings and native
diagnostics are in `native-converter-report.json`.

Added a paired experiment using the custom reference converter on exactly the
same three GLBs, grids, extraction, 256 samples and seeds 0/1/2. Across all six
cases, custom mean bidirectional sampled distance is lower than native: Avocado
16³ 0.000767 vs 0.001058; Avocado 32³ 0.000176 vs 0.000317; WaterBottle 16³
0.003506 vs 0.003627; WaterBottle 32³ 0.002369 vs 0.002556; Lantern 16³
0.003589 vs 0.003931; Lantern 32³ 0.001580 vs 0.001712. Native F-scores are
occasionally higher (WaterBottle, particularly 32³), while Lantern 16³ favors
custom; neither converter removes the observed nonmanifold edges. At each
asset/resolution the edge-incidence counts are the same or very close. This
small sampled comparison does not establish a general ranking or satisfy the
blueprint's Dora-Bench/Sketchfab-90 protocol. Full paired values and custom
converter crossing diagnostics are in
`artifacts/native-comparison/paired-comparison.json`.

## Track B BPT pilot quantization preflight

Added reusable `t3d.bpt.quantization_diagnostics` and extended
`tools/audit_bpt_corpus.py` to cover the three static and three rigged Khronos
pilot GLBs, normalizing rigged geometry with its node-default shared mesh/joint
transform. At 128 bins the lossless encoder rejects all six rather than silently
deleting faces: Avocado 0 collapsed/1 collinear; WaterBottle 957/28; Lantern
4,556/86; CesiumMan 324/18; Fox 14/0; RiggedSimple 0/6. All six have zero exact
zero-area source faces, although Lantern has six triangles below the 1e-8
doubled-area diagnostic threshold. A resolution sweep finds the first
degeneracy-free diagnostic setting at 256 for Avocado/Fox, 512 for
CesiumMan/RiggedSimple, and 2048 for WaterBottle; Lantern remains degenerate
through 8192. Higher resolutions are diagnostics only because the current BPT
grammar's fixed vocabulary supports 128 bins. This pilot establishes an input
eligibility/preprocessing gap, not general failure on artist meshes.
`artifacts/bpt/corpus-audit.json` records source hashes and per-resolution
collision/area counts; no faces were removed to make the check pass. None of the
six meshes becomes eligible at 128 bins under any of 24 proper axis-aligned cube
rotations, so the plan's quarter-turn augmentation alone does not resolve the
observed quantization losses.

Reviewed the primary BPT paper and EdgeRunner paper to resolve the plan's
fragmentation comparison. BPT aggregates faces around patch centers and reports
compression/locality; the cited 54.7 vs 199.5 subsequence counts are from
EdgeRunner's modified EdgeBreaker tokenizer. Local BPT patch counts therefore
cannot be directly scored against that target. The plan's <60 gate remains open
pending a specified common metric and an EdgeBreaker baseline. Documentation is
updated in `docs/bpt.md`; the local tokenizer is still an experimental format,
not a reproduction of upstream checkpoints.

## Track B expanded eligibility screen and tokenizer controls

Added three CC0 static samples from the official Khronos glTF Sample Assets
repository (Barramundi Fish, Boom Box and Sheen Chair), retaining each GLB,
README and LICENSE under `artifacts/bpt-candidates/`. Their SHA256 hashes and
per-resolution preflight counts are in `artifacts/bpt/corpus-audit.json`. The
geometry-only loader for Sheen Chair ignores material-only extensions while
rejecting rigging, animation and unsupported geometry features. All three new
models fail the 128-bin BPT eligibility check without exact zero-area source
faces: Barramundi Fish has 638 collapsed / 11 collinear faces, Boom Box 1,200 /
23, and Sheen Chair 20,634 / 228. The cube-rotation check found no safe
orientation. Also added the official [CC-BY 4.0 Box sample](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Box)
with its license/readme. This 12-triangle mesh passes at 128 bins under all 24
tested rotations, encodes to 36 tokens and three patches, and preserves its
oriented quantized faces. It is a simple box control rather than a representative
artist mesh. Added the CC0 [Scattering Skull](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/ScatteringSkull), a 188,871-face high-resolution source. It fails with 99,659 collapsed / 393 collinear faces at 128 bins. Added a 33,902-face CC0 Poly Haven Lantern 01 glTF candidate, with its referenced textures and source hashes preserved in `artifacts/bpt-candidates/PolyHaven_Lantern_01/`. It fails with 17,681 collapsed / 180 collinear faces at 128 bins. The updated screen covers nine static and three rigged assets; eleven of twelve fail the 128-bin gate, while none have exact zero-area source faces. This remains an eligibility sample, not a representative corpus.

Measured explicit QEM simplification on the three new meshes. It did not
reliably honor requested face counts and retained the same quantization
collapses for each source: Barramundi produced 3,808 faces for 2,000 and 1,000
face targets; Boom Box produced 5,999 for 6,000 and 5,458 for 4,000/2,000/1,000;
Sheen Chair remained at 38,646 faces for targets from 20,000 down to 1,000. All
reported sampled F-scores were 1.0 at the loose 0.01 tolerance, so those scores
do not establish quality after meaningful decimation. This is not a validated
repair path. The reproducible measurements are in
`artifacts/bpt/decimation-audit.json` and
`tools/benchmark_bpt_decimation.py`.

Added a clean synthetic closed-surface control via
`tools/benchmark_bpt_icosphere.py`. The 10,242-vertex / 20,480-face
five-subdivision icosphere is eligible at 128 bins, round-trips all oriented
quantized faces and closed edge incidence, and uses 35,996 tokens versus 184,320
raw coordinate tokens (80.47% reduction), with 3,551 BPT patches. This control
confirms the local encoder can encode a clean high-face closed mesh; it is
synthetic only, not curated data or real-asset quality evidence. The report and
tokens are `artifacts/bpt/icosphere_report.json` and
`artifacts/bpt/icosphere_tokens.npy`.

## Track B exploratory locality and entropy audit

Following the plan's Step 7 CR/PTME/AVD checks, added
`tools/benchmark_bpt_locality.py` and `artifacts/bpt/locality-audit.json`. The
report measures the CC-BY 4.0 12-face Box control and the existing synthetic
planar grid / 20,480-face icosphere. With the recorded local conventions (CR =
tokens / 9F; PTME = empirical base-2 token entropy normalized by raw coordinate-token
count; AVD@t = mean distance to all preceding t emitted vertex occurrences),
Box reports CR 0.333, PTME 1.306, AVD@5 1.164; planar grid reports CR 0.215,
PTME 1.514, AVD@5/25/50 0.040/0.097/0.164; icosphere reports CR 0.195,
PTME 2.293, AVD@5/25/50 0.214/0.336/0.381. The two synthetic
fixtures meet CR < 0.26 and PTME < 2.6 under this implementation; the Box is a
small tokenizer control, not evidence of the target on artist meshes. The
mixed three-fixture aggregate is explicitly exploratory. The paper does not publish the exact AVD averaging formula, so this is an
explicit operational interpretation, not a reproduction of upstream BPT's
protocol. Center-token patch counts are not
comparable to EdgeRunner subsequence fragmentation. A representative real-mesh
corpus and matched tokenizer baselines remain necessary.

## Track B artist-asset cleanup and locality extension

Queried Poly Haven's public model catalog and metadata; it lists 521 CC0 model
assets, including 311 with reported counts at or above 8,000 polygons. Screened four
additional stratified candidates and retained each 1k glTF, referenced buffer
and textures, CC0 note, and source hash manifest under
`artifacts/bpt-candidates/PolyHaven_{CoffeeTable_01,WoodenChair_01,food_lime_01,metal_toolbox}/`.
Counts span 8,100 to 19,992 faces. The updated corpus audit covers thirteen
static and three rigged assets: only the untouched 12-triangle Box passes the
128-bin gate; all other fifteen fail, with no exact zero-area source faces.

Implemented `tools/benchmark_bpt_quantized_cleanup.py` to explicitly remove
quantization-degenerate faces, optionally taking the union over all 24 proper
cube rotations, then verify retained oriented faces, topology, and sampled
surface distance after BPT round-trip. Avocado loses one of 682 faces (0.15%);
Food Lime loses 151 of 8,100 (1.86%). Both cleaned inputs are safe in all 24
orientations and preserve every retained oriented quantized face. At 1,024
area-weighted samples and 0.01 tolerance, their mean bidirectional surface
distances are 0.00170 / 0.00128 and F-scores are 1.0. The decoded Food Lime has
95 boundary edges and one nonmanifold edge, which remains a topology defect.
The source files are unchanged; cleaned token/face NPZ fixtures and full
measurements are in `artifacts/bpt/cleaned_*_all_24_rotations.npz` and
`artifacts/bpt/quantized-cleanup-audit.json`.

Extended `tools/benchmark_bpt_locality.py` to include these two cleaned source
meshes. With CR measured against retained faces, Avocado has CR 0.253 / PTME
2.181 and Food Lime CR 0.218 / PTME 2.424. Both meet the local CR < 0.26 and
PTME < 2.6 targets; these are two filtered assets, not corpus estimates. AVD
remains an explicit interpretation of the paper's prose. Current artifacts
also retain the earlier untouched-source audit, QEM experiment, synthetic
icosphere control, and CC0 source manifests. A diverse representative artist
corpus and matched EdgeBreaker fragmentation baseline remain open.

## Track B paper-derived fragmentation audit

Added `tools/benchmark_edgerunner_fragmentation.py`, an independent CPU counter
for `B` subsequence starts from EdgeRunner's published Appendix A.1.1 traversal
pseudocode. It uses the local BPT-round-tripped topology and reports both
EdgeRunner-style subsequence starts and BPT patch starts: Box 2/3; planar grid
65/2,080; 20,480-face icosphere 27/3,551; cleaned Avocado 16/142. Food Lime is
rejected because the decoded mesh has a nonmanifold edge. The open-boundary
missing-twin rule is explicitly documented as an interpretation because the
paper initializes boundary vertices as visited but does not specify that case.
The local counts do not reproduce the paper's dataset averages (54.7 for
EdgeRunner and 199.5 for AMT); asset set, preprocessing, and face ordering
differ. Full per-case data and scope are in
`artifacts/bpt/edgerunner-fragmentation-audit.json` and `docs/bpt.md`.

## Track B point-conditioned causal Transformer

Added `t3d/bpt_ar.py`: an unordered XYZ point encoder with learned-query
attention pooling, a causal self-attention decoder with cross-attention, and
shifted teacher-forcing batches with explicit BOS/EOS/PAD IDs. The bounded
context baseline has no Hourglass shortening, sliding cache, constrained
sampling or DPO yet. Focused tests cover future-token isolation, point-order
invariance, padding-weighted loss and gradients through conditioning.

Built and submitted private Kaggle notebook version 1 at
`sumu143/transformer-native-3d-bpt-ar-probe` with source bundle SHA256
`743b3fc6f1f1620708eeae7b35f96f1145544617919602f893770c04bf3a154b`.
It runs the focused checks on CUDA, followed by 100 full-stream optimization
steps on cleaned CC0 Avocado conditioned on 4,096 sampled surface points.
Local PyTorch is unavailable. Kaggle version 1 completed on a Tesla T4 with
PyTorch 2.11.0+cu130 / CUDA 13.0. All three focused tests passed in 1.609 seconds.
The full-stream next-token loss fell from 8.70884 to 0.210439 after 100 steps;
teacher-forced training accuracy was approximately 100%. Downloaded evidence
and checkpoint are in `artifacts/kaggle-bpt-ar-v1`. Source and fixture SHA256
values match the submitted bundle and local cleaned input.
This is a single-asset learning probe and provides no generalization or generated
mesh evidence. See `docs/bpt_ar.md` for the model and notebook workflow.

## Track B cached generation and held-out probe

Added per-layer KV-cache inference and single-asset greedy/temperature generation
to `t3d/bpt_ar.py`. The point condition is encoded once per generation, and each
new token attends to all cached positions. Generation returns EOS separately
from context/token exhaustion; geometry still requires BPT decode validation.
Focused checks now include cached/full causal-logit parity and termination rules.

Submitted dedicated Kaggle notebook version 2 with source SHA256
`9d93e620dbec3d1fd3c04f9435fe00cee43bb7300b07e201619eb9e37cb7795b`.
After the same 100-step Avocado optimization, it reloads the checkpoint, measures
teacher-forced loss/accuracy on the complete Box and Food Lime streams, and
generates greedily for all three conditioning point clouds. Food Lime evaluation
uses nonoverlapping context-sized windows with learned positions reset per
window, so this is a documented local window metric. No ground-truth tokens are
provided during generation. Version 2 completed on Tesla T4 with PyTorch
2.11.0+cu130 / CUDA 13.0. All five focused tests passed in 2.289 seconds,
including cached/full causal-logit parity and generation termination.

The reloaded checkpoint greedily reproduced all 1,550 Avocado tokens exactly,
emitted EOS and decoded to 363 vertices / 681 faces. Held-out teacher-forced
loss/accuracy was Box 10.2779 / 0% and Food Lime 10.1292 / 0.05127% (full
15,604 targets evaluated in the documented windows). Both held-out conditioning
clouds generated the **exact Avocado training stream**, confirmed independently
against local fixture tokens; both exported meshes match independent BPT decode.
This is direct evidence of single-asset memorization and failure to generalize
the condition. Valid grammar and EOS termination do not establish conditional
mesh generation quality. Downloaded reports, token arrays, OBJ/NPZ meshes,
checkpoint and tests are in `artifacts/kaggle-bpt-ar-v2`; independent token and
mesh comparisons are in `independent-generation-audit.json`. Corpus training
with distinct assets and held-out reconstruction/generation metrics remains
necessary before treating this as a useful generator.

## Licensed, asset-disjoint Track B training pilot

Acquired official Khronos Duck and Cesium Milk Truck sources with source URLs,
SHA256 manifests, README and license notices. Duck is under the SCEA Shared
Source License 1.0; its full license and source documentation are retained and
bundled with the modified training data. Milk Truck is animated and is explicitly
rejected by the static input pipeline. No articulation is silently discarded.

Added `tools/build_bpt_training_pilot.py`. With a declared 5% face-deletion
limit across 24 proper cube rotations, it accepts Duck (4,212 to 4,096 faces;
2.75% removed), Food Lime (8,100 to 7,949; 1.86%), Avocado (682 to 681; 0.15%),
and the unchanged 12-face Box. It preserves retained source-face indices and
verifies every oriented retained quantized face through BPT decode. Four Y
quarter-turn variants per source have matching 4,096-point surface conditions.
Duck and Food Lime are training assets; Avocado is validation and Box is a
simple control. Rotations never cross source-based splits. Manifest hashes
and all sixteen point-cloud array shapes/finiteness were checked locally.

Added position offsets for causal crop training and a focused cache-parity test
with nonzero position offsets. `tools/train_bpt_multiasset.py` uniformly samples
eight training streams for 600 updates, using 512-token crops with explicit
first/final/random sampling. It evaluates every target of each asset's unrotated
stream, reloads its checkpoint, and performs unconstrained greedy generation
with explicit EOS/context-limit reporting. Full-context generation differs from
crop training and may expose history-length generalization defects.

Created private Kaggle dataset `sumu143/transformer-native-3d-bpt-pilot`,
confirmed ready. The initial 1.5MB embedded-data notebook received a generic
SaveKernel 400 response; switched to the existing mounted-dataset pattern.
Notebook version 1 at `sumu143/transformer-native-3d-bpt-multiasset` was accepted
with source SHA256
`e669da90c8b223552d8037b9b1a073baee50d5c07d98c2b09a612a350800a8b8`.
GPU results remain pending. See `docs/bpt_training_pilot.md`. This four-asset
set is an integration pilot, not the blueprint's curated research-scale corpus.

Duck's source-relative cleanup quality was measured separately with 1,024
area-weighted samples per direction at three seeds. Mean distances were
0.001290/0.001320/0.001291, F-score 1.0 at tolerance 0.01. Its quantized mesh
has 6 boundary edges and 4 nonmanifold edges. The face-loss gate is not a
topology-quality gate; production artist-data eligibility remains unresolved.
Full values: `artifacts/bpt-training-pilot-source-quality.json`.

## Multiasset outcome and grammar-constrained sampling

The multiasset v1 job completed on Tesla T4. Its downloaded report at
`artifacts/kaggle-bpt-multiasset-v1/bpt-multiasset-report.json` matches submitted
source SHA256 e669da90c8b223552d8037b9b1a073baee50d5c07d98c2b09a612a350800a8b8.
After 600 steps, Duck/Food Lime training loss was 7.0871/7.8501; held-out
Avocado loss was 8.8632 with 0.322% accuracy, and Box loss was 10.8667 with
0% accuracy. Every generated stream reached the 15,604-token context limit
and failed decode with a degenerate triangle. The report was retrieved; full
checkpoint and token artifact retrieval remains pending.

Added `t3d/bpt_grammar.py` with incremental patch-state masks, EOS eligibility,
and integer-coordinate triangle area checks. Integrated optional
`grammar_constrained=True` in autoregressive generation. It masks malformed
patch starts and repeated/collinear triangle vertices, but does not enforce
manifold topology, face uniqueness or conditioning fidelity. Context exhaustion
can still truncate a patch; it is reported separately from EOS. All three CPU
grammar tests passed, including random accepted-stream decode checks. The
model-level constrained-sampling test also passed in the later Kaggle T4 run.
Existing unconstrained runs remain a separate baseline.

## Same-checkpoint grammar sampling evaluation submitted

Added `tools/evaluate_bpt_constraints.py` and its separate notebook builder.
The evaluation mounts multiasset v1 outputs, checks checkpoint training-source
identity and the exact training manifest hash, and uses the same four unrotated
conditioning fixtures. It compares new constrained samples with the recorded
unconstrained baseline, retaining explicit EOS/context-limit status and decode
results. It does not train or alter the baseline checkpoint. Incremental JSON
results and token/mesh arrays are saved after each completed asset.

Private Kaggle notebook `sumu143/transformer-native-3d-bpt-constraints` version 1
was submitted with source SHA256
`51c6f2f6517c996a45f6fe64144468c40f2da9eb8129c23d928b2bb0f90017c1`.
It runs the model and grammar tests before evaluation. The job completed on
Kaggle's Tesla T4; its test log reports all 10 included tests passing. The
saved report is in `artifacts/kaggle-bpt-constraints-v1/bpt-constraints-report.json`.

## Independent constrained-output audit prepared

Added `tools/audit_bpt_generation.py` to require the complete expected asset
set, independently decode each saved token stream, check parity with the saved
mesh, and count zero-area and duplicate unoriented triangles. For decodable
outputs it measures bidirectional sampled distance to each hash-verified cleaned
quantized target at three seeds and records edge-incidence topology. It preserves
EOS versus context-limit termination so a syntactically valid truncated mesh is
not treated as completed generation. The independent artifact audit completed.
All three outputs that decoded have zero zero-area faces, but contain 392–464
duplicate unoriented triangles and hundreds or thousands of boundary/nonmanifold
edges. Food Lime and Box stopped at EOS; Duck reached the context limit despite
being decodable. Held-out Avocado also hit the context limit and its truncated
block/offset pair did not decode. Across three seeds, sampled mean distances
were about 0.063–0.077 for Duck, 0.073–0.077 for Food Lime, and 0.204–0.210 for
Box; F-scores were low (0.097–0.157, 0.097–0.147, and 0). Grammar constraints
prevent degenerate triangles, but these results do not show useful topology or
conditioning fidelity. They remain a small integration pilot, not validation
of the broader seven-step blueprint.

## Animated rig sampling and held-out SkinAE evaluation

Extended `t3d.import_rig.load_rig_glb` to sample core glTF translation, rotation,
and scale channels with STEP, LINEAR/quaternion-slerp, and CUBICSPLINE
interpolation. It evaluates the skinned surface using source inverse-bind
matrices, preserving one shared normalization across poses. Animated matrix
nodes, morph weights, multiple skins and required non-core extensions remain
unsupported.

Added `tools.audit_rig_animations.py`; it sampled five times per clip from
RiggedSimple, CesiumMan and Fox's Survey/Walk/Run animations. Maximum normalized
vertex movement from the first sample was 0.291, 0.554, 0.266, 0.271 and 0.571.
Face topology was unchanged at every sample, with zero nonmanifold edges in this
audit. This confirms real-source parser/skinning execution, not deformation
accuracy. Detailed measurements and source hashes are in
`artifacts/rigged-inputs/animation-audit.json`.

Built `artifacts/skin-pose-pilot` with 1024 barycentric surface samples across
five sampled poses per clip: 10 training pose sets from RiggedSimple/CesiumMan
and 15 validation pose sets from held-out Fox. Submitted private Kaggle notebook
`sumu143/transformer-native-3d-skinae-multi-pose-pilot` version 1 to Tesla T4.
All 8 included importer/SkinAE tests passed. Bundle source SHA256
`d25ec5991dcf3a3d7dbde1743c0daf2e63c6c47142e1ecd70026c34cdbe20150` and local
manifest SHA256 `1ab35fa09abea93f86158443c1a0856c983ffbb4b9b37912bdecb8c61ca8327f`
match the downloaded report. The best checkpoint was selected at update 500 of
1200. On 15 held-out Fox poses, KL improved 2.6204 to 1.0551 (uniform baseline
2.6297); weight MAE is 0.0369 and dominant-joint accuracy is 0.7291. Checkpoint
SHA256 is `f73a7be776b6ac653f2e248d78307eeddea833ea0c5284011e7a92bf2543e9eb`.
The corpus remains two training assets and one validation asset; articulated
quality benchmarks and independent deformation validation remain open.

## Checksummed container profiles

Added checksummed raw profile 3 and latent profile 4 while retaining readers for
legacy profiles 1 and 2. Both new profiles append SHA-256 over the complete
header and payload; the readers verify it before payload interpretation. New
`save` and `save_latents` writes validate fully, write to a sibling temporary
file, and atomically replace the destination. Checksums detect accidental
corruption and do not authenticate the file. Streaming and upstream `.vxz`
compatibility remain unimplemented. All 9 raw/latent container tests pass,
including payload bit-flip rejection, legacy profile parsing, round-trip, and
atomic save/load. A local smoke audit loaded all 70 existing `.t3d` artifacts
through the backward-compatible readers.

## Three-fold SkinAE source holdout on Kaggle

Added `--holdout-asset` to the pose trainer and submitted a three-fold Tesla T4
notebook; each source rig is held out once while the other two train. All 8
included importer/SkinAE tests passed. Fold reports match bundle SHA256
`111507df606fa4e2b475ee69daa29879d4ea840b98089e9b2bf039aa8d4cbf3f` and local
pose manifest SHA256 `1ab35fa09abea93f86158443c1a0856c983ffbb4b9b37912bdecb8c61ca8327f`.
An independent audit verified disjoint source splits, pose counts and all
checkpoint hashes. Held-out KL was 0.5558/2.3190 for CesiumMan, 1.0551/2.6297
for Fox, and 0.1498/0.3183 for RiggedSimple (model/uniform); mean KL was 0.5869
versus 1.7557 uniform. RiggedSimple weight MAE is 0.2056, so its KL and dominant
accuracy do not capture its remaining weight error. Full fold results are in
`artifacts/kaggle-skin-pose-cv-v1/independent-audit.json`. Three assets remain
insufficient for research-corpus or deformation-quality validation.

## Recovering joints from voxel articulation fields

Added `t3d.articulation.infer_skeleton_from_field`. It reconstructs joint
endpoints from the stored joint/parent displacement pairs, clusters nearby
endpoints under a caller-supplied tolerance, votes parent links, imposes stable
coordinate ordering, and rejects cycles or rigs above the configured size.
Seven articulation tests pass. An exact three-joint chain recovered exactly;
the same chain with 0.0005-unit Gaussian displacement noise recovered the
correct parent links and joint positions within 0.00028 units at tolerance
0.01. A source-matched Kaggle version 3 notebook then exercised sparse feature
sampling, hierarchy recovery and the frozen SkinAE decoder on a Tesla T4 using
checkpoint SHA256
`f73a7be776b6ac653f2e248d78307eeddea833ea0c5284011e7a92bf2543e9eb`. All 512
synthetic vertices had sparse support and decoded to finite normalized weight
distributions. An independent affine interpolation fixture had zero error, and
missing support was detected. All five included export/SkinAE tests passed.
Bundle SHA256 was
`a5fe59a78e3ee638972b458ce9b5108a608e5e20db9f12046c5e1478e815974b`; artifacts
are in `artifacts/kaggle-skin-export-v3/`. It does not test generated-model
field quality. See `docs/articulation-inference.md`.

## Full synthetic skinned raw-export probe on Kaggle

Extended `tools/probe_export.py` to run a skinned cube through the public
`export_raw` path, including CUDA dual-grid extraction, CuMesh cleanup and UVs,
reference-BVH PBR baking, sparse feature sampling, frozen SkinAE decoding and
GLB reload. Kaggle T4 version 2 passed all 16 included articulation/export/GLB/
SkinAE tests. The rigged output has 186 faces, three joints and normalized
weights (maximum row-sum error 0); SkinAE checkpoint SHA256 is
`f73a7be776b6ac653f2e248d78307eeddea833ea0c5284011e7a92bf2543e9eb`. An
independent Khronos glTF Validator run reports zero errors, warnings or infos.
The full report and output GLBs are under `artifacts/kaggle-rigged-export-v2/`;
bundle SHA256 is `a7bfe7d666746200ba0853a66d87b6b7ab10f4964295f41c345ec74b9ab3b762`.
This verifies integration on a synthetic cube only; it does not establish
predicted-field accuracy, meaningful deformation, real-rig material/animation
transfer, or production quality.

## Real-source rigged export and strict-remesh validation on Kaggle

Submitted Kaggle T4 version 4 from source bundle SHA256
`a41a18b61973b994902094a701a2eed04c4bc23b4b873306383ce7cff1a3795b`. The
source-matched probe ran 19 focused tests across articulation, export, GLB,
surface metrics and SkinAE. A real RiggedSimple source (160 vertices, 188
faces, 2 joints; SHA256
`3a79dabb67bb0cd598a18d08b954d9d357c27c30672f82ef5d3f4e7fe6ca3401`) passed
the in-memory raw Track A export path and reloaded as a 654-vertex, 988-face
skinned GLB. At 512 samples and tolerance 0.01, mean surface distance was
0.000458 and F-score 1.0; position-welded topology had zero boundary and
nonmanifold edges. The held-out RiggedSimple SkinAE checkpoint
`95bc334058ada68c97800fcd5c3c83998059a03ea1d69e9e033a6c20ac4439a7` produced
weight MAE 0.20713 and KL 0.15845, showing that the decoded continuous skin
weights remain a material quality gap. Added `tools/audit_skin_export.py` for
an independent closest-triangle weight target and nearest-joint baseline audit.
On the same 654 output vertices, SkinAE has MAE 0.20713 and dominant-joint
accuracy 0.9021; nearest-joint one-hot assignment has MAE 0.31058 and accuracy
0.6865. The strict-remesh branch has SkinAE MAE 0.20821/accuracy 0.9020 versus
nearest-joint 0.31776/0.6807. The audit JSONs are retained alongside the GLBs.
Thus SkinAE beats this geometry-only baseline on this rig, while its continuous
weight errors remain substantial; this is not a deformation-quality test.

The same probe enabled strict remeshing. It produced 966 faces with mean
surface distance 0.004596 and F-score 0.87596 at the same tolerance; welded
boundary and nonmanifold counts were both zero. Both GLBs reloaded with
normalized skin weights and independently passed Khronos glTF Validator with
zero errors, warnings or infos. The report, source test logs, CUDA dependency
build log, validator summaries, both GLBs and a hash manifest are in
`artifacts/kaggle-real-rigged-export-v4/`.

Added `tools/audit_export_surfaces.py` and repeated the 512-sample surface
comparison at seeds 0–4. Standard export mean distance ranged 0.000437–0.000464
and F-score stayed 1.0. Strict remesh mean distance ranged 0.004220–0.004736
and F-score ranged 0.865–0.909. Every run reported zero boundary and
nonmanifold edges after position welding. This checks seed sensitivity on this
single asset; it is not the full blueprint benchmark suite.

This is one source rig at grid 32, with only 512 surface samples. The probe
replaces source PBR with a constant material and does not transfer or exercise
animation. It does not measure generated-field quality, viewer deformation, or
production-scale benchmarks. The separate real `.t3d` CLI extension is verified
in version 6 below. Treat the measurements as an integration pilot, not
completion of Steps 6 or 7.

## Real-rig checksummed-container CLI run submitted

Submitted Kaggle version 5 from source bundle SHA256
`82f453990d8b3aaa7c74e1b198a1f2ac9c5610143478007539e7956fa143e814`. This
extends the v4 probe by serializing the real RiggedSimple-derived raw asset to
the checksummed `.t3d` container, invoking `python -m t3d.export` with the
held-out SkinAE checkpoint, and checking input/output hashes, joint count,
normalized weights and sampled surface metrics. The job was confirmed RUNNING
after submission; its report and independent GLB validation are pending. A
local quantization preflight found that the submitted explicit `1e-6` joint
tolerance is invalid after the container stores the six-channel field as
float16. The terminal v5 log confirms export completed but inferred 10 joints
instead of the source's 2, and the probe rejected the output. Its GLB passed the
structural Khronos validator, demonstrating that structural validity does not
catch the semantic skeleton error. The failed run, `.t3d`, GLB and logs are
retained under `artifacts/kaggle-real-rigged-export-v5-failed/`. On CPU, the
documented voxel-based default (`0.0078125` for this grid) recovers the correct
two-joint hierarchy; `0.0005` also recovers it within `2.2e-5` units. Added a
CPU regression test for float16 field recovery at the documented default
tolerance; it passes locally.

## Corrected real-rig `.t3d` CLI run submitted

Submitted Kaggle version 6 from source bundle SHA256
`0195d63d8acd18731edbf363bb7e997a41ee017c33070b60efb61395c23694ea`. It uses
the voxel-based default tolerance and includes the new float16 field regression
test. The focused suite now contains 20 tests. The job was confirmed RUNNING;
the checksummed-container CLI output is a 659-vertex, 992-face skinned GLB with
2 recovered joints and normalized weights. The CLI audit confirms the input
container hash, and new `tools/audit_rigged_container.py` independently loaded
the `.t3d`, recovered the exact `[-1, 0]` parent hierarchy and joint positions
within `3.4e-5` units. The output passed input-hash verification;
its surface comparison has mean distance 0.000468 and F-score 1.0 at tolerance
0.01, with zero welded boundary/nonmanifold edges. The independently run
Khronos validator reports zero errors, warnings or infos on the CLI output;
standard and strict-remesh direct outputs also pass. The local weight audit on
the CLI output measures SkinAE MAE 0.20730 and dominant-joint accuracy 0.8983,
versus nearest-joint baseline MAE 0.30998 and accuracy 0.6904. The run report,
checksummed `.t3d`, GLBs, test/build logs, validator JSONs and weight audit are
retained in `artifacts/kaggle-real-rigged-export-v6/`. The 20 focused tests
passed on Tesla T4. The run covers the constructed real-rig raw container and
CLI path, while source material transfer, animation, deformation validation,
large-scale benchmarks, and full Steps 6–7 remain incomplete.

## Rigged source PBR import and textured extension

Extended `load_rig_glb(..., include_pbr=True)` to retain supported per-primitive
glTF base-color, metallic, roughness and alpha factors, face assignments and
corner UVs for voxel `sample_pbr`. RiggedSimple contains a factor-only material (base color
`[0.2796, 0.6400, 0.2109, 1]`, metallic 0, roughness default 1); local CPU
voxelization and sampling reproduced those six-channel values across 584 active
voxels. Added unit coverage for factor preservation and explicit textured-input
texture preservation; all six `test_import_rig` tests passed locally.

Version 7's 26 focused tests passed on Kaggle T4, but the real-rig export failed
at `pad_pbr`, which rejected a baked image outside `[0,1]`. The source material
contains channels exactly at 1; likely float32 interpolation drift, so the
baker now clamps only excursions within `1e-5` and rejects larger ones. The
failed log and tests are retained in `artifacts/kaggle-rigged-export-v7-failed/`.

Extended rig PBR import to decode embedded base-color and metallic-roughness
textures with `TEXCOORD_0` plus repeat/clamp samplers, using bounded GLB image
bufferViews or data URIs. Nonzero emissive, normal/occlusion maps, material
extensions and unrepresentable wrapping/UV transforms are rejected. Local
CesiumMan and Fox GLBs both decode 1024² base-color textures. A CPU grid-32
CesiumMan material projection produced 918 active voxels and 326 distinct voxel
colors, with all six values bounded. `sample_pbr` now accepts the exact BVH
projection path for larger source meshes; a seven-case material suite passes.

The first v8 upload was rejected by Kaggle because bundling the 438 KB textured
GLB plus checkpoint pushed the notebook over its upload limit. Created private
dataset `sumu143/transformer-native-3d-textured-rig-inputs` with the source,
license/source metadata, held-out checkpoint and hashes; Kaggle reports the
dataset `ready`. The rebuilt notebook mounts that dataset, reducing notebook
source to 137 KB. Submitted the corrected Kaggle T4 version 8 with source bundle
SHA256 `95b45d7b89b60184971e017de31890e31dc78274f3597a79b440cb204e73147a`.
It switches the integration source to held-out CesiumMan, exercises its
embedded texture and includes importer/material tests. All 33 tests passed, but
the export stopped in GLB skin packing because dense softmax weights exceeded
the packer's 16-influence limit. The failed outputs and logs are retained in
`artifacts/kaggle-textured-rig-export-v8-failed/`.

Added `t3d.glb.limit_skin_influences` to explicitly keep the highest 16
per-vertex weights, renormalize them and report the discarded mass; direct
`attach_skin` and `pack_glb` behavior remains lossless and still rejects
unrepresentable inputs. Its local top-k/normalization test passes. Kaggle T4
version 9 was submitted with bundle SHA256
`08a71010d2fae3e2070a13bc89d3f93b2854727698491e2569e36e68db18a449`; it
includes this decoder boundary policy and the GLB regression test. All 34
focused tests passed. The real CesiumMan export retained 19 joints, with a
source skin-weight MAE of 0.0407. Its standard output has 1,818 faces, mean
sampled surface distance 0.000922 and F-score 0.9971 at tolerance 0.01; 512
surface samples were used. Top-16 influence truncation discarded mean mass
0.000125 and maximum mass 0.002324, with normalized output rows. The strict
remesh output has F-score 0.8074 at the same sample/tolerance and two welded
nonmanifold edges.

The v9 report's scope string incorrectly said source PBR was replaced by a
constant. Independent reloading and structural audit of the actual GLBs show
that the 1024² source base-color map was sampled into sparse PBR values and
rebaked into a varying 128² output texture (5,260 unique RGB values; channel
standard deviations 0.151, 0.101, 0.119). The source has no metallic-roughness
texture, and the output metallic/roughness map is constant. This confirms map
embedding and variation, not perceptual texture fidelity or HDRI relighting.
The standard, strict-remesh and checksummed-container GLBs each pass the
independent Khronos validator with zero errors, warnings or infos. Reports,
inputs, GLBs, test/build/kernel logs, texture audit, validators and hashes are
retained in `artifacts/kaggle-textured-rig-export-v9/`.

Corrected the probe scope description and added texture audits for direct,
strict-remesh and `.t3d` CLI output. Kaggle T4 version 10 completed with source
bundle SHA256
`e69f7378c4b831737c100cae90b7507beefa51a36a0b298f358b70e47241de6f`; all 34
focused tests passed. The GPU report includes separate audits for all three
outputs, confirming varying embedded base-color maps (5,260 / 5,162 / 5,302
unique RGB values for standard / strict / `.t3d` CLI) and constant metallic /
roughness maps. Each GLB again passes independent Khronos validation with zero
errors, warnings or infos. The complete run is retained in
`artifacts/kaggle-textured-rig-export-v10/`. These audits establish structural
retention and variation only; no rendered material-fidelity metric is claimed.

Added `tools/audit_material_fidelity.py` to compare area-weighted samples on an
exported surface against source PBR sampled at the closest source triangles,
after the atlas UVs have been repacked. Across five seeds of 4,096 points on
the v10 standard 32³ output, mean linear base-color RMSE is 0.2978 / 0.2296 /
0.2473 for R/G/B (seed standard deviations 0.0007 / 0.0010 / 0.0025), with
mean source projection distance 0.00091 and average p95 0.00356. Metallic,
roughness and alpha match their constant source values. Strict remeshing has
similar color error but a larger p95 projection distance (0.0240), reinforcing
that its material comparison includes greater geometric deviation. These are
surface attribute comparisons, not rendered-image scores.

Extended the real-rig probe with grid/texture size controls, three material
fidelity audits and a 64³ / 1024² run configuration. Kaggle T4 version 11
completed with source bundle SHA256
`387d6b7a8db77be16b51dd42573052f7d3c992e300b396813a1fdd11bc474d3a`; all 34
focused tests passed and the standard, strict-remesh and `.t3d` CLI GLBs pass
independent Khronos validation without errors, warnings or infos. Across five
seeds of 4,096 area-weighted output-surface points, base-color RMSE is 0.1978 /
0.1535 / 0.1650 for R/G/B (standard deviations 0.0030 / 0.0019 / 0.0023),
about 33% lower per channel than v10's 32³ / 128² result. At 512 samples and
tolerance 0.01, the standard surface F-score is 1.0 and mean distance
0.000147, but the mesh still has 59 boundary and 91 nonmanifold edges. Strict
remesh has F-score 0.963 and zero boundary edges but 16 nonmanifold edges.
These findings show that higher resolution improves pointwise color retention
without establishing topological or rendered quality. The run and audits are
retained in `artifacts/kaggle-textured-rig-export-v11/`.

Applied the plan's duplicate-face removal, nonmanifold repair, small-component
removal, hole filling and face-orientation pass after strict remeshing. Kaggle
T4 version 12 completed with source bundle SHA256
`d83b438da0f0dd7050c5d54b7aa5cf1decb0b543e72c3f55c401c5f6915a8d65`; all 34
focused tests passed and all three GLBs pass independent Khronos validation
without errors, warnings or infos. On the strict output, the cleanup changed
the welded nonmanifold count from 16 to 16 and boundary count stayed zero. It
added 26 vertices during hole filling and slightly improved the 512-sample
surface F-score from 0.963 to 0.976 (mean distance 0.00227). This does not
resolve the strict branch's nonmanifold edges. The standard and `.t3d` CLI
outputs remain unchanged. The source, reports, logs and outputs are retained
in `artifacts/kaggle-textured-rig-export-v12/`.

The next resolution sweep uses grid 128³ and a 2048² atlas, matching the
blueprint's example atlas size. Kaggle T4 notebook version 13 completed with
source bundle SHA256
`d83b438da0f0dd7050c5d54b7aa5cf1decb0b543e72c3f55c401c5f6915a8d65`; all 34
focused tests passed and all three outputs passed independent Khronos
validation with zero errors, warnings or infos. Across five seeds of 4,096
area-weighted surface points, base-color RMSE is 0.1220 / 0.0955 / 0.1014 for
R/G/B (standard deviations 0.0025 / 0.0024 / 0.0052), versus 0.1978 / 0.1535 /
0.1650 at 64³ and 0.2978 / 0.2296 / 0.2473 at 32³. The standard output has
9,306 faces, 19 joints, skin-weight MAE 0.0407, 512-sample F-score 1.0 and
mean distance 0.0000587; position-welded topology still finds 16 nonmanifold
edges. Strict remesh has F-score 1.0, mean distance 0.000882, one boundary
edge and nine nonmanifold edges after cleanup. The output embeds a 2048²
base-color atlas with 72,581 unique RGBs. These are single-asset, low-grid
results without rendered relighting or animation evaluation; full artifacts
are in `artifacts/kaggle-textured-rig-export-v13/`.

Earlier factor-only run source bundle SHA256
`01570acbd64450c6d845346d71e4f75115350c4c8d5c091298136527e49d3bfa`. The
v7 real-rig bake failed as described above; v9 is the first completed
textured-source full-rig export on Kaggle.

## Source-clip retargeting and posed-deformation audit

Added `rig_joint_world` to `t3d.import_rig` and
`joint_deformation_matrices` to `t3d.articulation`. The former returns skin
joint node ids and world matrices at bind or a sampled animation time; the
latter composes bind-relative world deformations from those matrices. Added
`tools/audit_posed_deformation.py`, which maps source joints to an exported
rig by normalized bind position, verifies the parent hierarchy after that
permutation, transfers source joint motion through the exported weights, and
compares the deformed export surface against the source's own posed surface.

On the CesiumMan source (SHA256
`b7001eaeea8254bd44773bcd247e78696d94169388fbb2a1800fc69434e777d9`) and the
v13 standard export (SHA256
`52bebe3dc8c2675e028124bbd43d62db294f9661e9015f3953833fce62f6bff3`), the
19 joints match one-to-one at mean position distance 0.00080 (maximum 0.00085)
with zero parent mismatches after permutation. Bind-pose surfaces align at
0.00042 mean canonical distance with F-score 1.0. Across five samples of the
single CesiumMan clip, source vertex motion has mean 0.070–0.109 and the
retargeted export moves similarly (0.091–0.100). Using the exported SkinAE
weights, the posed surface error is 0.0160–0.0188 mean distance (mean 0.0179)
with F-score 0.56–0.62; the static-export baseline is 0.0456–0.0550.
Projecting source weights onto the export surface as an oracle control gives
0.00049–0.00053 mean distance and F-score 1.0 at every sampled time. The SkinAE
weights differ from that transfer by 0.0407 mean absolute weight (maximum
0.721), with 0.842 dominant-joint agreement.

Added `t3d.animation` with `append_trs_animation`, `validate_animation` and
`retarget_animation`. The retargeter samples source bind-relative joint motion
at every source key time, composes it onto the output rig's bind world matrices,
decomposes each local transform into translation/rotation/scale, and appends
LINEAR glTF tracks to the packed GLB. `validate_animation` checks sampler and
channel references, accessor spans, ordered finite times, positive scales and
unit rotations without claiming Khronos validation.

Six cases were audited. For the CesiumMan-to-v13 reconstructed export the posed
result is unchanged: SkinAE weights give 0.0179 mean canonical distance
(F-score 0.60) versus 0.00052 with source-transferred oracle weights. The
exported 57-channel, 48-key animated GLB reloads within 3.3e-6 maximum vertex
difference of the in-memory deformation and reproduces the sampled surface
metric to 4e-7. Same-topology packed round-trip rigs (RiggedSimple, CesiumMan
and Fox Survey/Walk/Run) retarget at 5.8e-9 to 1.1e-8 mean canonical distance
with F-score 1.0 and reload within 5.4e-7, confirming the animation writer and
joint-space mapping across three rigs and five clips.

This isolates the retargeting and encoding math from the learned weights: the
composition is correct to surface and float precision, and the remaining
articulated error on a reconstructed export is dominated by SkinAE weight
quality. It covers five samples per clip and 512 samples per surface
comparison in canonical units. It does not test viewer rendering, Khronos
validation of animated output, broad deformation benchmarks, retrained SkinAE,
or additional reconstructed exports.

## Influence-limited SkinAE weights

Added `t3d.glb.rewrite_skin_weights`, which rewrites existing `JOINTS_n` /
`WEIGHTS_n` accessors in place without changing geometry, materials, animation
or the joint hierarchy, and `tools/reweight_skin_glb.py`. `t3d.export.export_raw`
and its CLI now accept `--max-skin-influences` (default 16, the previous
behavior). The rewriter uses `limit_skin_influences`, so the top-k weights are
renormalized and the discarded mass is recorded.

On the v13 reconstructed export, limiting SkinAE weights monotonically improves
the correspondence audit at the five sampled times:

| Influences | Mean distance | F-score | Animated round-trip |
| ---: | ---: | ---: | --- |
| 16 (dense) | 0.017868 | 0.6032 | valid |
| 4 | 0.016121 | 0.6131 | valid |
| 3 | 0.013876 | 0.6713 | valid |
| 2 | 0.010919 | 0.7786 | valid, max vertex delta 4.0e-6 |
| 1 | 0.009789 | 0.8436 | invalid at t=0.0417 (degenerate triangle) |

Top-2 cuts the measured mean distance by 38.9% and raises F-score by 0.175
relative to dense weights; the oracle source-transfer control stays at 0.00052
/ 1.0. Top-1 is numerically lower still but is not safe: its animated reload
contains a zero-area triangle at the first sampled time, which is why the audit
now reports invalid round-trips rather than crashing. This supports weight
sparsification as a partial mitigation, not a fix for SkinAE quality. The
comparison is still correspondence-based, not rendered or artist quality, and
top-2 can introduce harder joint transitions; those need viewer/render checks
before changing the export default.

Reports, influence-limited GLBs and animated GLBs are under
`artifacts/posed-deformation/`; the combined manifest SHA256 is
`38f011bb8391cb23b175407ad3edda84a20c6ff75605d4e1750b23bd2cf93809`. All 149 local tests pass with 32 optional-dependency skips.

## Continuation review and independent animated GLB validation (2026-11-03)

Read `note.md`, the continuation evidence in this log, and the animation
implementation. Checked all 51 entries in the continuation manifest: every
file size and SHA256 matches; manifest SHA256 remains
`38f011bb8391cb23b175407ad3edda84a20c6ff75605d4e1750b23bd2cf93809`.
The reported 149 passing tests / 32 skips belong to the prior session; they
were not rerun during this review.

Ran the independent Khronos glTF Validator on all 11 saved animated GLBs.
Every file reports zero errors, warnings and infos. Per-file validator reports,
`animated-validator-summary.json`, and a separate SHA256 evidence manifest
are saved in `artifacts/posed-deformation/`; the original continuation manifest
is preserved. This closes the missing structural validation of animated output.
It does not establish rendered deformation quality: top-1 still has the
previously measured degenerate triangle under motion despite structural
validation passing. Top-2 remains opt-in, and SkinAE weight quality, visual
deformation evaluation, mesh topology, and the broader Part 2 requirements
remain unfinished.

## Source skin-weight sparsity inspection (2026-11-03)

Inspected the 1,024 sampled source-weight rows per asset in the existing pose
pilot corpus; results are saved in `artifacts/posed-deformation/source-weight-sparsity.json`.
CesiumMan has 3.39 weights above 1e-4 per sample on average; retaining only the
top two discards 6.24% of source weight mass on average, versus 1.45% for
top three and 0.165% for top four. Fox loses 5.20%, 0.227%, and effectively
zero respectively; RiggedSimple has only two joints. Thus top-two is a useful
empirical mitigation of this checkpoint's error, but source supervision itself
does not generally justify a universal two-influence constraint. These are
three assets, not a representative corpus; no new training or rendering was
performed. Continue with deformation-aware training and visual assessment.

## Blender rendered influence comparison (2026-11-03)

Added `tools/render_skin_comparison.py` and rendered five sampled CesiumMan
poses in Blender 5.2.2 Cycles, with dense v13 weights on the left and top-two
weights on the right. Images and capture settings are in
`artifacts/posed-deformation/render-comparison/`. Both inputs are the already
validated animated exports, imported and rendered in the same scene at 24 fps.
The first and middle pose show conspicuous sharp folds and protrusions around
the top-two leg and arm; the dense version also shows distorted crossing legs.
These frames support keeping top-two opt-in despite its lower surface-distance
score. This is a single frontal camera, one asset and five sampled poses;
there is no source reference in the images or objective rendered metric, so
it cannot establish overall visual quality or attribute every defect to weight
limiting. No HDRI relighting, broad asset evaluation or training was performed.

## Source-reference Blender comparison (2026-11-03)

Added `tools/render_skin_reference.py` and rendered the original CesiumMan,
the dense-weight reconstruction and the top-two reconstruction side by side
at the five audit times. Source normalization is applied in Blender's Z-up
coordinates using the importer's bind-pose transform. The images and settings
are in `artifacts/posed-deformation/render-reference/`. In the first frame the
source legs are separated while the reconstructed exports appear crossed or
folded, and the reconstructed base-color map is visibly less detailed. This
indicates a substantial viewer-level discrepancy. It is not yet a controlled
numeric render benchmark: confirm Blender's imported animation timing and
skinning evaluation against the Python pose audit before assigning the visual
mismatch entirely to reconstruction; only one asset and frontal view are shown.

## Blender pose agreement with Python importer (2026-11-03)

At Blender frame 25 (24 fps), compared evaluated world-space mesh vertices
against `load_rig_glb(..., animation_index=0, time=25/24)` after converting
Blender Z-up to glTF Y-up. Nearest Python-vertex distances have mean / maximum
1.91e-7 / 4.67e-7 for source CesiumMan, 7.23e-8 / 3.62e-7 for dense v13,
and 8.25e-8 / 3.47e-7 for top-two. Each comparison covers the full corresponding
mesh vertex count. Blender's generated Icosphere bone widgets must be excluded;
including them produces misleading large outliers. Blender frame 25 agrees with
25/24 seconds, while comparing against 24/24 seconds gives visibly larger
errors. This confirms agreement for that sampled pose and supports the rendered
comparison's timing; it does not validate all times or render quality. The
inspection script is `tools/inspect_blender_poses.py` (writes temporary arrays).

## Rendered source-weight oracle control (2026-11-03)

Added `tools/build_oracle_skin_control.py`: it projects original CesiumMan skin
weights onto the reconstructed bind surface, remaps joints and rewrites the
existing animated v13 GLB. `tools/render_skin_oracle.py` renders source, learned
weights and this oracle control side by side for the five audit times. The first
frame shows the oracle control's leg pose closely following the source, while
the learned-weight export's legs cross. This visually corroborates the prior
quantitative oracle comparison and isolates skin weights as a major cause for
this asset. The control requires the original rig weights and is not a learned
or generative solution. Images are in `artifacts/posed-deformation/render-oracle/`;
only one frontal view and one asset are covered, with no broad benchmark claim.

## SkinAE joint-position loss ablation (2026-11-03; running)

Added an optional nonnegative `--joint-position-loss` to the pose trainer. It
adds the squared difference between predicted and target weight-averaged joint
positions to the existing KL objective; zero preserves the baseline objective.
Evaluation now records mean joint-position error. This is a pose-dependent proxy,
not the actual linear-blend-skinning deformation loss. The new notebook builder
`tools/build_skin_position_notebook.py` bundles a paired 0 versus 1.0 loss
comparison for each of the three source-held-out folds, with the same seed and
1,200 steps per run. Best checkpoints are selected using held-out KL as before.
Kaggle kernel `sumu143/transformer-native-3d-skinae-cross-validation` version 2
was submitted and confirmed RUNNING; its source bundle SHA256 is
`5c7a8f11b098da06b181424f7ea205adf274b87dbeb74036a62a25953b678d83`.
Version 1's downloaded evidence remains under `artifacts/kaggle-skin-pose-cv-v1/`.
No outcome is available yet; inspect this same job before submitting another.

Version 2 failed before training: the bundled `import_rig.py` now imports
`materials.py`, absent from the older notebook bundle. Updated the ablation
builder to bundle all `t3d/*.py` modules and submitted version 3 with source
bundle SHA256 `189903e12766d96fb1c877784642bd01a6e5801ee80ab2d6ff51475d36cb51b2`.
No training results were produced by version 2; inspect version 3's status and
logs before any further submission.

## Animated vertex evaluator and prior checkpoint baseline (2026-11-03)

Version 3 passed all 10 bundled checks and proceeded into training; latest
Kaggle status remains RUNNING. Added `tools/audit_skin_pose_checkpoints.py`,
which loads each asset-held-out checkpoint, encodes/decodes original bind
weights, then compares predicted versus target-weight LBS vertices at five
times per clip in canonical coordinates. This measures actual articulated
vertex error on original topology, not reconstructed or generated geometry.
It verifies the asset was held out in checkpoint metadata. Installed CPU
PyTorch locally for this evaluation; model training remains on Kaggle.

On the preserved version-1 checkpoints, mean animated vertex error is 0.02987
for CesiumMan, 0.02471 for Fox, and 0.03344 for RiggedSimple; corresponding
bind-weight MAEs are 0.04732, 0.03124 and 0.30134. The result is saved in
`artifacts/kaggle-skin-pose-cv-v1/deformation-audit.json`. This is a different
evaluation from the reconstructed-v13 surface audit and should not be compared
numerically as the same metric. Apply this evaluator to both loss settings
once version 3 completes; no new checkpoint is promoted yet.

## Completed joint-position ablation, Kaggle version 3 (2026-11-03)

Kaggle version 3 is COMPLETE; outputs and logs downloaded successfully to
`artifacts/kaggle-skin-position-ablation-v3/`. All six checkpoint SHA256 values
match their training reports and all source bundle hashes match the submitted
`189903e12766d96fb1c877784642bd01a6e5801ee80ab2d6ff51475d36cb51b2`.
All 10 notebook checks passed. Paired CPU LBS evaluation on held-out source
bind vertices gives mean canonical vertex error baseline -> penalty:
RiggedSimple 0.033444 -> 0.031431 (6.0% lower), CesiumMan 0.029869 -> 0.027191
(9.0% lower), Fox 0.024708 -> 0.023025 (6.8% lower). Held-out KL also improves
for all three folds. Results and hashes are saved as `deformation-audit.json`,
`paired-deformation-comparison.json` and `evidence-manifest.json`.

This supports the auxiliary penalty on this three-asset, single-seed pilot.
Checkpoint selection used validation KL, so this is validation evidence, not
an untouched test set. It does not establish reconstructed export, source-free
generation, multi-seed robustness or broad corpus quality. Checkpoints are
retained as candidates; the existing export checkpoint is not replaced.

## Reconstructed export with joint-position candidate (2026-11-03; running)

Built `tools/build_rigged_position_export_notebook.py` to bundle the
CesiumMan-held-out loss-1.0 checkpoint from the version 3 ablation. It re-runs
the complete 128³ / 2048² GPU export and re-encodes skin fields with the same
checkpoint used to decode them, preserving the matching latent space. Kaggle
`sumu143/transformer-native-3d-rigged-export-probe` version 14 is RUNNING;
source bundle SHA256 is
`2396b21e9e73b2fc45f82490ac01975b4a9367a5fe2f1f1dbd9307f9d29af61f`.
Do not infer a quality improvement before the run and posed deformation audit
complete. The candidate was selected using this asset as validation, not an
untouched test asset. Inspect this same job before any retry.

## Animated-vertex reference cross-check (2026-11-03)

Added `tools/audit_lbs_reference.py` to compare the checkpoint evaluator's
bind-relative matrix LBS reference against directly imported source glTF poses.
Across five RiggedSimple poses, five CesiumMan poses, and fifteen Fox poses,
maximum per-vertex canonical discrepancies are respectively 6.6e-9, 7.4e-8,
and 2.0e-8. Detailed results are in
`artifacts/kaggle-skin-position-ablation-v3/lbs-reference-audit.json`. This
checks the reference construction used for the prior vertex-error comparison;
it does not broaden its three-asset or validation-set scope.

Version 14 reached COMPLETE. Its downloaded standard GLB has SHA256
`e6a5b3b5eefc18bd96f5e453e3f9b49f3205c8486878bca250a8d6a055f87d3c`,
matching `export-probe.json`; the 37 bundled checks passed. Artifact download
and `tools.compare_candidate_export` were started and were still running at
this update. Inspect those processes and their completed output before drawing
a candidate-versus-v13 deformation conclusion.

Version 14 artifact download finished. Independent Khronos validation of the
standard, strict and container GLBs returned zero errors, warnings and infos;
individual reports are stored beside the GLBs. The CPU posed comparison is
still running; no deformation result is asserted yet.

The completed five-pose comparison of reconstructed CesiumMan gives v13 vs
candidate mean surface distance 0.017868 -> 0.016589 (7.2% lower) and F-score
0.6032 -> 0.6396 at tolerance 0.01. Candidate animation round-trip has maximum
vertex difference 3.39e-6 with no invalid sampled poses. The exported candidate
and v13 have independently reconstructed geometry and the asset was used for
checkpoint selection; this is a single-asset validation comparison, not a
causal isolation or proof of generalization. The source-weight oracle remains
0.00053 / F-score 1.0, so considerable deformation error remains. Full report
and animated GLB are in `artifacts/kaggle-textured-rig-export-v14/posed-comparison/`.

## Candidate rendered pose inspection (2026-11-03)

Rendered the source, v13 baseline and v14 candidate side by side using
`tools/render_skin_candidate.py`; the five frames and capture settings are in
`artifacts/posed-deformation/render-candidate-v14/`. Inspection of the first
and middle poses shows both reconstructed exports with severely crossed or
folded legs while the source legs are separated. The 7.2% sampled surface
improvement therefore does not resolve the visibly incorrect articulation.
This is a single frontal view of one asset; it is not a multi-view rendered
benchmark or a basis for promoting the candidate to production.

## Animated LBS loss ablation (2026-11-03; running)

Prepared `artifacts/skin-deformation-pilot/` with canonical bind sample points
and per-pose bind-relative joint matrices, using hash-verified source rigs.
Added optional `--deformation-loss` to the SkinAE pose trainer: it penalizes
actual LBS position differences between predicted and source weights for each
sampled bind point and pose; zero preserves the original KL baseline.
Both target and prediction use the same interpolated bind point and weights.
Barycentric interpolation and LBS do not commute: maximum discrepancy versus
interpolated posed source triangles reaches 0.0535 for RiggedSimple, 0.00561
for CesiumMan, and 0.0422 for Fox. Thus this is an approximate sampled-point
LBS objective, not an exact posed-surface target. Data and Python files
compiled; no quality claim is made before remote execution.

Submitted Kaggle `sumu143/transformer-native-3d-skinae-cross-validation`
version 4, source bundle SHA256
`dc317982b8015614d3fb8e72eafd3eeeb23c6195de71e4c8ec2dd246e006f258`.
It compares loss strengths 0 and 1 across three held-out folds; latest status
RUNNING. Inspect this exact job before resubmitting. Prior version 3 artifacts
remain archived locally.

Version 4 remains RUNNING and its 10 bundled checks passed. Independently
checked the new Torch LBS penalty on 32 CesiumMan samples against the NumPy
`linear_blend_skinning` implementation: losses 0.0239728726 and 0.0239728747
(difference 2.1e-9); gradients were finite and nonzero (norm 0.00596). This
verifies the loss computation for that case, not training or quality outcomes.

Version 4 completed; its ten bundled checks passed, and all six downloaded
checkpoint hashes and source bundle hashes match their reports. Evaluating the
paired checkpoints on original held-out vertices gives baseline -> deformation
penalty mean canonical LBS error: RiggedSimple 0.033444 -> 0.039446 (17.9%
worse), CesiumMan 0.029869 -> 0.030479 (2.0% worse), Fox 0.024708 -> 0.022817
(7.7% better). Checkpoints were selected on validation KL; the same three
assets, one seed and sampled-point approximation limit the result. This setting
does not consistently improve held-out deformation and is not promoted. See
`artifacts/kaggle-skin-deformation-ablation-v4/paired-deformation-comparison.json`
and `deformation-audit.json`. Download of remaining notebook artifacts was
still running when this note was written; six checkpoint files and reports had
already passed hash checks.

## Vertex-sampled deformation experiment (2026-11-03; running)

Version 4 artifact download completed. To remove the interpolated sample LBS
approximation, `tools/build_skin_vertex_data.py` samples 1,024 original mesh
vertices per asset using area-weighted face sampling and a random corner.
Across the sampled source animation frames, maximum discrepancy between LBS
and directly imported posed vertices is 6.6e-9 for RiggedSimple, 7.4e-8 for
CesiumMan, and 2.0e-8 for Fox. Duplicate vertices are possible; this is an
area-weighted vertex sample, not a uniform surface sample or broader corpus.

Submitted paired baseline vs deformation-loss training on these new samples,
three folds, seed 31, as Kaggle cross-validation notebook version 5. Source
bundle SHA256 `ee3e928a740c1ba57bc0e7bbe7e46ccfa727b6a40dfa6d535d8fd12a64c0ca6a`;
latest status RUNNING. Version 5 changes the sample distribution, so compare
its paired runs within version 5 and do not attribute differences versus v4
solely to the loss. No result or training completion is asserted yet.

Version 5 passed its 10 bundled checks and is running. The 1,024 sampled
vertices contain 117 unique vertices for RiggedSimple (160 source vertices),
680 for CesiumMan (3,273), and 666 for Fox (1,728). Area-weighted sampling
with replacement intentionally repeats vertices; this limits training
coverage and must be considered when interpreting the experiment. Full counts
are in `artifacts/skin-vertex-pilot/sample-coverage.json`.

Version 5 completed and all 10 bundled checks passed. All six checkpoints match
their training-report SHA256 and the submitted source bundle hash. Within this
vertex-sampled run, held-out original-vertex mean LBS error baseline -> penalty:
RiggedSimple 0.040447 -> 0.031682 (21.7% better), CesiumMan 0.030595 ->
0.033010 (7.9% worse), Fox 0.020996 -> 0.018478 (12.0% better). Results are in
`artifacts/kaggle-skin-vertex-ablation-v5/paired-deformation-comparison.json`.
Selection used validation KL, and the experiment has only three assets and one
seed; it does not show a consistent improvement and no checkpoint is promoted.
Remaining notebook artifacts were still downloading at the time of this note;
all six checkpoints and reports were already verified.

## Additional RiggedFigure evaluation (2026-11-03)

Downloaded RiggedFigure from Khronos glTF Sample Assets; the source README
credits Cesium, copyright 2017, Creative Commons Attribution 4.0. Its GLB
SHA256 is `d6be85417d3e256861ee733eea6916093a7af7c79c16366181fd8abcaeb38cf5`.
The 370-vertex, 19-joint asset imports with its single animation. Evaluated all
six version-5 checkpoints using its original weights and five pose samples;
none lists RiggedFigure among training assets. Paired baseline -> deformation
loss mean vertex errors are 0.00880 -> 0.01148 (CesiumMan-holdout model),
0.00245 -> 0.00291 (Fox-holdout model), and 0.01977 -> 0.00424
(RiggedSimple-holdout model). Results are in
`artifacts/kaggle-skin-vertex-ablation-v5/external-RiggedFigure-deformation-audit.json`.
This asset shares a creator and potentially related design with CesiumMan; no
claim of statistical independence or broad generalization is made. These are
three differently trained checkpoints evaluated on one asset and may not be
pooled as three independent assets. No checkpoint is promoted.

BrainStem also imports (34,159 vertices, 18 joints, one animation), but its
README references a separate Poser EULA. It is retained only as an inspected
candidate; review its license before using it in training or evaluation.

## Joint-position checkpoints on RiggedFigure (2026-11-03)

Evaluated the six version-3 joint-position ablation checkpoints on the same
RiggedFigure source and poses. Mean vertex error baseline -> penalty is
0.008101 -> 0.008767 for the CesiumMan-holdout training fold, 0.003328 ->
0.003560 for Fox-holdout, and 0.026933 -> 0.026005 for RiggedSimple-holdout.
These are three models on one external asset, with a potentially related
Cesium source; the result does not establish broad generalization. Added a
source SHA256 field to the evaluator and a check rejecting an external source
whose hash matches any known training asset (or whose training asset hash is
unavailable). Re-ran both external comparisons with these checks. This only
rules out byte-identical assets, not common authorship or derived content.
Reports are `artifacts/kaggle-skin-position-ablation-v3/external-RiggedFigure-deformation-audit.json`
and the equivalent file under `artifacts/kaggle-skin-vertex-ablation-v5/`.

## External CC0 animal evaluation (2026-11-03)

Downloaded Corgi, Bat and Shark from
`https://github.com/Ariescar/gobkit-free-assets/tree/main/animal`; the source
repository designates these assets CC0, and its license is retained at
`artifacts/rigged-candidates/GOBKIT-LICENSE.txt`. Individual GLB hashes and
import audits are retained with each candidate. Each imports as a single
16-joint rig with one animation. No external file has the same SHA256 as a
known training source. Five evenly spaced frames per asset were evaluated on
all six version-5 checkpoints, using the original bind weights as encoder
inputs. `external-animals-comparison.json` and per-asset audit files are in
`artifacts/kaggle-skin-vertex-ablation-v5/`.

The deformation-loss model's mean vertex error is lower in eight of nine
paired comparisons and essentially tied/slightly worse in Corgi/Fox fold
(0.009212 vs 0.009224). These animals are from the same pack and share rig
structure, so the nine model/asset pairs are not nine independent rig families.
Five samples over one baked animation per asset and source-weight encoder input
are limited evidence; this does not establish source-free generation or
reconstructed-export quality, and no checkpoint is promoted.

## Denser animation sampling for external animals (2026-11-03)

Added `--frames` to `tools/audit_skin_pose_checkpoints.py` and reran all six
version-5 checkpoints on Corgi, Bat and Shark at 25 evenly spaced times per
baked clip. The deformation-loss checkpoint now improves seven of nine paired
asset/model comparisons; Corgi/Fox fold and Shark/Fox fold are slightly worse.
Results and per-frame data are retained in
`artifacts/kaggle-skin-vertex-ablation-v5/external-animals-comparison-25frames.json`
and the three `external-*-deformation-audit-25frames.json` files. Mean errors
are generally larger than with five samples, showing that the sparse sample
underestimated average deformation error for this particular clip set. The
three models still come from one asset pack and share rig structure. This is
not independent corpus coverage, rendered quality, or source-free generation.

## Family-held-out deformation training (2026-11-03; running)

Prepared `skin-family-vertex-pilot`: RiggedSimple, CesiumMan and Fox are
training sources; all three Gobkit animals are validation sources, with source
hash split checks and no animal-pack asset in training. Exact sampled-vertex
LBS reference discrepancies are under 7.4e-8 for all six assets. The animals
are a related family and have already been examined, and validation KL is used
to select a checkpoint, so this is a family-held-out validation experiment,
not an untouched test. CC0 license and per-source hashes remain in
`artifacts/rigged-candidates/`.

Submitted paired baseline versus deformation-loss training as Kaggle
cross-validation kernel version 6; bundle SHA256
`4f50dd7905686677bd00e301783d221fb66ac170e20ded40678b687e6f88e587`,
manifest SHA256 `b7d0a7d221793a7aee7da156ace07a069fa7ff32fb2664348c8a884961e292c8`.
The bundle includes all six datasets, uses 1,200 steps per model and averages
validation KL over the three related animal assets. The dataset loader verified
three disjoint training and three validation assets locally; inspect the same
remote run before retrying and do not claim improvement before evaluation.

Version 6 completed and passed its 10 bundled checks. The 25-frame original
vertex evaluation verifies checkpoint SHA256 and the exact training/validation
asset split. Mean canonical vertex error baseline -> deformation loss:
Corgi 0.012455 -> 0.013738, Bat 0.018078 -> 0.019805, Shark 0.011406 ->
0.012104. The penalty worsens all three assets for this family-held-out,
single-seed experiment, so it is not promoted. Checkpoints were selected by
validation KL on these same three related assets, so this is validation evidence,
not an independent final test. Results are in
`artifacts/kaggle-skin-family-ablation-v6/family-comparison.json` and
`family-deformation-audit.json`; remaining notebook artifacts were still
being downloaded when recorded.

## Deformation-based checkpoint selection (2026-11-03; running)

For version 6, evaluated both family-held-out models on RiggedFigure at 25
frames: mean canonical vertex error was 0.004474 (baseline) vs 0.004253
(deformation loss). This asset was not used in that run's training or checkpoint
selection, but it was examined in earlier work and may be related to CesiumMan.
Results are in `artifacts/kaggle-skin-family-ablation-v6/external-RiggedFigure-25frames.json`.

Added validation deformation error to the pose trainer and an optional
`--select-metric deformation_error`, retaining KL selection as the default.
The score uses sampled bind points, source weights, and pose matrices; it is
not a rendered metric. The evaluator rejects deformation selection if matrices
are unavailable. A local smoke check executed the new metric. Submitted paired
family-held-out training as Kaggle cross-validation kernel version 7, with
both models selected by validation deformation instead of KL; source bundle
SHA256 `81c77ae757a4aa52220bc343a8d3cba96116b548afa638151fb78de671bc90c4`.
The same animal validation family guides selection; it is not an untouched test.
No performance conclusion is made until completion and independent evaluation.

Version 7 passed all 10 bundled checks and is running. Its validation deformation
metric uses joint embeddings encoded from each posed skeleton, matching existing
pose-training evaluation. The reconstructed export and the independent vertex
audit encode from the bind skeleton. Therefore checkpoint selection is a
pose-conditioned proxy rather than the same evaluation path as export; compare
with the bind-conditioned 25-frame evaluation before drawing conclusions.

Prepared `tools/build_skin_family_bind_data.py` and the separate
`artifacts/skin-family-bind-pilot/` dataset, adding normalized source bind
joint locations for all six rigs without modifying the version-7 inputs. This
will permit validation that encodes the bind skeleton before evaluating poses,
matching the export path. The new dataset has not yet been used for training.

Version 7 completed and passed all 10 bundled checks. Its two checkpoint hashes
and asset splits were verified. In the 25-frame bind-conditioned original-vertex
validation, baseline -> deformation loss mean error is Corgi 0.012455 ->
0.012247, Bat 0.018078 -> 0.017964, Shark 0.011406 -> 0.011482. The first
two are slightly lower and the third slightly higher; these are related
validation assets used for selection, and the differences are small. The
report is `artifacts/kaggle-skin-family-deformation-select-v7/family-comparison.json`.
Remaining notebook artifacts were still downloading at this update.

Implemented optional `bind_deformation_error` selection in the pose trainer,
using the saved bind joints to encode weights before applying each pose. A
local evaluation smoke check passed. This is a closer match to the export's
bind-conditioned decode than the version-7 metric, but still samples validation
vertices rather than reconstructed voxel fields. The bind-conditioned option
has not yet been trained or evaluated remotely.

## Bind-conditioned checkpoint selection (2026-11-03; running)

Version 7 artifact download completed. Checked the new bind-conditioned
validation metric for one Corgi pose with the version-7 baseline checkpoint:
Torch reports 0.0094673466 and independent NumPy LBS reports 0.0094673715,
a difference of 2.5e-8; this validates one calculation, not generalization.
Submitted Kaggle cross-validation kernel version 8 using the bind-joint dataset,
the same family split and paired loss strengths 0 and 1. The score selects
checkpoints by bind-conditioned deformation error; source bundle SHA256 is
`024e04ac54dc2b58d7f07cfc3a4a92325a65ea8506f764137cc6934441bf4dac`.
The job was confirmed RUNNING. Validation assets guide selection and are not
untouched tests; inspect this same remote job before retrying.

Version 8 completed and its 10 bundled checks passed. Both checkpoint hashes
match reports and the three-asset training/three-asset family-validation split
was verified. Bind-conditioned selection chose the same training steps as
version 7 (300 for baseline, 900 for deformation loss); their model state
arrays are bitwise equal despite differing checkpoint-file hashes. Consequently
25-frame bind-conditioned vertex results exactly match version 7: Corgi
0.012455 -> 0.012247, Bat 0.018078 -> 0.017964, Shark 0.011406 -> 0.011482.
This is a single-seed result on the same validation family and is not evidence
of improved generalization. The full comparison is under
`artifacts/kaggle-skin-family-bind-select-v8/`. Remaining notebook artifacts
were still downloading at this update.

## v14 nonmanifold edge classification (2026-11-03)

Added `tools/audit_nonmanifold_edges.py` to enumerate welded edge incidence,
original vertex-edge counts, duplicate position-welded triangle participation,
and incident normal relationships. The v14 standard output has 15 welded
nonmanifold edges: 12 with incidence four and three with incidence eight; it
also contains 12 geometrically duplicated position-welded faces. Every flagged
edge touches at least one duplicate face. The original CesiumMan has zero
position-welded duplicate faces and zero nonmanifold edges by the same audit.
A hypothetical in-memory deduplication reduces nonmanifold edges from 15 to 9
but creates nine boundary edges; this is not a repair and was not written to
an output GLB. Exact positional welding can merge disconnected touching
surfaces, and the UV/skin attributes of duplicates need consideration. Full
evidence is in `artifacts/kaggle-textured-rig-export-v14/nonmanifold-analysis.json`
and `source-nonmanifold-analysis.json`. Production topology remains unresolved.

## Duplicate-sheet topology simulation (2026-11-03)

`tools/audit_duplicate_faces.py` compares exact-position duplicate triangles
in the v14 GLB. It finds 12 duplicate pairs: nine opposite-facing and three
same-facing. Matched corner weights are identical in all pairs; two pairs have
different UVs. Some pairs share face IDs, so counts of removed faces are not
simply twice the pair count. In an in-memory experiment, removing both faces
of opposite-winding pairs and the second face of same-winding pairs removes 18
unique faces. Exact-welded topology changes from 15 to three nonmanifold edges
with zero boundary edges, and a 512-sample surface audit gives F-score 1.0 and
mean distance 0.00005813 (before: 0.00005833). Detailed simulated face IDs and
metrics are in `duplicate-removal-simulation.json`; corner UV and normal
comparisons are in `duplicate-face-analysis.json`. This is not yet an export
fix: coincident surfaces can be intentional, two pairs differ in UVs, the
remaining three edges are unresolved, and no texture or animation appearance
has been evaluated. No GLB was modified.

## Candidate GLB after duplicate-sheet removal (2026-11-03)

Created a separate `duplicate-removal-candidate.glb` by rewriting only the index
accessor of the hash-verified v14 source, using the 18 simulated face IDs;
UVs, materials, vertices, skin attributes and the original GLB remain intact.
The candidate has 9,294 faces and SHA256
`5f982e4fb6281f3eca805c88887ed59404f7c0a3d50512545878e2510818cc3a`.
Khronos glTF Validator reports zero errors, warnings and infos. A 4,096-point,
seed-zero linear-RGB attribute audit yields RMSE 0.1264 / 0.0982 / 0.1028,
versus 0.1273 / 0.0989 / 0.1043 for the original v14 (different sampled
triangles); this is not a rendered texture check. The animated posed audit
was started and remains running at this update. The candidate still has three
nonmanifold edges, and intentional coincident geometry and UV appearance remain
unresolved, so this experimental removal is not part of the export pipeline.

The duplicate-removal candidate's five-pose surface audit completed: mean
posed distance is 0.016595 and F-score 0.6402, compared with original v14's
0.016589 and 0.6396; differences include resampling after face removal. The
animated round-trip was valid at all sampled times, max vertex delta 3.39e-6.
The three remaining welded nonmanifold edges each have four incident faces,
four separate pre-weld UV edges, and mixed incident normals; see
`duplicate-candidate-nonmanifold.json`. Rendered source, v14 and candidate at
five poses via `tools/render_duplicate_candidate.py`; the first frontal frame
shows no obvious gross texture change between v14 and candidate, while both
retain visibly folded legs. Images are in
`artifacts/posed-deformation/render-duplicate-removal/`. This is not an HDRI,
multi-view or pixel-matched material verification. The candidate remains an
experimental separate GLB; no generic removal or production change is justified
by one asset, and three nonmanifold edges remain.

## Remaining three edges: local face-deletion search (2026-11-03)

The candidate's three incidence-four edges are short (0.00382, 0.00452,
0.00382 canonical units); two are approximately mirrored. They involve 12
incident triangles. Exhaustively searched all 4,096 subsets of those 12 faces,
requiring each touched exact-welded edge to retain incidence zero or two.
No subset satisfies that local condition. The search is restricted to deleting
incident faces and does not exclude a broader patch reconstruction, vertex
splitting, or intentional contact between separately modeled sheets. Evidence
is in `artifacts/kaggle-textured-rig-export-v14/remaining-edge-local-search.json`.
No further candidate was modified; topology remains unresolved.

## One-ring topology patch prototype (2026-11-03)

`tools/analyze_topology_patches.py` found that deleting the one-vertex-ring
around the three remaining defects removes 66 faces and yields three simple
boundary loops of 9, 9 and 10 edges. `tools/simulate_topology_patch.py` closes
those loops using 28 triangle fans in memory. Exact-weld incidence then shows
zero boundary and zero nonmanifold edges; a 512-point global surface sample
returns F-score 1.0, mean 0.0000581. However, targeted samples at the new patch
vertices and centroids show mean distance 0.00139 and maximum 0.00642 to the
source, substantially larger than the global mean. The fan prototype has no
UVs, skin weights, normals, self-intersection check or rendered validation,
and no GLB was written. This is evidence of a possible local repair strategy,
not a production-quality or verified watertight export. See
`fan-patch-simulation.json` and `nonmanifold-patch-neighborhoods.json`.

## Projected fan-center prototypes (2026-11-03)

Compared the unprojected one-ring fan to projecting each new center onto a
reference surface. With 112 targeted patch vertices/centroids compared against
the original CesiumMan, unprojected mean/max distance is 0.001392 / 0.006420;
projecting centers onto the original source gives 0.000250 / 0.003499 (oracle
requiring source geometry). Projecting centers onto the available pre-patch
candidate instead gives 0.000500 / 0.003346, with zero exact-weld boundary or
nonmanifold edges and 512-sample F-score 1.0. Reports are
`fan-projected-oracle-simulation.json` and `fan-projected-candidate-simulation.json`.
These probes do not bound continuous triangle deviation or detect intersections;
none of the variants generates UVs, skin weights or a GLB. They are local
single-asset experiments, not a production repair or quality guarantee.

## Projected patch intersection and denser distance checks (2026-11-03)

Saved the in-memory projected fan geometry as
`fan-projected-candidate-geometry.npz`. Blender's BVH overlap query found zero
patch/mesh triangle overlap pairs after excluding pairs sharing a welded
vertex; this is not an exhaustive robust self-intersection guarantee, especially
for adjacent or coplanar triangles. A regular 20-subdivision barycentric grid
on all 28 patch triangles gives 6,468 source-distance probes: mean 0.000809,
p95 0.002803, max 0.004865 canonical units. These directional samples are more
dense than the prior 112 probes but do not bound continuous maximum or the
source-to-patch distance. Reports are `fan-patch-bvh-overlap.json` and
`fan-patch-dense-distance.json`. Geometry remains a prototype without UVs,
skin weights, a patched GLB or production integration.

## Packaged projected fan candidate (2026-11-03; evaluation running)

Added `tools/build_fan_patch_candidate.py`, producing a separate experimental
`fan-patch-candidate.glb` with triangle-corner UVs and interpolated top-16 skin
weights sampled from a nearest pre-patch triangle. Boundary copies reuse exact
position and skin attributes; repeated fan centers reuse skin attributes.
The first packaging attempt exposed 72 cracks from floating-point differences
at duplicated corners; corrected corner/skin reuse reduced exact-welded boundary
and nonmanifold edge counts to zero in the actual reloaded GLB. The corrected
GLB SHA256 is `16b15f7debd4f75cf933801d78635555e9c2ff0ca1a90784b327b3c9f194e5bf`.
Khronos validator reports zero errors, zero warnings, and 12 informational
unused-accessor messages because the experimental rewriter retains the old
buffer/accessors. Posed animation and 4,096-point material audits were started
and remain running. The UV projection can smear across seams, and no rendered
or animated patch verification is complete. This one-asset candidate is not
integrated into the production export pipeline and does not establish general
watertightness or self-intersection freedom.

The corrected candidate's material audit completed: 4,096-point seed-zero
linear RGB RMSE is 0.12454 / 0.09668 / 0.10143, versus original v14
0.12727 / 0.09891 / 0.10433 with independently sampled triangles. This is a
whole-surface attribute metric and may miss tiny patched regions; it does not
establish rendered patch quality. The posed audit remains pending.

## Packaged fan patch: targeted material and animation evaluation (2026-11-03)

The corrected GLB's five-pose audit completed: mean posed distance 0.016485,
F-score 0.6451, with valid animated round-trips at all five times and maximum
vertex delta 3.39e-6. V14's independently sampled result is 0.016589 / 0.6396;
this is not proof of improvement because the faces and surface samples differ.
A targeted 4,096-point area sample confined to the 28 patch faces compares
linear RGB against the source at the same spatial points. Original pre-patch
candidate values at the nearest pre-patch surface have RMSE 0.0794 / 0.0651 /
0.0731; patched values are 0.1074 / 0.0858 / 0.0952. Thus the patch's UV
projection worsens local material fidelity on this asset despite the global
sample being similar. See `fan-patch-only-material.json` (nearest-surface
attribute comparison, not rendering).

Five frontal Blender renders are in `artifacts/posed-deformation/render-fan-patch/`;
the first shows no obvious gross difference at this scale between original v14
and patched version, while both retain the folded legs. Small patches and UV
artifacts can be invisible at this resolution. This experimental single-asset
GLB remains outside the export pipeline; local material loss, topology/visual
quality and robust self-intersection checks require further work.

## Patch UV projection alternatives (2026-11-03)

Compared UV mapping choices at the same 4,096 area-weighted patch samples.
Using independently nearest faces for each corner crosses atlas charts and
increases linear RGB RMSE from 0.1074 / 0.0858 / 0.0952 to 0.4929 / 0.3798 /
0.4069. A single nearest face per patch triangle with *unclamped planar UV
barycentric extrapolation* instead of clamping gives 0.08224 / 0.06716 /
0.07617, close to the pre-patch nearest-surface comparator 0.07942 / 0.06507 /
0.07308. This comparator is sampled at patch points against source colors;
it is not rendered quality. The extrapolated experimental GLB has zero
exact-weld boundary and nonmanifold edges; the validator reports zero errors,
zero warnings and 12 unused-accessor infos. Its position, normals, joints,
weights and index accessors are bitwise equal to the already posed-audited fan
candidate; only UV values change. UVs happen to lie in [0,1] for this asset.
Reports and candidates are in `artifacts/kaggle-textured-rig-export-v14/`.
Extrapolation can cross chart boundaries on other meshes; one asset and this
attribute metric do not validate a general export repair or rendered quality.

## Patch face material and animated topology inspection (2026-11-03)

Added per-face metrics to `fan-patch-extrapolated-only-material.json` using the
same 4,096 patch samples. All 28 patch faces received at least ten samples; ten
have higher RGB RMSE than the nearest pre-patch comparator. The worst per-face
candidate RMSE is 0.23765 against 0.22708 baseline on 197 samples. This is an
attribute comparison against the source, not proof of visual quality or a
statistical estimate for every face.

`tools/audit_fan_patch_motion_topology.py` reloads the animated fan candidate
at 25 evenly spaced times. Each pose has zero exact-weld boundary edges, zero
nonmanifold edges, and no triangle area <= 1e-12; minimum observed triangle
area is 2.20e-7 canonical square units. Detailed results and candidate SHA256
are in `fan-patch-posed/motion-topology-25frames.json`. The extrapolated-UV
candidate has identical position, joint, weight and index accessors, so these
geometry/skin findings apply to its matching animation if retargeted identically.
This checks discrete samples, not continuous motion, self-intersections,
quality of leg articulation, or general behavior on other assets.

## Guarded reusable local fan geometry (2026-11-03)

Added `t3d.topology_repair.repair_local_fans`, a geometry-only experimental
function that exact-welds, removes the investigated duplicate-sheet pattern,
forms one-ring neighborhoods, projects fan centers to the input surface, and
rejects open input, ambiguous winding, large removals, non-simple loops,
remaining nonmanifold edges, and exactly degenerate output faces. The audit
returns retained/deleted face IDs and counts near-degenerate faces; this is not
integrated into export and does not transfer materials, UVs or rig attributes.
Exact position welding can merge intentionally touching sheets.

On v14 it removes 18 duplicate and 66 local faces, adds 28 fan faces, and
produces 9,256 faces with zero welded boundary/nonmanifold edges; coordinates
match the saved projected prototype within 1.4e-17. The output has no faces
with double area <= 1e-12 (minimum 2.63e-7). A closed tetrahedron was retained
and an open tetrahedron was rejected. These narrow checks do not prove the
algorithm preserves intentional coincident geometry, avoids intersections,
works on other assets, or preserves UV/material/animation appearance.

## Guarded repair across retained export variants (2026-11-03)

Ran `tools/audit_topology_repair_variants.py` on standard and strict outputs
from v10–v14, all derived from the same CesiumMan source. It accepted v10
strict, v13 standard and v14 standard, producing zero exact-weld boundary and
nonmanifold edges in each; 512-sample F-scores were 0.814, 1.0 and 1.0
respectively. It rejected the other seven: five because the input was open and
two because the neighborhood exceeded the default 3% face-removal budget.
These are useful fail-closed outcomes, not successful repairs for those seven
variants. The accepted low-resolution strict result is still geometrically poor,
and all results are geometry-only; exact edge incidence does not prove
manifold vertices, absence of self-intersection, UV/skin fidelity, or broad
asset generalization. Hashes, metrics and rejection reasons are saved in
`artifacts/topology-repair-variants.json`.

## Geometry repair controls on other sources (2026-11-04)

`tools/audit_topology_repair_controls.py` ran the geometry-only repair on seven
additional existing source assets. RiggedFigure was accepted unchanged (256
faces, no duplicate/region removal or added faces); Lantern, WaterBottle,
Avocado, Corgi, Bat and Shark were rejected because their exact-welded input
meshes have boundary edges. Results and hashes are saved in
`artifacts/topology-repair-controls.json`. These are original assets rather
than reconstructed outputs. Rejection is the intended current guard, not a
repair of common open meshes; accepting one clean rig does not establish
attribute preservation, intersection freedom or generalization.

## Optional preservation of existing boundaries (2026-11-04)

`repair_local_fans(..., allow_open=True)` now preserves the exact set of
original boundary edges and refuses any repair touching their vertices; the
default still rejects open meshes. The function validates that all new patch
loops are separate and the output has the same original boundaries, no
nonmanifold edges, and no exactly degenerate triangles. On seven source controls,
the opt-in accepts all unchanged, preserving their original 0–264 boundary
edges; details are in `artifacts/topology-repair-open-controls.json`. A v14
mesh with an isolated open triangle was repaired while preserving exactly its
three boundary edges; a triangle attached to a repair neighborhood caused the
expected fail-closed rejection. This remains geometry-only, uses exact position
welding, and is not integrated into GLB export. These controls do not prove
attribute preservation, source fidelity or general-purpose repair quality.

## Boundary-preserving repair on export variants (2026-11-04)

Re-ran all ten v10–v14 CesiumMan standard/strict exports with `allow_open=True`.
The same three as before were accepted; five open variants were rejected
because their defect neighborhoods touch existing boundaries, and two strict
variants exceeded the 3% face-removal guard. See
`artifacts/topology-repair-open-variants.json`. Preserving boundaries does not
make it safe to alter those neighborhoods.

As a sensitivity probe only, raised the removal limit to 15% for v11/v12 strict.
Both were accepted, each removing 277 faces (about 3.0%, just above the default
limit) and yielding zero exact-weld boundary and nonmanifold edges. Their
512-sample source F-scores before -> after are 0.963 -> 0.973 for v11 and
0.976 -> 0.976 for v12; these sparse samples cannot establish local patch
quality. Results are in `artifacts/topology-repair-strict-budget-probe.json`.
The default guard is unchanged; UV, skin, local material, self-intersection,
and broad asset validation are still required.

## Coupled occupancy flow foundation (2026-11-04; CUDA probe running)

Added `t3d.coupled_occupancy.CoupledOccupancyFlow`: separate shape and skeleton
occupancy DiTs exchange features through zero-initialized linear adapters at
all blocks. It validates binary contained skeleton targets, supports prompt
conditioning/dropout, and masks generated skeleton occupancy to generated shape
support. `ThreeStageFlow(separate_skeleton=True)` now exposes the paired loss
and returns skeleton coordinates alongside shape/material/optional rig latents;
the existing default single occupancy branch remains available. This is coarse
voxel support containment, not proof that predicted bones fit within a decoded
surface; prompt encoders and trained generator weights remain absent.

Local CPU smoke checks verified four loss terms/backward, generated support
containment, and rejection of skeleton coordinates outside shape; the two
existing generation tests passed. Submitted a synthetic 8-step CUDA integration
probe as Kaggle cross-validation kernel version 9, bundle SHA256
`47d6b4a8a36ed01615ed2cc82fd1fe7e2e07a0d618e574360ed5dc329a651dda`.
This tests the implementation path, not generation quality or full plan-scale
training. Inspect the existing job before retrying.

## Coupled occupancy CUDA v9 and containment correction (2026-11-04)

Kaggle version 9 completed on a Tesla T4: eight synthetic optimizer steps ran,
the cross-branch adapter received nonzero gradients, and sampled skeleton
support was restricted to shape support. Report and logs are retained in
`artifacts/kaggle-coupled-occupancy-v9/`; these results verify integration only.

Review identified that requiring skeleton voxels to be *surface* voxels excludes
valid interior bones. Corrected the new branch to allow skeleton coordinates
anywhere in the axis-aligned bounding box of occupied shape coordinates;
training rejects targets outside it, and sampling masks outside it. A local
smoke case confirms an interior non-surface voxel is accepted and an external
voxel rejected; the existing two generation tests pass. This AABB is only a
coarse bound: it does not enforce inside-mesh containment or represent a solid
occupancy, so the plan's actual containment requirement is still unmet. The
old v9 subset result describes the superseded code, not the corrected behavior.

Submitted corrected synthetic CUDA probe as the same Kaggle kernel version 10,
source bundle SHA256 `2ab7b96ff3a47ee169ce465de963c6ee27a3e18b6bff2e879c5a9e43d974743d`.
It writes the bundle hash into its report; inspect this job before resubmission.

## Discrete enclosed-volume option (2026-11-04)

Kaggle version 10 completed on Tesla T4; its hash-verified report records
nonzero adapter gradients, 40 shape voxels, 12 skeleton voxels and AABB
containment on the synthetic probe. Artifacts were downloaded to
`artifacts/kaggle-coupled-occupancy-v10/`.

Added `skeleton_containment='enclosed'` as an opt-in coupled-occupancy mode.
It fills the complement of a 6-connected exterior flood on a complete coarse
grid and permits skeleton voxels on the surface or in enclosed cells. It
rejects incomplete grids or a flood exceeding its iteration limit. A local
closed-cube shell test retained the center; removing one shell cell caused
exterior leakage and rejected the center. Both existing generation tests pass.
This is a discrete approximation: open/thin shells can leak, and it does not
prove bones or segments lie within the final decoded continuous surface.
The default remains AABB. Updated `docs/generation.md` accordingly.

Submitted Kaggle synthetic probe version 11 including CUDA checks for closed
and open shells, bundle SHA256
`2d2fd727d91a51edba18731ab55375b4d3261534dc305f295abb98582f8cde1f`.
Await that same run before retrying. Neither the T4 probes nor these fixtures
provide trained generation or benchmark quality evidence.

## CUDA enclosed-fill and real-rig voxel audit (2026-11-04)

Kaggle version 11 completed on Tesla T4; the saved report matches source bundle
`2d2fd727d91a51edba18731ab55375b4d3261534dc305f295abb98582f8cde1f`,
records nonzero adapter gradients, successful closed-shell fill and detection
of an opened-shell leak. Reports and logs are in
`artifacts/kaggle-coupled-occupancy-v11/`; this remains a synthetic integration
probe, not a trained generation benchmark.

`tools/audit_real_voxel_containment.py` downsamples the real v14 CesiumMan
128³ surface field to 16³, 32³ and 64³ and rasterizes sampled bind bone
segments. At 64³ only 12 of 152 bone voxels coincide with surface voxels;
all 152 fall within both the AABB and 6-connected filled shell. At 32³,
42 of 72 are surface and all 72 pass filled-shell; at 16³, 37 of 39 are
surface and all 39 pass filled-shell. Thus surface-subset masking would reject
most bone support at higher resolution on this example. See
`artifacts/real-voxel-containment-audit.json` for hashes and full counts. This
uses a downsampled source surface and sampled bind segments, not a generated
mesh, continuous inside test, representative corpus, or proof of bone validity.

## Coupled occupancy real-asset overfit probe (2026-11-04; running)

`tools/build_coupled_real_pilot.py` created a hash-tracked 16³ pilot from the
v14 CesiumMan surface field and rasterized bind skeleton segments: 226 surface
and 39 skeleton voxels. This is one source asset and provides no unseen data
or image/text conditioning. `tools/train_coupled_real_pilot.py` trains the
coupled occupancy backbone on those real targets for 200 CUDA steps, records
fixed-noise pre/post shape and skeleton IoU and checkpoint hash, and uses the
discrete enclosed-shell constraint. This is an overfit/integration probe, not
research-scale training or an evaluation of generalization.

Submitted Kaggle cross-validation kernel version 12; source bundle SHA256
`55b24662d7dc87bf13e84265827417ebf88a7bc6e4667665af38dcb67f6759df`,
pilot NPZ SHA256 `7c5321ee9080f447fa31928e47a5b7f1a8560fdd2917de0225de1ef521b2e387`.
Inspect this same run and actual reported IoU before asserting that it learned.

## Real coupled occupancy v12 and position ablation (2026-11-04)

Kaggle version 12 completed on Tesla T4; source and checkpoint hashes match the
archived report under `artifacts/kaggle-coupled-real-v12/`. Fixed-noise single
training-asset shape IoU changed 0.0459 -> 0.0331, while skeleton IoU changed
0.0145 -> 0 after 200 steps. This is a failed overfit result, not evidence of
learning or generalization. The shape flow receives coordinates through
relative RoPE only, which may limit learning a canonical fixed-grid shape; this
is a hypothesis, not a proven root cause.

Added an optional absolute-coordinate embedding to `CoupledOccupancyFlow` and
submitted Kaggle version 13 to compare baseline and the new option on the same
16³ source and nominal seed, 200 steps each; bundle SHA256
`d7f2cee55d329b2736f05351cc7223658e85c56088f6905ea200fad7db4840ca`.
Important limitation discovered after submission: model initialization consumes
extra random numbers in the absolute-coordinate model, so its saved fixed-noise
evaluation inputs differ from the baseline. The run can provide preliminary
per-model observations but is not a paired fixed-noise comparison. Correct
this with an independent evaluation generator in a subsequent run, after
version 13 reaches a terminal state; do not overwrite or restart the live job.

The local pilot now draws evaluation noise from a dedicated generator seeded
4301, independent of model initialization; this change is not in running v13.
Version 13 was confirmed RUNNING after the change. Its outputs must remain
labeled as unmatched-noise observations.

Prepared the next paired experiment with five explicit evaluation-noise seeds
(4301–4305) shared across variants and reset the training-noise/timestep RNG
after model initialization. This local code is not part of the currently
running v13 bundle. Exposed the new embedding as
`ThreeStageFlow(..., separate_skeleton=True, absolute_occupancy_coords=True)`;
the default remains unchanged, and a local finite-loss/backward smoke check
passed. This is a reference implementation choice, not evidence of fidelity.

Version 13 completed; downloaded both checkpoint/report pairs and verified
checkpoint hashes and the submitted bundle hash. Baseline vs absolute-coordinate
post-training shape IoU is 0.03315 vs 0.04010; skeleton IoU is 0 vs 0. Both
initial report summaries are numerically identical, but the run did not record
or independently control evaluation noise tensors, so this does not prove a
paired-noise comparison. The result is one noise sample and one training asset;
neither model reconstructs the example well. Artifacts are in
`artifacts/kaggle-coupled-real-position-v13/`.

Submitted version 14 with independent seeded evaluation generators for five
matching noise seeds and reset training RNG after initialization. Bundle SHA256
`5a6761bcdeb8cd7d07491e49b538ea52db728eb868d8f5509c5cafff20ec387c`.
The models still differ in initialization and architecture, and this remains
one training seed/asset with no held-out set. Inspect the running job before
retrying; no improvement is asserted yet.

While v14 trains, compared `closed_shell_solid_mask` against an independent
six-neighbor queue-based BFS on 20 seeded 7³ random voxel grids (two surface
densities); all masks matched exactly. This validates flood-fill behavior for
those small discrete cases, not the continuous or real-mesh containment claim.

Version 14 paired-noise probe completed on T4. Both checkpoint hashes and
bundle hash `5a6761bcdeb8cd7d07491e49b538ea52db728eb868d8f5509c5cafff20ec387c`
match their reports; the same five evaluation seeds were used. Mean shape IoU
before -> after is 0.04957 -> 0.05297 for baseline and 0.04957 -> 0.02323 for
absolute coordinates; mean skeleton IoU before -> after is 0.00922 -> 0 for
both. Absolute coordinates did not improve this single-asset probe, and neither
model generated matching skeleton voxels on these seeds. This is one training
seed, one training asset, and no held-out generalization result. Remaining
version-14 notebook artifacts were still downloading when recorded.

Added `t3d.conditioning.PromptConditioner` with frozen DINOv2 image and CLIP
text backbone interfaces and trainable common-width token projections. A local
mock-backbone check verifies output shapes, attention-mask trimming, frozen
backbones and projection gradients. This is not verification of pretrained
models, semantic alignment, or trained image/text generation. Prepared a pinned
DINOv2-small/CLIP T4 probe (bundle SHA256
`eadebd2b0dd343d15921e31a5aa27feff734067e954a30585d42d92a85a70ba4`), but
have not submitted it while the version-14 download is still active.

Version-14 artifacts finished downloading. Submitted the pinned pretrained
prompt-conditioning integration probe as Kaggle kernel version 15; it tests
one rendered image and one prompt through DINOv2-small, CLIP text and the
untrained flow, with frozen backbones and trainable projections. This is an
integration probe only. Inspect this same job before any retry; external model
availability/download and actual execution remain unverified.

## Pretrained image/text conditioner integration on T4 (2026-11-04)

Kaggle version 15 completed. The log records the pinned `facebook/dinov2-small`
revision `ed25f3a31f01632728cabb09d1542f84ab7b0056` and
`openai/clip-vit-base-patch32` revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
On one cropped CesiumMan render and a text prompt, the frozen backbones and
trainable projections produced 257x32 image tokens and 10x32 text tokens;
projection gradients were present, backbone gradients absent, and both token
sets passed through the untrained coupled flow with finite losses on a T4.
This does not establish image/text alignment, differentiable training through
the frozen encoders, trained projections, or generated 3D quality. The run
used Transformers 4.x and pip reported unrelated preinstalled-package version
conflicts; downstream packages were not checked. Full artifact download to
`artifacts/kaggle-prompt-conditioning-v15/` was still running at this update.
Updated `docs/generation.md` to describe the optional conditioner and limits.

## Text-conditioned multi-asset occupancy pilot (2026-11-04; running)

Version 15 artifacts finished downloading and the report's source bundle hash
was verified. Built `artifacts/prompt-occupancy-pilot/` from three CC0 Gobkit
animals (Corgi, Bat, Shark) as training assets and the CC-BY Cesium RiggedFigure
as a validation asset, with hand-written class prompts, source/data hashes and
license provenance in the manifest. At 16³, Corgi has one bone voxel outside the
filled-shell mask, so the training probe uses AABB containment rather than
silently discarding a target. These four meshes are a tiny, curated sample;
RiggedFigure and these animals were examined in prior work, and the animals
share a source family. This is not an untouched benchmark or aligned corpus.

Added `tools/train_prompt_occupancy_pilot.py`: frozen CLIP tokens are cached,
the text projection and coupled flow train for 300 steps, and correct vs swapped
prompts are sampled with matching per-asset noise on all four assets. Reports
include IoUs and mask differences, not a claim of semantic generalization.
Submitted Kaggle cross-validation kernel version 16, bundle SHA256
`e570250cc9815a2cc27cb92973e1fbb3bc97523c8b843ea26b1f39d0a0f96b24`.
The initial run must be checked for actual completion and effects before any
claim; pretrained download and package conflicts remain environment-dependent.

Version 16 has loaded the pinned encoders and reached training step 50 with
finite losses. Independently checked all four pilot data-file SHA256 values,
source SHA256 uniqueness, distinct prompt labels and train/validation labels;
all matched the manifest. Source hash uniqueness only rules out byte-identical
files, not related source content. No generation or prompt-effect result is
available yet; the same Kaggle job remains running.

Version 16 completed its 300-step text-conditioned run and printed prompt-swap
results; artifact download to `artifacts/kaggle-prompt-occupancy-v16/` is still
active, so checkpoint/report hashes and full results remain to be checked. The
logs show the validation figure's swapped prompt produced zero skeleton overlap;
full correct-vs-swapped metrics should be read from the verified report rather
than the truncated log. No semantic quality claim is made.

Added an optional `foreground_weight` to coupled occupancy flow losses, with
normalized weighted velocity MSE. Weight 1 preserves the previous unweighted
objective; greater weights change the flow training objective and are an
experimental response to sparse occupancy, not a proven correction. A local
arithmetic/gradient check and both existing generation tests passed. It has not
yet been trained on T4 or enabled in the running text experiment.

The complete v16 download finished; checkpoint SHA256 and source bundle hash
match the report. Fixed-noise prompt swaps changed 201–326 shape voxels per
asset, demonstrating prompt sensitivity in this probe. However, shape IoU for
correct vs swapped prompt was Corgi 0.0815 vs 0.0070, Bat 0.0245 vs 0.1084,
Shark 0.0181 vs 0.1183, and validation RiggedFigure 0.0277 vs 0.0694.
Skeleton IoU remained 0–0.0154. Thus swapped prompts outperform correct prompts
on three of four cases, and prompt sensitivity does not establish semantic
alignment or adequate reconstruction. These are one seed per asset and a tiny,
previously inspected dataset. Full evidence is
`artifacts/kaggle-prompt-occupancy-v16/prompt-occupancy-report.json`.

## Raw-vs-masked skeleton diagnosis prepared (2026-11-04)

Version 17 (foreground-weighted single-asset comparison) was submitted with
bundle SHA256 `0dc7aa99adca2e3d73b1ae4ef0bafba8f30996384ce52f1749f7954f612053c9`;
latest Kaggle status RUNNING. Independently added optional `return_values=True`
to coupled sampling and a local smoke check confirmed the returned raw fields
match thresholded outputs. Prepared `tools/audit_coupled_masking.py` and a T4
notebook builder to evaluate the archived v14 baseline/absolute checkpoints on
five matched noise seeds. It will record raw skeleton IoU, masked skeleton IoU,
removed predictions, and target skeleton voxels excluded by the generated
filled shell. This will distinguish prediction failure from masking loss on this
single training asset; it does not establish generalization. The audit notebook
has not been submitted while v17 is live; no diagnosis result is asserted yet.

## Foreground-weighted pilot and masking audit (2026-11-04)

Kaggle v17 completed on T4; saved checkpoints and source hashes were verified in
`artifacts/kaggle-coupled-real-foreground-v17/`. With five matched evaluation
noises on the single CesiumMan training asset, foreground weight 1 yielded mean
shape/skeleton IoU 0.0530/0.0000 after 200 steps, while weight 8 yielded
0.1573/0.0368. Both began at 0.0496/0.0092. This is one training seed and one
asset, so it demonstrates an overfit probe improvement rather than evidence
of generalization. Existing CPU generation unittests passed (2 tests).

The v18 masking audit now includes v14 baseline/absolute and v17 baseline/weighted
checkpoints, all hash checked. A first notebook push was rejected while embedding
all checkpoints; instead a private Kaggle dataset stores the unmodified saved
checkpoints (archive SHA256
`92e147a6b999f57bba6d1f88747b96dc13259d4036bf345e115ebea0bf51d9d8`).
The revised GPU notebook was submitted successfully with source SHA256
`74c9776a87d813589369c26c3cd416ea608d82039a57e09db9d4be8c55c2eabc`.
The v18 result has not yet been checked.

Kaggle v18 failed before inference: Kaggle extracted the uploaded zip into
individual dataset files rather than mounting `checkpoints.zip`. Dataset listing
confirmed all eight files. Updated the notebook builder to copy the extracted
`artifacts` tree; individual model files are still checked against their report
hashes. The same audit was resubmitted as v19; no v18 metrics were produced.

Kaggle v19 completed and the downloaded masking report's source hash matches
`74c9776a87d813589369c26c3cd416ea608d82039a57e09db9d4be8c55c2eabc`.
Across five fixed noises on the one training asset, the unweighted model's
raw/masked skeleton IoU averaged 0.0073/0.0000 and 35.4 of 39 target voxels
fell outside the predicted filled shell. The weighted model averaged
0.0348/0.0368 and excluded 5.0 target voxels, but predicted 502 raw skeleton
voxels on average for a 39-voxel target. Thus the baseline has both weak raw
prediction and restrictive masking; weighting overpredicts heavily, and none
of this establishes held-out performance. The report is
`artifacts/kaggle-coupled-masking-v19/coupled-masking-audit.json`.

Prepared and submitted v20, a paired text-conditioned multi-asset comparison
of foreground weights 1 and 8 with three matched noise seeds per asset and
correct versus swapped prompts. It retains the prior three related training
assets and previously inspected validation asset, so it is still a limited
pilot, not an untouched generalization benchmark. Both runs have the same
training seed and 300-step budget; source SHA256 is
`6899afb0e379608407afa6e7550bb19a3da12976fab01641c6856be3639f4a3c`.
No v20 metrics are asserted yet.

## Source identity split guard (2026-11-04)

Extended the prepared `.t3d` dataset manifest with optional original input
`source_sha256` and curator-supplied `source_group`. Either identity crossing
train/validation/test is rejected, while derived variants in the same split
remain valid. Two dataset unittests pass, including cross-split source and family
checks; legacy manifests remain supported. This guard cannot discover missing
or incorrect identities or semantic duplicates. The v20 Kaggle job is running.
The pilot dataset builder now carries source hashes from its import reports
into newly generated manifests; existing manifests were not regenerated, which
preserves their fingerprints for prior checkpoints and experiments.

## Multi-view curation rendering prototype (2026-11-04)

Added `tools/render_curation_views.py` and `docs/curation_render.md` to render
GLB color, depth and normal passes for requested camera azimuths and fields of
view, with input/output hashes and camera metadata. A Blender 5.2.2 CPU probe
rendered RiggedFigure from two azimuths at 32x32; files are in
`artifacts/curation-render-pilot/`. OpenImageIO inspection found finite foreground
depths and signed, nonconstant normal channels. This is a render integration
probe, not validated VLM filtering, aesthetic quality, a production curation
pipeline, or pose canonicalization. Blender 4 compatibility is untested.

## Text-conditioned foreground comparison (2026-11-04)

Kaggle v20 completed and both checkpoints and source bundle hashes were
verified in `artifacts/kaggle-prompt-foreground-v20/`. Three matched noise
seeds per asset and one shared training seed show correct-prompt shape IoU
(weight 1 -> 8): Corgi 0.1002 -> 0.3668, Bat 0.0252 -> 0.2218, Shark
0.0196 -> 0.2428, RiggedFigure 0.0382 -> 0.1189. Correct-prompt skeleton
IoU improved respectively 0 -> 0.0835, 0.0051 -> 0.1218, 0.0101 -> 0.1408,
and 0 -> 0.0496. Under weighting, correct prompt shape IoU exceeded swapped
prompt IoU for all four assets averaged over those seeds, unlike baseline.
However, weighted skeletons predict on average 188–532 voxels for targets
of 43–81 voxels. Training is only 300 steps on three related animal assets;
the validation asset has been examined previously. Results are not evidence of
production quality or untouched generalization, and no image conditioning was
trained.

A four-view, two-FOV Blender render integration probe also completed for the
Avocado sample with matching recorded file hashes. A brighter ambient setting
was applied after inspecting the first renders; VLM filtering remains unproven.

## Multi-view CLIP scoring probe submitted (2026-11-04)

Added `tools/score_curation_views.py`, which checks render image hashes and uses
a pinned frozen CLIP model to report per-view and mean cosine similarities for
curator-provided prompt categories; it makes no automatic filtering decisions.
Created a tiny diagnostic set of Avocado plus generated cube, sphere, and plane
(two azimuths and two FOVs each). Kaggle v21 was submitted to check the scoring
path and these examples, source SHA256
`a6db99380c6ebbef08c8ac9a73c5bbc8f9b4736d6b615b053acbdf3060a5056b`.
There is no labeled quality corpus or calibrated decision threshold and no
result is asserted until the job completes.

## Skeleton threshold diagnosis prepared (2026-11-04)

Prepared `tools/audit_prompt_skeleton_threshold.py` and a T4 notebook to
re-evaluate the saved v20 weighted checkpoint at thresholds 0.1–1.0 using
three new noise seeds per asset. The threshold will be selected on the three
training assets, and reported separately on the one previously inspected
validation asset; this is not an untouched benchmark. The original checkpoint
and report were stored in a private Kaggle dataset with hash verification.
Notebook source SHA256 is
`afa057b1212e0d04eab0ce589c9c0ec83c772486240b7d6ad02ce9b346c7b738`.
The audit has not been submitted while the v21 CLIP job is running.

Kaggle v21 CLIP scoring completed on T4 and all four reports carry the expected
source hash. Mean object vs primitive cosine scores: avocado 0.2628/0.2512,
cube 0.2439/0.2437, sphere 0.2610/0.2544, plane 0.2386/0.2396. The cube and
sphere incorrectly favor the object category under a simple maximum-score rule,
so these prompts and renders are not reliable enough to filter a corpus. This
four-example diagnostic does not establish classifier quality, and no assets
were automatically rejected. Reports are in `artifacts/kaggle-curation-clip-v21/`.
The skeleton threshold audit was subsequently submitted as Kaggle v22; no
threshold results are asserted until it completes and is inspected.

## Curation crop diagnostic prepared (2026-11-04)

Added optional alpha-bounded cropping with a margin and square padding to CLIP
scoring; the default remains the original uncropped input. A local image check
verified foreground preservation, square padding and rejection of empty alpha.
Rendered four views of two additional known assets, Corgi and RiggedFigure, and
prepared a six-asset cropped scoring notebook (source SHA256
`fd3e8164bf34ee89a76c85f111a9c08b021a80b64f0a41422de1e9e923787b71`). This is
an exploratory diagnostic on known examples, not a calibrated filter. The v22
threshold audit completed and its output is downloading; no v23 job has been
submitted yet.

Kaggle v22 completed and its report has the expected source hash. Among
thresholds 0.1–1.0, 0.6 maximized mean IoU on the three training assets across
three new noises each. Compared with 0.5, mean training skeleton IoU changed
0.1080 -> 0.1114 and predicted voxel count 302.8 -> 257.9 (target 60.7).
On the previously inspected validation asset, IoU changed 0.0496 -> 0.0516,
recall 0.2636 -> 0.2403 and predicted count 201.7 -> 170.0 (target 43).
This small change does not solve overprediction; no default was changed.
Report: `artifacts/kaggle-prompt-skeleton-threshold-v22/prompt-skeleton-threshold.json`.
Kaggle v23 was submitted for the foreground-crop CLIP diagnostic; results are
pending and the known examples remain unsuitable for measuring generalization.

## Experimental sparse Triton convolution prepared (2026-11-04)

Added `t3d/triton_sparse.py` as an opt-in prototype: a Triton float32 forward
convolution over a PyTorch-built neighbor map with a PyTorch backward pass.
Five randomized CPU neighbor maps matched an independent dictionary lookup.
It is not wired into the default SC-VAE, and CUDA execution, gradient parity,
performance and high-resolution memory behavior remain unverified. Prepared
`tools/probe_triton_sparse.py` and its T4 notebook (source SHA256
`5cd6a28451ba45a9f472273294c66c99e7744fb8decaa3c2dc8854dd4d38eef6`)
for forward/gradient comparisons and end-to-end forward timings. It has not yet
been submitted while the v23 curation job is live.

Kaggle v23 completed; all six cropped-view reports have the expected source
hash in `artifacts/kaggle-curation-clip-crop-v23/`. Averaged object/primitive
cosines were avocado .2607/.2466, Corgi .2552/.2259, RiggedFigure .2479/.2276,
cube .2388/.2403, plane .2371/.2391 and sphere .2566/.2492. The simple
maximum-score rule would still miss the sphere; these known examples cannot
calibrate or validate a curation filter. No automatic filtering was enabled.

Two local neighbor-map/fallback unittests passed for the experimental sparse
kernel. Kaggle v24 was submitted with the previously recorded source bundle to
check CUDA forward and gradients against the PyTorch reference and measure
small-case forward times. The default training path remains unchanged.

Kaggle v24 completed on T4 and the downloaded report's source hash matches
`5cd6a28451ba45a9f472273294c66c99e7744fb8decaa3c2dc8854dd4d38eef6`.
Forward outputs and feature/weight/bias gradients passed parity checks against
the PyTorch reference on three small random sparse grids (31–512 active cells,
4–33 channels); worst absolute error was 7.63e-6. Forward wall time including
validation and neighbor mapping was 8.0–8.5 ms vs 10.6–11.5 ms for reference.
These small-case timings do not establish production-scale performance or
optimized backward behavior. Seven existing sparse-block/compression tests also
passed locally. Report: `artifacts/kaggle-triton-sparse-v24/triton-sparse-probe.json`.

Submitted Kaggle v25 to test 2,048 and 9,600 randomly sampled cells from a
synthetic 64-cubed box surface at 32/64 channels, source SHA256
`8aeb0b4816e7e6475dde1e259bf660d7b22d2d1c6532fca831bf5429df134407`.
These remain synthetic probes rather than a full SC-VAE training benchmark.

Kaggle v25 synthetic surface test completed with matching source hash. At
2,048 cells/32 channels and 9,600 cells/64 channels, forward and feature,
weight and bias gradients passed reference comparisons; the largest absolute
gradient difference was 6.11e-5. End-to-end forward times, including mapping,
were 8.42 vs 10.66 ms and 10.74 vs 12.01 ms respectively. This covers one
sampled synthetic box surface on one T4, not real SC-VAE performance.
Report: `artifacts/kaggle-triton-surface-v25/triton-surface-probe.json`.

Submitted v26: compare deterministic loss, all parameter gradients, forward+
backward timing and peak incremental allocation with experimental versus
reference blocks in a tiny SC-VAE on the existing prepared Avocado asset.
Source bundle SHA256
`0ef37cc49478da3965eb0dc24e30b4294efab8bca36ae6fcc4f2c18d362d709c`.
It is a 16-cubed, teacher-forced single-asset integration test, not a full
training-quality or production benchmark. No default backend was changed.

Kaggle v26 completed with the expected source hash. On one prepared 16-cubed
Avocado asset (498 active voxels), the reference and experimental tiny VAEs
had identical deterministic loss 1.6802964 and passed all parameter-gradient
checks (largest absolute difference 1.49e-8). Three forward/backward timings
averaged 188.8 ms reference vs 161.7 ms experimental, with measured incremental
CUDA allocation 1,053,696 vs 310,272 bytes. This is one teacher-forced case,
not an optimizer or high-resolution benchmark.

Wired the experimental path as `--convolution-backend triton` in the training
CLI and as `convolution_backend='triton'` in SC-VAE, retaining reference as the
default and checkpoint-compatible state dictionary keys. The trainer stores
non-default backend selection and rejects an explicit resume mismatch. CPU tests
covering the existing VAE, training/resume and sparse blocks passed, as did
three backend/map checks; Triton remains float32 CUDA only. Kaggle v27 was
submitted to compare three paired optimizer steps and verify Triton checkpoint
resume on the prepared asset, source SHA256
`b8d7ecc5107355532966534865c3d6f7c6ac37d66c199dcb37dc9ac9e76c3009`.
No v27 results are asserted yet.

Kaggle v27 completed with the expected source hash. Three paired stochastic
optimizer updates on the 16-cubed Avocado asset produced matching losses and
maximum parameter difference 2.98e-8; saving and resuming the Triton trainer
reproduced the next update and restored the backend. Report and checkpoint are
in `artifacts/kaggle-triton-training-v27/`. This is a short integration test.

Built a separate 32-cubed shape-only dataset from the previously checked
Avocado and WaterBottle custom voxelizations, verifying their hashes against the
paired conversion report and recording original source hashes. It contains
1,991 train and 1,904 validation voxels; the objects have been inspected
previously and this is not a held-out generalization benchmark. Kaggle v28 was
submitted for 30 matched training steps per backend, timing, peak allocation,
and validation losses, source SHA256
`c2d6ba8fbde9ed829924a11ad4095a652e183e8d03ef74c2c00eb89add6f43e5`.

## 32-cubed paired training and free-support audit (2026-11-04)

Kaggle v28 completed with verified report/source and checkpoint hashes. Across
30 steps, reference and Triton reported identical displayed losses and
teacher-forced validation total 1.6866 -> 1.6508. Elapsed time was 6.76 vs
6.67 seconds; peak extra CUDA allocation was 36.6 vs 27.7 MB in this one run.
The timing difference is too small to establish a robust speedup. These are
small shape-only VAEs trained on one asset. The first full output download hit
a connection interruption; a retry succeeded and essential checkpoint hashes
matched. Evidence: `artifacts/kaggle-triton-paired-32-v28/`.

A local CPU check of the reference checkpoint on WaterBottle showed that
teacher-forced support is exact by construction, while predicted support at
threshold 0.5 had 91 intersecting voxels, 2,206 predictions, 1,904 targets and
IoU 0.0226. This exposes a substantial free-decoding failure despite the small
teacher-forced loss decrease. Submitted v29 to audit predicted and teacher
support for both paired checkpoints and both assets, including extracted mesh
triangle counts; source SHA256
`bf1ef6d60f1299be14e94395f97a783df3d448af0a6181078c34da217520e020`.
The private dataset contains the hash-verified v28 checkpoints and report.

## Coarse-to-fine support diagnosis (2026-11-04)

Added `tools/audit_32_decoder_trace.py` and ran the reference v28 checkpoint
on CPU. For the training asset, free decoding retained 14/28 occupied voxels
at the first decoder scale, and 122/1,991 at the finest scale; 1,783 of 1,869
final misses had already lost a parent. On validation, first-scale recall was
8/16, and 1,756 of 1,813 final misses were unreachable due to earlier pruning.
Trace: `artifacts/kaggle-triton-paired-32-v28/decoder-trace-reference.json`.
These are posterior reconstructions, not prompted generation results.

Prepared a continuation of the v28 Triton checkpoint from 30 to 600 steps,
reporting teacher loss and free-support metrics at 30, 100, 300 and 600 steps
on the same one training and one previously inspected validation asset; source
SHA256 `e37ab7bd837877809358d3c97a6a5c718458d1091c34a0519e14145ebd74b8f5`.
This probe is intended to test whether additional optimization improves early
pruning; it has not yet been submitted.

Kaggle v29 completed; all eight reconstruction artifacts and source hashes
match the report. Reference and Triton produce identical free-support counts:
train IoU 0.0299 (122/1,991 target voxels retained, 2,214 predicted; 70 mesh
triangles), validation IoU 0.0226 (91/1,904 retained, 2,206 predicted; 92
triangles). Teacher support was supplied ground truth and is not evidence of
learned occupancy quality. Report: `artifacts/kaggle-triton-reconstruction-v29/paired-reconstruction-audit.json`.
The 600-step continuation was subsequently submitted as Kaggle v30; results
are pending and should be interpreted only within this tiny pilot.

## Coarse-stage loss experiment prepared (2026-11-04)

Added optional normalized coarse-to-fine occupancy loss weights to SC-VAE and
its trainer/CLI, preserving equal weighting by default and checkpointing the
chosen objective. Four VAE and three trainer tests pass, including checking a
coarse-only loss against its decoder stage and weighted checkpoint resume.
Prepared a matched continuation from the same step-30 checkpoint with weights
`[4,2,1,1]`, reporting both weighted and comparable unweighted teacher loss and
free support; source SHA256
`a7c72309261d08a948c99a536426bd65636dedad95559cf042612699cb73ad7d`.
The baseline v30 is still running; early streamed metrics at step 300 show
free occupancy IoU 0.1378 train and 0.1183 validation, versus 0.0299/0.0226
at step 30. This is one seed and one asset per split; final results pending.

## 600-step baseline and weighted comparison (2026-11-04)

Kaggle v30 completed; source, final checkpoint and both reconstruction hashes
were verified. Free-support IoU at steps 30/100/300/600 was
0.0299/0.0284/0.1378/0.2944 on training and
0.0226/0.0268/0.1183/0.1438 on validation. Teacher-forced validation loss fell
from 1.6508 to 1.4818 at step 300, then rose to 1.9957 at step 600, consistent
with overfitting in this one-asset pilot. Only the final baseline checkpoint was
saved, so step-300 quality cannot be re-audited from its weights.

A CPU reference-backend trace of the final checkpoint showed first-stage recall
improved from 0.5 to 1.0 for both assets; finest-stage recall was 0.437 train
and 0.253 validation, with 448/889 target voxels respectively unreachable due
to earlier pruning. This is not evidence of production fidelity or generalization.

Submitted v31 for the matched coarse-weighted continuation. The builder was
updated to retain step-100 and step-300 checkpoints before submission; its
actual source SHA256 is `5bfea7b38b665ad216c62f407ad69067a89dd7d46608176c3577f79933028027`.

A 128-sample CPU surface comparison of the step-600 validation reconstruction
against the prepared 32-cubed WaterBottle voxel mesh found 328 predicted vs
4,172 reference triangles, mean bidirectional distance 0.0630 canonical units,
and F-score 0.0276 at tolerance 0.01. The extracted prediction had 455
boundary edges and one nonmanifold edge. This uses a small sample and a
voxelized reference, not the original source mesh or the blueprint benchmark;
it confirms that occupancy improvement has not yet yielded an adequate mesh.
Details: `artifacts/kaggle-triton-occupancy-32-v30/validation-surface-audit.json`.

## Source-surface evaluation and mixed-asset pilot (2026-11-04)

Compared the baseline v30 predicted meshes directly against normalized original
GLB surfaces at three seeds, 128 samples each. Mean unsquared distance/F-score
at tolerance 0.01 was 0.0571/0.0329 for train Avocado and 0.0641/0.0048 for
validation WaterBottle. The prepared input voxelizations scored 0.00017/1.0
and 0.00231/0.9921 respectively. These small sampled measurements show the
trained decoder is much worse than its prepared input, but are not the named
research benchmarks. Full report and source hashes:
`artifacts/kaggle-triton-occupancy-32-v30/original-surface-audit.json`.

Kaggle v31 finished and checkpoint/source hashes verified. At steps 300/600,
coarse-weighted free-support IoU was 0.1200/0.2658 train and 0.1071/0.1400
validation, versus baseline 0.1378/0.2944 and 0.1183/0.1438. It did not improve
this one-seed pilot; the equal-weight default remains unchanged. Weighted
snapshots at steps 100, 300 and 600 were retained. Results:
`artifacts/kaggle-triton-coarse-weight-32-v31/`.

Built `artifacts/mixed-shape-pilot-32/`: four training assets (Avocado and three
related CC0 Gobkit animals), and two previously inspected validation assets
(WaterBottle and CC-BY RiggedFigure). It contains source hashes, license
attribution and source-family labels; it is shape-only and is not a curated or
untouched benchmark. All six files loaded and matched their manifests. Submitted
Kaggle v32 to warm-start the step-30 Avocado model, preserve its optimizer and
explicitly switch to this dataset, then round-robin the four training assets to
step 600. Source SHA256
`a52b2282ea21053aaae18445cda1d27fead375536629f84885bac2aa96693455`.
Results will not support a causal dataset comparison without matched controls.

## Deterministic multi-asset order (2026-11-04)

Added optional per-epoch shuffling to the regular training CLI via
`--shuffle-seed`. Ordering uses a separate local RNG, caches one permutation
per epoch, and does not consume model RNG. New CLI checkpoints record shuffle
seed and batch size; explicit mismatches are rejected on resume. Legacy
checkpoints have no batch-size metadata, so their original batch size cannot be
verified. Four trainer tests passed, including shuffled order across an epoch,
checkpoint restoration and RNG independence. The ongoing v32 notebook used
its own explicit round-robin order and was submitted before this change.

Kaggle v32 completed; both checkpoints, all six final reconstructions and
source hashes verified in `artifacts/kaggle-triton-mixed-32-v32/`. At step 600,
free-support IoU was Avocado 0.186, Corgi 0.159, Bat 0.156, Shark 0.166,
WaterBottle validation 0.146 and RiggedFigure validation 0.063. This warm-start
and different asset schedule cannot establish a causal comparison to the
single-asset run; all assets were previously inspected and mesh fidelity still
needs verification.

Fetched 24 source-pinned GLBs from six Kenney packs into
`artifacts/kenney-raw-pilot/`, with creator, original pack URLs, source hashes,
and whole-pack split groups. The [mirror catalog](https://github.com/Hidencod/tge-assets)
identifies them as Kenney CC0 assets; the corresponding Kenney pack pages
also identify CC0 (for example https://kenney.nl/assets/car-kit and
https://kenney.nl/assets/food-kit). This is a tiny raw acquisition sample,
not a filtered or semantically deduplicated corpus. Initial strict import
accepted 6/24 and rejected 18, often due to texture handling or degenerate
triangles. A separate geometry-only static GLB import now permits material-only
UV transforms while retaining mesh/animation/unknown-extension checks; five
importer tests pass. Conversion with explicit removal of zero-area faces is
being evaluated separately, and it does not preserve material data.

## Pack-separated Kenney shape pilot (2026-11-04)

Converted all 24 pinned Kenney mirror GLBs to shape-only 32-cubed .t3d assets
using `tools/prepare_kenney_pilot.py`. The conversion reports 72 zero-area or
grid-collapsed triangles removed across the assets; no material or rig data is
retained. One initially rejected asset was retried with the explicit grid-space
collapse check. All files and source groups were verified through AssetDataset:
16 training assets across car, food, furniture and nature packs; eight validation
assets across space and pirate packs. This is a source-family split, not proof
against semantic duplicates; the set has not been VLM-filtered or quality scored.
Manifest SHA256 is
`59b1a8a38d4e34889777a42ebffaa24c0c0a46547b4fd881e88b349fb632a5a4`.
Acquisition and conversion evidence is under `artifacts/kenney-raw-pilot/` and
`artifacts/kenney-shape-pilot-32/`.

Submitted Kaggle v33 with private datasets holding the verified v32 mixed-model
checkpoint and the Kenney pilot. It explicitly warm-starts the previous model,
then trains 600 more steps with deterministic shuffled order and snapshots at
steps 800/1000/1200, reporting per-asset free occupancy. Source SHA256
`9983e9c237d1b83c82bfc1a9b0c5b5ff55ef14c22d9bbbb2e302e511ab15086c`.
Pack separation does not guarantee independent shapes; this is an uncurated
small pilot, not the requested large training corpus or a production benchmark.

## Candidate index and voxel-fidelity screen (2026-11-04)

Created a source-pinned catalog index for 944 candidates across the six Kenney
packs; only 24 have been downloaded. The 944 entries are references with
expected sizes and pack-level splits, not a downloaded or quality-filtered
corpus. Index: `artifacts/kenney-raw-pilot/candidate-index.json`.

Measured the 24 prepared voxel meshes against their normalized source GLBs using
64 samples each at distance tolerance 0.01. All comparisons were valid; mean
sampled F-score was 0.996 train and 0.994 validation, with the lowest 0.977.
This narrow check is for input conversion fidelity only; it does not validate
material, topology, VLM quality, semantic uniqueness, or learned reconstruction.
Report: `artifacts/kenney-shape-pilot-32/voxel-quality-audit.json`.

Started bounded, resumable raw acquisition of the 944 catalog entries using
`tools/fetch_kenney_index.py` (four workers, per-file SHA256 metadata and
atomic writes) into `artifacts/kenney-catalog-download/`. At the latest check,
100 entries had completed with no reported errors; the local process is still
running (session 95995). These are raw candidates, not converted or vetted
training assets. The script was tightened after launch to validate IDs and URLs
on future invocations; the running process loaded the earlier version and uses
the already-validated immutable catalog index.

## Full six-pack raw download and sharded preprocessing (2026-11-04)

The pinned catalog download finished: all 944 entries were re-hashed against
per-file acquisition records with zero reported failures. They remain raw and
unfiltered. The private Kaggle dataset is `t3d-kenney-pinned-raw-catalog`;
the uploaded archive SHA256 is
`a99eb47c47d5901ec0edbe068eaffe6eea61aaf87d87672032dea6d6bcefcde2`.
A local two-asset conversion shard succeeded and its two .t3d files loaded.
Prepared a CPU Kaggle conversion notebook for one of five interleaved shards;
it records per-asset failures and checks exact duplicates within the shard only.
The first notebook failed before processing because its fixed input-index path
was not found. Dataset listing confirms the index is present; the revised v2
embeds the index and discovers the mounted raw root, and is now running at
`sumu143/t3d-kenney-corpus-conversion-shard-0`. The v2 source SHA256 is
`7506aaf2adde39e39ae751c00a748b7b5088d94e88ee2ab757db9ae83ff782a8`.
Cross-shard and semantic deduplication and quality filtering remain unfinished.

Kaggle v33 finished; all three checkpoints, eight validation reconstructions
and the source hashes were verified in `artifacts/kaggle-triton-kenney-32-v33/`.
Mean train/validation free occupancy IoU was 0.0974/0.0968 at warm-start step
600, 0.1843/0.1465 at 800, 0.2183/0.1608 at 1000, and 0.2516/0.1563 at 1200.
Validation teacher-forced loss fell 1.462 -> 1.317, while free IoU peaked at
step 1000 on these eight models. One warm-started run on a small uncurated set
cannot establish generalization or reliable mesh quality.

## Separate 32-cubed material VAE pilot (2026-11-04)

Prepared strict-PBR 32-cubed Avocado/WaterBottle .t3d inputs with source hashes
and attribution in `artifacts/material-pilot-32/`; all six PBR channels load.
The diagnostic conversion explicitly permits 11 and 208 multi-crossing edges
respectively, so these inputs can lose thin/overlapping geometry and are not
production-quality corpus evidence. Submitted Kaggle GPU v34 to train an
independent material VAE for 600 steps on Avocado and report per-channel errors
at 0/100/300/600 on both previously inspected assets using target support.
Source SHA256 `9087826cfad5a4b09d847fd0ac2235432ac01a78cf5049cda133fbd47d0f32e4`.
This is not image/text-conditioned generation, free-support material decoding,
rendered PBR evaluation, or a generalization benchmark.

The separate CPU corpus-conversion notebook's revised v2 found the mounted raw
catalog at Kaggle's qualified `/kaggle/input/datasets/...` path and is running
shard 0; other shards and cross-shard deduplication remain pending.

After shard 0 processed its first 20 candidates without reported failures,
submitted the remaining four independent CPU shards. All five Kaggle notebooks
reported RUNNING at the status check, each with source bundle SHA256
`7506aaf2adde39e39ae751c00a748b7b5088d94e88ee2ab757db9ae83ff782a8`.
Each processes an interleaved fifth of the 944 candidates. Prepared
`tools/merge_kenney_shards.py` to verify reports, raw/prepared hashes and
split provenance, and remove exact source or prepared-content duplicates
across shards. It is not yet run; semantic deduplication and quality filtering
remain required.


## Full Kenney corpus merged, deduplicated, and GPU scaled training (2026-11-04)

All five parallel CPU conversion shards completed on Kaggle and their report/zip
artifacts were downloaded to `artifacts/kaggle-kenney-corpus-shard0/` through `shard4/`.
Executed `tools/merge_kenney_shards.py` to reconcile all 944 catalog candidates.
Across the entire catalog, exactly 1 candidate was rejected (`pirate-kit/chest`,
rejected strictly due to containing skeletal rigging/animation, avoiding silent
loss of articulation in a geometry-only pass), and 75 within-split exact duplicates
were removed.

The final consolidated dataset contains 868 verified 32-cubed `.t3d` assets
in `artifacts/kenney-corpus-32/`: 649 train assets across car, food, furniture,
and nature packs; and 219 validation assets across pirate and space packs.
All 868 assets load cleanly through `AssetDataset`. Manifest SHA256 is
`af2f3179db2cbfe1288a3ab9e67e0d7093df61a7591b9c7586cc782fc72824bc`.
Pack separation guarantees source-pack isolation between training and validation.

Bundled the complete 868-asset dataset into `kaggle-kenney-corpus-32/kenney-corpus-32.zip`
(12.6 MB) and published the private Kaggle dataset `sumu143/t3d-kenney-corpus-32`.
Verified all 870 files unzipped and ready on Kaggle.

Created `tools/train_kenney_corpus_32.py` and `tools/build_kenney_corpus_training_notebook.py`.
The training pipeline warm-starts from `triton-mixed-600.pt` and trains for 1,300
gradient updates (steps 600 through 1900, roughly 2 full epochs over the 649 training
models), evaluating teacher-forced loss and free-support occupancy IoU across 24
balanced held-out validation models from pirate and space packs at steps 600, 900,
1200, 1500, and 1900. Added `tests/test_kenney_corpus_eval.py`; all 161 unit tests
pass. Pushed GPU training notebook `sumu143/transformer-native-3d-kenney-corpus-training`
to Kaggle, running on dual Tesla T4 GPUs. Handover state documented in `note.md`.


Kaggle version 35 completed in 641.28 seconds on dual Tesla T4 GPUs. All four
checkpoints (`triton-kenney-corpus-900.pt`, `1200.pt`, `1500.pt`, `1900.pt`), the
24 held-out predicted `.t3d` validation reconstructions, and the run report were
verified and downloaded to `artifacts/kaggle-kenney-corpus-training-v35/`.
Mean validation free-occupancy IoU across the 24 balanced held-out models
monotonically increased: 0.0998 at step 600, 0.1703 at 900, 0.1965 at 1200,
0.2218 at 1500, and 0.2536 at 1900. Mean validation loss dropped steadily from
1.4845 to 1.0526. Unlike the 16-asset pilot (v33), which plateaued and suffered
an overfitting reversal after step 1000 (falling 0.1608 -> 0.1563), training
across the 649-asset corpus sustained continuous generalization gains (+57.7%
IoU relative improvement over v33 peak). All 161 local unit tests pass.


## Voxelizer hardening and 32³ vs 64³ resolution scaling audit (2026-11-04)

Hardened `t3d/voxelize.py` to detect and filter degenerate or collinear faces
where cross-product surface normals approach zero, preventing divide-by-zero
NaNs and `ValueError: Face requires finite matching (N, 3) points and normals`.
All existing and new voxelization unit tests pass.

Evaluated geometric fidelity of $32^3$ versus $64^3$ grid resolution against
original normalized GLBs using `surface_comparison` at 256 area-weighted samples
and a tight 0.005 tolerance across representative assets:
- `car-kit/ambulance` (2,788 faces): mean distance fell from 0.00097 ($32^3$)
  to 0.00010 ($64^3$) (10× error reduction); F-score (0.005) rose 0.9252 to 0.9961.
- `pirate-kit/ship-medium` (1,716 faces, thin masts): mean distance fell from
  0.00179 to 0.00040 (4.5× error reduction); F-score rose 0.8633 to 0.9881.
- `space-kit/alien` (285 faces): mean distance fell from 0.00013 to 0.00001
  (13× error reduction); F-score reached 1.0000.
- `nature-kit/tree-thin-fall` (228 faces): mean distance fell from 0.00025 to 0.00005
  (5× error reduction); F-score rose 0.9902 to 0.9980.

Active surface voxel counts scaled predictably ($32^3$: 1,000–3,800 voxels,
19–68 KB; $64^3$: 4,300–15,500 voxels, 77–272 KB), confirming that $64^3$
representation remains compact (mean ~180 KB per `.t3d` file) while reducing
geometric surface distortion by an order of magnitude.

## Stage 1 Text-to-3D Coarse Occupancy DiT Flow on Kenney Corpus (2026-11-04)

Generated aligned natural English prompts for all 868 Kenney models using
`tools/generate_corpus_prompts.py`, saved in `artifacts/kenney-corpus-32/prompts.json`.
Implemented `tools/train_kenney_occupancy_dit.py` and
`tools/build_kenney_occupancy_dit_notebook.py` to train a Stage 1 DiT on downsampled
16³ coarse occupancy grids (4,096 tokens) conditioned on frozen HuggingFace CLIP
(`openai/clip-vit-base-patch32`) text embeddings via a trainable 32-channel linear
adapter.

Configured `SparseFlowDiT` with `width=96, heads=4` to satisfy 3D Gram-Schmidt
axial RoPE requirements (`head_dim=24` divisible by 6). Submitted Kaggle GPU
version 37 (`sumu143/transformer-native-3d-occupancy-dit-flow`).
The run completed in 201.88 seconds on dual Tesla T4 GPUs. Checkpoint
`occupancy-dit-600.pt` (SHA256 `1cce8fc76b1f6258ded81263dfbea021a3930807a86e4c15717e3ce944b185b5`)
and report `occupancy-dit-report.json` were retrieved and verified into
`artifacts/kaggle-kenney-occupancy-dit-v37/`.

On 24 held-out validation models from pirate and space packs, 15-step Euler flow
sampling demonstrated an average prompt difference of 8.83 cells when text
prompts were swapped against mismatched assets, proving that cross-attention
steerability guides 3D spatial occupancy. All 163 unit tests pass.
