# Transformer-native 3D format

Implementation of the supplied blueprint, with its complete Part 2 preserved in
[plan.md](plan.md). See [PROGRESS.md](PROGRESS.md) for evidence and remaining work.
An experimental Track A/Track B container implementation exists; production
generators remain pending, while the CUDA export pipeline has passed small
synthetic and one-real-rig integration pilots. Validated raw Track A
tensors and Z/TZ/H/TH serialization mappings are available in
`t3d.schema.VoxelAsset` and `t3d.serialization`. `VoxelAsset.canonical()` sorts
all present geometry, material, and rig arrays together. `serialization_order()`
returns order and inverse indices for indirect feature access;
`serialization_orders()` returns all four variants. Hilbert uses the CPU
`hilbertcurve` package. Raw binary I/O is available through `t3d.container.save/load`
and `dumps/loads`; see [the experimental profile](docs/container.md).
`t3d.qef.solve_qef` fits bounded local dual vertices from Hermite intersection
positions and normals. `t3d.voxelize.mesh_to_voxels` now discovers intersections
by direct triangle/cell clipping and returns a geometry-only `VoxelAsset`:

```python
from t3d.voxelize import mesh_to_voxels

asset = mesh_to_voxels(
    vertices=[(.4, 0, 0), (.4, 1, 0), (.4, 0, 1)],
    faces=[(0, 1, 2)], grid_size=(8, 8, 8), aabb=[(0, 0, 0), (1, 1, 1)],
)
```

This is a CPU reference converter. Production acceleration and robust handling
of grid-coincident edges remain pending; see the limitations in PROGRESS.md.
Extract triangles with `t3d.extraction.voxels_to_mesh(asset)`. Repair raw winding
with `faces, audit = t3d.topology.orient_faces(vertices, faces, outward=True)`.
The outward option treats closed shells independently; nested cavity semantics
require containment analysis. Open sheets keep their first face's direction.
Run `python -m tools.roundtrip_demo` for a cube
round trip and inspectable OBJ/JSON artifacts under `artifacts/roundtrip`.

Device-resident extraction is implemented in `t3d.extraction_torch`; see
[GPU extraction scope](docs/cuda-extraction.md). It uses sorted-key lookup;
the custom hash kernel and broad real-asset latency validation remain pending.
Synthetic sheet extraction met 100 ms through 1536³; standard CuMesh cleanup/UVs
passed a separate cube probe. These results do not establish production quality.

An integrated raw export command is available as
`python -m t3d.export input.t3d output.glb`; see [export usage](docs/export.md).
It connects CUDA extraction, CuMesh UV preparation, sparse PBR baking and GLB
packaging. Source-matched Kaggle T4 pilots cover synthetic and real-rig export;
checksummed `.t3d` ingestion through the public CLI has also passed on one
real-rig-derived asset. Source-material transfer, animation deformation and
production quality remain unverified.

Install locally with `python -m pip install -e .` (requires NumPy and hilbertcurve), then run
`python -m unittest discover -s tests -v`.

Current preprocessing API:

```python
from t3d.normalization import normalize_asset
from t3d.augmentation import quarter_turn
from t3d.sampling import sample_conditioning

vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
faces = [(0, 1, 2), (1, 0, 3)]
mesh, joints, transform = normalize_asset(vertices, joints=[(0, 0, 0)])
mesh = quarter_turn(mesh, 1, axis='y')
joints = quarter_turn(joints, 1, axis='y')
uniform, sharp = sample_conditioning(
    mesh, faces, uniform_count=4096, salient_count=4096, seed=42,
)
# Each row is [x, y, z, nx, ny, nz]. Sharp may be empty on smooth meshes.
```

Quarter-turns assume the asset has already been aligned upright. They do not
infer its pose. Apply the inverse rotation before using `transform.inverse`.

To attach source materials, use `t3d.materials.sample_pbr(asset, vertices, faces,
materials, face_materials, corner_uv=uv)`. Materials are `PBRMaterial` instances;
textures are decoded float RGB/RGBA arrays and UVs have shape `(F,3,2)`. Output
channels are linear RGB, metallic, roughness, alpha. This CPU reference supports
bilinear repeat/clamp sampling and optional exact `projection_method='bvh'` for
repeated large-mesh queries.

Rig preprocessing lives in `t3d.articulation`: `Skeleton` validates parent graphs,
`transfer_skin_weights` projects source weights onto query points,
`skin_embeddings` aggregates supplied joint embeddings, and
`attach_articulation` samples the skeleton field at voxel centers. The
experimental SkinAE implementation and inference decoder are available; see
`docs/articulation-inference.md` for measured limits. Rigged GLB factors and
embedded base-color/metallic-roughness textures can be requested through
`load_rig_glb(..., include_pbr=True)`; unsupported PBR extensions are rejected.
`linear_blend_skinning` supports deformation checks with supplied skin matrices;
`rig_joint_world` returns bind or sampled joint world matrices and
`joint_deformation_matrices` composes them into bind-relative motion.
`t3d.animation.retarget_animation` maps source joints to a packed rig by bind
position, bakes source motion into LINEAR glTF TRS tracks, and appends them to
the GLB; `validate_animation` performs structural track checks. Run
`python -m tools.audit_posed_deformation` to retarget the available clips and
compare posed surfaces. It currently reports that SkinAE weight error, not
retargeting or track encoding, dominates the measured error on the one
reconstructed export; same-topology rigs round-trip near float precision.
`t3d.glb.rewrite_skin_weights` and `python -m tools.reweight_skin_glb` limit
weights in an existing GLB; `export_raw` also accepts `max_skin_influences`
(CLI `--max-skin-influences`). Limiting the v13 SkinAE weights to two influences
cuts the measured audit error from 0.0179 to 0.0109 and raises F-score from 0.60
to 0.78 with valid animation round-trips, but rendered quality is still
unchecked, so the export default remains 16.

Track B tokenization is available as `t3d.bpt.encode(vertices, faces)` and
`decode(tokens)`. Vertices must be in [-0.5,0.5]; decoding is lossy at 128-bin
precision. See [BPT conventions and limitations](docs/bpt.md). Token sequences
can be passed to `t3d.container.dumps(asset, bpt_tokens=tokens)`.

GPU work runs in the private [Kaggle notebook](https://www.kaggle.com/code/sumu143/transformer-native-3d-gpu-bootstrap).
The initial run verified dual Tesla T4 allocation, project tests, and CUDA gradient
updates. See [Kaggle workflow](kaggle/README.md) for submission and output retrieval.

Step 4 resampling foundations are in `t3d.sparse_compression` (requires PyTorch
from the Kaggle GPU runtime). They include differentiable 2× shortcuts, reference
learned residual branches, and occupancy prediction/pruning. A composed VAE has
passed small CUDA probes; plan-scale SC-VAE training and its rendering losses
remain pending. See PROGRESS.md for evidence and gaps.
Submanifold convolution and ConvNeXt-style residual blocks are available in
`t3d.sparse_blocks`; see [compression details](docs/sparse-compression.md).

Prepared `.t3d` corpora can use the [manifest and resumable training workflow](docs/training.md).
It preserves source/license metadata, verifies asset hashes, separates validation,
and saves optimizer/RNG/dataset position for resume. A curated real-asset corpus
and the full production training protocol remain pending.

Prepare static textured GLB files with `python -m t3d.import_glb input.glb output.t3d
--grid-size 32`. See [import scope and real-asset evidence](docs/import.md).
The Avocado integration fixture is prepared locally; large-scale dataset curation
and held-out training still remain.

[VAE reconstruction diagnostics](docs/reconstruction.md) now save decoded meshes
and surface measurements. The short pilot currently fails reconstruction quality;
passing execution tests does not indicate a usable trained generator.

Compressed latent storage uses `t3d.latent_container`: profile 2 records latent
coordinates/dimensions, shape/material/rig streams, decoder checkpoint SHA256s,
and optional joint hierarchy. Use `load_latents(..., expected_models=...)` to
check the decoder identity. See [binary layout](docs/container.md). The profile
requires external matching decoder checkpoints and does not imply model quality.

Step 5 foundations are in `t3d.flow`: conditional sparse DiT, Gram-Schmidt 3D RoPE,
rectified-flow loss and CFG/Euler sampling. See [flow scope](docs/flow.md).
The three-stage orchestration has passed execution tests, while trained stages,
conditioning encoders and end-to-end generation remain pending.

`t3d.generation.ThreeStageFlow` orchestrates occupancy, geometry/optional rig
latents, then shape-conditioned material latents. See [stage semantics](docs/generation.md).
It currently provides execution foundations rather than a trained usable generator.
