# Raw Track A export command

The integrated command is:

```sh
python -m t3d.export input.t3d output.glb --source-up y --target-faces 100000 --texture-size 2048
```

Rigged raw assets also require the frozen SkinAE checkpoint:

```sh
python -m t3d.export input.t3d output.glb --skinae-checkpoint skin-pose-pilot.pt
```

`--joint-tolerance` optionally overrides endpoint-clustering tolerance; the
default is one quarter of the smallest voxel dimension.

It requires CUDA PyTorch, CuMesh, FlexGEMM, nvdiffrast and OpenCV in addition to
the base project dependencies. The pinned Kaggle integration workflow builds
these optional libraries. `--remesh` selects the plan's narrow-band remeshing
branch. `--source-up` explicitly describes the raw geometry; use y for imported
glTF coordinates and z for a z-up native asset. Rasterizer UVs are flipped when
packaged for glTF. `--alpha-mode auto` defaults to BLEND when source alpha is
below one, otherwise OPAQUE; explicit MASK accepts `--alpha-cutoff`.

The command loads checksummed raw profile 3 (and retains compatibility with
legacy raw profile 1), extracts triangles on CUDA, applies
CuMesh cleanup/UVs, bakes sparse PBR through the reference BVH and writes GLB bytes
plus `output.glb.json`. The audit includes source/output hashes, device, stage
counts, valid texture coverage, options and total elapsed time. Output files are
written through temporary files and replaced individually after export succeeds.

`export_raw(asset, ...)` is the corresponding Python API, returning bytes and an
audit dictionary. Raw PBR is required. If both articulation fields are present,
the command clusters joint and parent endpoints, samples sparse skin embeddings
onto prepared output vertices, runs the frozen SkinAE joint/vertex decoder, and
attaches normalized weights and the recovered hierarchy. The audit records
checkpoint hash, inferred joint count, tolerance and skin-influence projection.
`max_skin_influences` (CLI `--max-skin-influences`) controls how many SkinAE
influences are kept, from 1 to 16; the default remains 16. The strongest weights
are kept, renormalized and the dropped probability mass is reported. Limiting
the v13 SkinAE weights to two influences lowered the correspondence-audit mean
distance from 0.0179 to 0.0109 and raised F-score from 0.60 to 0.78, but the
default stays at 16 until rendered quality confirms that harder joint
transitions are acceptable. The low-level packer remains
lossless and rejects unrepresentable inputs. Assets with only one rig field are
rejected. Latent profile 2 requires matching trained VAE decoding first.
Optional BPT token counts are recorded in the CLI audit; this command extracts
raw Track A geometry and does not invoke Track B retopology.

Source-matched Kaggle Tesla T4 runs verified both a synthetic skinned cube and
one real RiggedSimple source through voxelization, articulation attachment,
SkinAE decoding, GLB packaging and reload. The standard real-rig export measured
mean surface distance 0.000458 and F-score 1.0 at tolerance 0.01 over 512
samples. The `--remesh` branch measured 0.004596 mean distance and F-score
0.876 at the same tolerance. Both outputs have zero welded boundary and
nonmanifold edges, and both pass the Khronos glTF Validator with no issues.
Five-seed repeats kept standard F-score at 1.0 (mean distance 0.000437–0.000464);
strict-remesh F-score ranged 0.865–0.909 (mean distance 0.004220–0.004736).
These remain 512-sample measurements on one asset.
The held-out SkinAE weight MAE is 0.207, so weight fidelity remains weak. The
independent nearest-joint baseline has MAE 0.311 and dominant-joint accuracy
0.687, compared with SkinAE's 0.207 and 0.902 on the same exported vertices.
This indicates a useful improvement over that baseline but leaves substantial
continuous-weight error. The probe uses constant replacement PBR and does not
transfer or test animation.
Version 6 also serialized the real-rig-derived asset to checksummed `.t3d` and
ran the public CLI on a Tesla T4. It recovered two joints, emitted a
659-vertex, 992-face GLB, measured mean surface distance 0.000468 and F-score
1.0 at tolerance 0.01, and passed the independent Khronos validator. Its
SkinAE weight MAE is 0.20730 versus nearest-joint baseline 0.30998. Evidence is
under `artifacts/kaggle-real-rigged-export-v6/`; the failed tolerance attempt is
preserved under `artifacts/kaggle-real-rigged-export-v5-failed/`. An independent
CPU read of the checksummed container recovers its exact two-joint parent tree
and joint positions within 3.4e-5 units.

These one-rig measurements are integration evidence, not production-quality
validation. Source PBR transfer, animation transfer, viewer deformation and
multi-asset benchmark coverage remain unverified. Earlier real GLB packer
round trips on three source rigs are recorded in `PROGRESS.md`.
