# FORGE-3D: Raw Mesh -> Polished / Textured / Rigged Asset

Status: design locked 2026-11-07. Owner: senior ML research (this repo).
Companion to `docs/PRODUCTION.md` (text/image -> t3d track). FORGE-3D is a
separate product track: **3D in, 3D out**. No text conditioning, no CLIP.

## 1. Mission

Input: any raw triangle mesh (GLB/OBJ/STL/PLY): low-poly, scanned, damaged,
untextured, non-manifold, with holes.
Output: a production asset:

1. **Polished geometry** - watertight-ish, smooth, consistent normals,
   uniform topology, detail recovered, decimated to a target budget.
2. **PBR materials** - base color, metallic, roughness assigned from shape
   context + input texture hints, exported as glTF material clusters /
   vertex colors (texture atlas upgrade later).
3. **Rig** - humanoid/creature-like assets get a 16-joint skeleton and skin
   weights; non-riggable shapes export static with an explicit flag.

Non-goals: text/image to 3D, animation synthesis, photorealistic material
reconstruction (no diffusion texture model in v1).

## 2. Why 3D -> 3D

- Supervision is free and exact: take clean assets, corrupt them
  (decimate, punch holes, jitter, flip normals, degrade texture) and train the
  model to invert the corruption. No captions, no CLIP alignment, no
  multi-view rendering loss in the loop.
- The model only has to learn **repair and refinement over the input shape**,
  a far smaller function class than text->3D. Dual T4 is enough for a useful
  model, unlike text->3D which needs GPU-months.
- The output is anchored to the user's input by construction: silhouette,
  scale, and semantics are preserved. This is what production users need.

## 3. Pipeline

```
raw mesh -> sanitize/normalize -> 64^3 occupancy + 2048 surface points
        -> GCT (geometry completion transformer) -> continuous SDF field
        -> 128^3 marching cubes -> smooth/decimate -> polished mesh
        -> MAT (material transformer) -> per-corner PBR -> glTF materials
        -> RIG (rig transformer) or UniRig -> joints + skin weights
        -> GLB (PBR [+ skin]) -> validate + invert normalization
```

Classical stages (sanitize, normalize, marching cubes, decimate, smooth,
watertight check) make the pipeline robust even before the learned stages.

## 4. Architecture: FORGE-Net

### 4.1 GCT - Geometry Completion Transformer (~18M params)

Input tensors:
- `occ64`: (1, 64, 64, 64) bit-packed solid occupancy of the raw mesh.
- `pts`: (512, 6) surface keypoints (xyz in [-0.5, 0.5]^3, unit normals)
  sampled from the raw mesh (FPS or voxel-stratified sampling).

Encoder:
- 3D conv stem: 1 -> 32 (64^3) -> 64 (32^3) -> 96 (16^3) -> 128 (8^3), GN+SiLU,
  residual blocks. Conv front-end gives locality and translation equivariance
  cheaply on T4.
- Grid tokens: 512 tokens (8^3) of width 384, 3D axial RoPE on voxel centers
  (Gram-Schmidt RoPE from `t3d.flow`).
- Point tokens: 512 tokens, Fourier(xyz) + normal -> MLP 384.
- Depth 8 full self-attention over the 1024 concat tokens + 16 learned global
  register tokens, with point<->grid cross-attention in the first 2 blocks.

Decoder:
- Token feature grid (128ch, 8^3) -> conv upsampling with skip connections from
  16^3/32^3/64^3 stem features -> feature field F=48ch at 64^3.
- Coarse head: 1x1x1 conv -> solid occupancy logits at 64^3 (auxiliary).
- SDF head: trilinear sample of F at any query point q, concat
  Fourier(q) (L=4), MLP (275 -> 256 -> 256 -> 1). Output: signed distance in
  normalized units (surface at 0). Dense 128^3 query at inference is cheap
  (~2M x 0.3 MFLOP).

Losses:
- `L_sdf`: Huber(delta=1/128) at 24-32k near-surface GT query points/asset.
- `L_norm`: 1 - cos(finite-difference normal, GT normal) on a 1/4 subset.
- `L_occ`: BCE + soft Dice on the 64^3 coarse occupancy.
- Optional eikonal `(|grad f| - 1)^2` on 1/8 subset (finite differences).

### 4.2 MAT - Material Transformer (~6M params)

- Tokens: clean surface corners at 64^3 with (coord, normal, input color hint,
  GCT feature sampled, has-texture flag).
- 6 layers alternating (a) block-local attention inside 8^3 cells with padding
  masks and (b) global attention over 512 block summary tokens; 6 heads,
  width 256; 3D RoPE.
- Head: per-corner linear RGB, metallic, roughness, alpha (6ch) + confidence.
- Training labels: per-corner PBR sampled from the source asset textures
  (`t3d.materials.sample_pbr`). Input hint: degraded colors sampled from the
  raw input texture (blurred/patch-dropped), zeroed when the input is untextured.
- Export: per-face clustering of predictions into <= 6 PBR material clusters
  (color + metallic + roughness), plus vertex colors within clusters. Texture
  atlas (xatlas) is a v2 upgrade.

### 4.3 RIG - Rig Transformer (~3M params)

- Encoder: same 512 surface tokens + GCT feature tokens.
- 16 learned joint queries cross-attend to tokens -> joint xyz, parent logits
  (16-way + root), presence logit.
- Skin head: per-corner token + joint embeddings -> 16 softmax weights.
- Training data: rigged game characters (Kenney blocky characters, Quaternius
  Ultimate Modular Characters, rigged Objaverse GLBs). GT joints from glTF
  node hierarchy; GT skin weights sampled from source vertices.
- Fallback: assets with low joint-presence confidence export static; optional
  UniRig integration for general (non-stylized) rigging is evaluated after
  the v1 model ships.

## 5. Data

Sources (all open licenses, provenance recorded per asset):

| Source | Assets (target) | Use | License |
| --- | --- | --- | --- |
| Kenney kits (local) | 1,648 static + 42 rigged | geom/mat/rig | CC0 |
| PolyHaven models | ~500 | geom/mat | CC0 |
| Objaverse LVIS subset | ~3,000 | geom/mat/rig | Sketchfab per-object |
| Thingi10K | ~3,000 | geom | CC per asset |
| Quaternius characters | optional | rig | CC0 |

Split policy: group-disjoint by source asset hash; 5% validation, 2% test;
Kenney split by kit family so no kit leaks across splits.

Self-supervised pairs: the clean asset is the target; the raw input is a random
corruption mix:
- vertex jitter (0.3-1.5% bbox), vertex-clustering decimation (x2-x8),
  random face dropout (5-25%), 1-8 spherical hole punches, normal flips,
  duplicate faces, texture blur/patch-dropout/desaturation.
- 10% identity (no corruption) so the model preserves already-clean detail.

Compact per-asset record (npz, ~300-500 KB):
- `occ64` bit-packed solid occupancy of the raw input.
- `pts` f16 (2048,6) raw surface points + normals; `cols` u8 (2048,3) raw colors.
- `sdf` i8 (N,) + `sdf_coords` u8 (N,3) + `sdf_normals` i8 (N,3), N~28k at 128^3.
- `occ_target64` bit-packed solid occupancy of the clean asset.
- `pbr_coords64` u8 (M,3), `pbr` u8 (M,6), `pbr_raw` u8 (M,3), M~15k.
- rig fields (rig subset only): joints f16 (K,3), parents i16 (K,),
  skin u8 (M,K).
- metadata: aabb f32 (2,3), normalization scale/center, flags, license/source.

Corpus packaging: tar shards of ~250 assets + `manifest.jsonl`; ~5-8k assets,
2.5-4 GB compressed, uploaded as one private Kaggle dataset.

## 6. Training on Kaggle (dual T4)

- Kernels are internet-disabled; source is embedded as a base64 zip (existing
  repo pattern), corpora and checkpoints mount as Kaggle datasets.
- DDP via `torch.distributed` with `mp.spawn`, NCCL; fallback to single GPU.
- AMP bf16/fp16, gradient clipping, cosine schedule, EMA of weights.
- GCT: ~7k train assets, batch 4 (2/GPU), 35-45 epochs / ~18-25k steps,
  ~8-10h per kernel; checkpoints every 1000 steps (~36MB fp16) + report JSON.
- MAT: ~4.5k textured assets, batch 8, 20 epochs, ~3-4h single T4.
- RIG: ~300 rigged assets, batch 8, 60 epochs, ~1-2h.
- Sessions are daisy-chained: kernel N downloads its outputs, we verify and
  publish a new version of the checkpoint dataset, kernel N+1 resumes from it.
- Compute budget: ~95 GPU-hours available across accounts; plan uses ~25-35h
  with 2x headroom.

## 7. Evaluation gates

Geometry (held-out corrupted test assets):
- SDF MAE (voxel units at 128^3), Chamfer distance and F-score@1% after
  marching cubes, normal consistency, hole count, watertight fraction.
- Must beat the raw input on CD/F-score for >= 90% of test assets.
- Visual QA: 4-view Blender renders, raw vs polished side by side.

Material: PBR MAE (linear RGB), cluster color error, visual renders.

Rig: joint position error (normalized, after Hungarian/nearest matching),
skin weight cosine vs GT on held-out characters.

Product gate for v1: pipeline runs end-to-end on arbitrary GLB/OBJ/STL,
exports semantically valid GLBs, and improves CD vs raw on the held-out set.

## 8. Milestones and persistence

- Code + reports: GitHub `Rachasumanth/transformer-native-3d`, branch
  `forge3d` (commit + tag per milestone).
- Weights + corpora: private Kaggle datasets
  `shivakrishna846/forge3d-*`; every kernel output becomes a new dataset
  version; SHA-256 of every checkpoint recorded in `MILESTONES.md`.
- Local mirror: `artifacts/forge3d/` (checkpoints, metrics, sample GLBs).
- M0 design+data, M1 GCT smoke, M2 GCT full, M3 MAT, M4 RIG, M5 pipeline+QA.

## 9. Risks and fallbacks

- T4 capacity: model sizes capped at ~20M params; conv front-end keeps token
  counts at 512, so attention stays trivial.
- Data domain skew (game assets vs arbitrary meshes): Thingi10K/Objaverse
  breadth + identity pairs mitigate; report metrics per source domain.
- SDF sign on open meshes: use closest-point normal sign; fall back to
  unsigned + double-sided reconstruction for ambiguous assets.
- Rigging generality: v1 gates rig to character-like inputs; UniRig fallback
  is an explicit stretch milestone.
- Kaggle flakiness: checkpoints every 1000 steps, resumable kernels, dual
  account fallback (profiles 4/6/7), local CPU smoke tests before every push.
