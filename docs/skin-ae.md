# SkinAE reference foundation

`t3d.skin_ae.SkinAE` implements a joint-count-independent encoder/decoder
foundation based on [AniGen section 3.3.1](https://arxiv.org/html/2604.08746v2#S3.SS3.SSS1).
Parent-relative bone vectors and parent-edge features receive Fourier encodings,
then a Transformer produces four-channel joint embeddings. Ground-truth skin
weights average these into four-channel vertex embeddings. Separate MLPs lift
joint/vertex embeddings to 64 channels. Their dot products, divided by a positive
learned per-vertex temperature, produce logits over the current joint set.
Softmax produces normalized skin weights; `skin_reconstruction_loss` implements
mean per-vertex KL(target || predicted), including zero target weights.

Widths, Fourier frequencies, parent-edge feature construction and temperature
floor are explicit local choices. This module does not claim compatibility with
upstream checkpoints. It processes one skeleton at a time and accepts int64
parents, with -1 for roots. Cyclic parent graphs are rejected. It uses no
joint-index positional embeddings, so consistent joint relabeling permutes the
outputs. `freeze()` disables parameter gradients and switches to evaluation mode;
call it on a validated pretrained checkpoint before generator training.

Kaggle notebook `sumu143/transformer-native-3d-skinae-check`, version 1, ran three
focused tests on Tesla T4, all passing without skips in 1.774 seconds. Evidence
in `artifacts/kaggle-skin-v1` matches bundle SHA256
`98932ec91a5cebc6c877f5024565e16ab9e888dd4a2cd167f61f947237ebe4f9`.
This covers gradients, variable joint counts, joint relabeling, normalized output,
weighted-average encoding, freezing and malformed input. At that version-1
stage, there was no rigged training corpus, learned reconstruction result,
sparse skeleton/skin autoencoder or GLB attachment. Later pilot training and
animation data are documented below. The separate CPU skeleton-confidence
target is not yet integrated into a learned skeleton regression branch.

## Animated-pose ingestion and optimization pilot

`t3d.import_rig.load_rig_glb` samples core glTF animation channels for
translation, rotation, and scale. STEP, LINEAR (including quaternion slerp), and
CUBICSPLINE tracks are supported following the
[Khronos glTF 2.0 animation sampler rules](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#animations).
Morph-weight animation, animated matrix nodes,
multiple skins, and non-core required extensions remain unsupported. The
animation sampler updates joint world transforms and evaluates the skinned
surface with source inverse-bind matrices. Each asset's normalization is shared
across all sampled frames.

`tools.build_skin_pose_pilot` records 1024 fixed surface samples and barycentric
skin-weight targets at five evenly spaced times per animation. RiggedSimple and
CesiumMan are training assets; all Fox clips remain in held-out validation. The
set contains 10 training pose sets and 15 validation pose sets, from only two
training assets and one validation asset.

Kaggle Tesla T4 version 2 trained the earlier node-default pilot for 400 updates.
RiggedSimple and CesiumMan KL fell to 0.0104 and 0.0334; held-out Fox reached
1.6419 KL versus a 2.6113 uniform baseline, with 0.5947 dominant-joint accuracy.
This checkpoint is an integration result and is not validated for production.

`tools.audit_rig_animations` sampled five frames per clip across RiggedSimple,
CesiumMan and Fox's Survey/Walk/Run clips. Maximum normalized vertex motion from
each clip's first sampled pose was 0.291, 0.554, 0.266, 0.271, and 0.571
respectively; edge incidence remained constant across each clip's frames. The
report is `artifacts/rigged-inputs/animation-audit.json`. This verifies parser
and skinning execution on real animation data, not independent reference
deformations or the plan's J2J/J2B/B2B and smoothness quality targets.

The multi-pose dataset and training code are bundled into private Kaggle notebook
`sumu143/transformer-native-3d-skinae-multi-pose-pilot`, Tesla T4, version 1.
Its source bundle SHA256 is
`d25ec5991dcf3a3d7dbde1743c0daf2e63c6c47142e1ecd70026c34cdbe20150`; the
downloaded report's manifest hash matches the local pose-pilot manifest. All 8
included importer/SkinAE tests passed on the GPU notebook. Over 1200 updates,
best checkpoint selection occurred at update 500. Across Fox's 15 held-out
poses, mean KL fell from 2.6204 to 1.0551 against a uniform baseline of 2.6297;
mean absolute weight error was 0.0369 and dominant-joint accuracy 0.7291. The
checkpoint is `artifacts/kaggle-skin-pose-v1/skin-pose-pilot.pt` with SHA256
`f73a7be776b6ac653f2e248d78307eeddea833ea0c5284011e7a92bf2543e9eb`. This is
still a two-training-asset/one-validation-asset experiment; no pose-quality
benchmark or independent weight/deformation reference has been completed.

To check sensitivity to the chosen validation asset, another Kaggle Tesla T4
notebook trained three folds, holding each source asset out once. All three
held-out KL scores beat their per-fold uniform baselines:

| Held-out asset | KL | Uniform KL | Weight MAE | Dominant-joint accuracy |
| --- | ---: | ---: | ---: | ---: |
| CesiumMan | 0.5558 | 2.3190 | 0.0392 | 0.8420 |
| Fox | 1.0551 | 2.6297 | 0.0369 | 0.7291 |
| RiggedSimple | 0.1498 | 0.3183 | 0.2056 | 0.9063 |

Mean held-out KL was 0.5869 versus mean uniform KL 1.7557. RiggedSimple's
weight MAE remains high despite its KL and dominant-joint accuracy. All 8
packaged tests passed; source splits, pose counts, manifest identity, and
checkpoint hashes were independently checked. Bundle SHA256 is
`111507df606fa4e2b475ee69daa29879d4ea840b98089e9b2bf039aa8d4cbf3f`; the
audit is `artifacts/kaggle-skin-pose-cv-v1/independent-audit.json`. Three
sources are insufficient to establish general SkinAE quality, and weight
reconstruction does not measure animated deformation fidelity.

## Export-time decoder probe

The retained multi-pose checkpoint was loaded on a Kaggle Tesla T4 and used for
synthetic export-time inference. A generated six-channel field recovered its
three-joint hierarchy; 512 sparse voxel features were sampled and decoded to
normalized weights. An affine feature field interpolated with zero error, and
missing sparse support was reported. Five export/SkinAE tests passed. The
source bundle and output report are in `artifacts/kaggle-skin-export-v3/`.
This is a code-path execution check only: the fixture was synthetic, and the
test did not assess generated field fidelity or run the full geometry-to-GLB
export pipeline.

## Influence limiting

`t3d.glb.rewrite_skin_weights` and the export-time `max_skin_influences` option can
project SkinAE output onto its strongest influences. On the CesiumMan-to-v13
correspondence audit, dense SkinAE weights give 0.0179 mean canonical distance
and F-score 0.60; top-2 gives 0.0109 and 0.78, top-3 gives 0.0139 and 0.67, and
top-4 gives 0.0161 and 0.61. Top-1 reaches 0.0098 and 0.84 but its animated
reload contains a zero-area triangle, so it is not safe. The oracle
source-transfer control remains 0.00052 and 1.0. This is a partial mitigation
on one reconstructed export, not a substitute for better weights or rendered
quality validation; the default export limit remains 16.
