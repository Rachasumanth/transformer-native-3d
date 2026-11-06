# Skeleton recovery from voxel articulation fields

`t3d.articulation.infer_skeleton_from_field(points, field, tolerance=...)`
recovers a joint forest from the six-channel representation stored in a raw
Track A asset. `points` must be the corresponding voxel-center world positions.
The first displacement triple locates a voxel's nearest joint; the second
locates that joint's parent, with roots represented by a self-parent endpoint.

The decoder clusters all inferred endpoints using a spatial tolerance grid and
radius checks between neighboring cells. It builds parent links from endpoint
pairs and selects the most frequent parent observation for each joint. Joint
output order is lexicographic by position for deterministic results. It rejects
cycles and rigs above `max_joints`; callers must choose a tolerance that reflects
field noise and the minimum intended joint separation. Clustering cannot
reliably distinguish two joints closer than the selected tolerance.

Focused synthetic tests recover an exact three-joint chain, a perturbed chain,
and a float16-quantized field. Raw profile 3 stores feature values as float16,
so the default tolerance of one quarter of the smallest voxel dimension is
appropriate for the measured quantization. An explicit `1e-6` tolerance split
the real two-joint RiggedSimple skeleton into ten clusters; the corrected default
recovered its exact parent hierarchy with maximum position error `3.4e-5`.

The CUDA `export_raw` path connects this decoder to sparse feature sampling and
frozen SkinAE weight decoding. A checksummed RiggedSimple-derived `.t3d` asset
passed the public Kaggle T4 CLI path and reloaded as a two-joint GLB. This is one
source rig and does not establish generated-field quality or pose deformation
accuracy. The held-out SkinAE weight MAE is about 0.207 on RiggedSimple.

`load_rig_glb(path, include_pbr=True)` retains per-primitive glTF base-color,
metallic, roughness and alpha factors, plus embedded base-color and
metallic-roughness textures using untransformed `TEXCOORD_0` and repeat/clamp
samplers. The image decoder supports self-contained GLB bufferViews and data
URIs. Nonzero emissive, normal/occlusion maps, material extensions, mirrored
repeat and separate wrap modes are rejected because the six-channel schema
cannot represent them. Real CesiumMan and Fox source GLBs decode their embedded
base-color maps and produce nonconstant six-channel voxel materials in local
CPU probes; source texture export is being validated separately on Kaggle.
