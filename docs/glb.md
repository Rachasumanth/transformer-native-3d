# Baked PBR GLB packaging

`t3d.glb.pack_glb` produces binary glTF from supplied vertices, faces, UVs,
normals and a floating H×W×6 texture (linear RGB, metallic, roughness, alpha).
It embeds both PNG textures and geometry buffers, converts linear RGB to sRGB,
preserves linear alpha, and packs roughness in G and metallic in B. Material
factors are unity. OPAQUE, MASK and BLEND modes are explicit; MASK accepts an
alpha cutoff. Normal vectors are normalized and included in the GLB.

The rasterizer UV convention is flipped vertically. Default z-up input maps
(x,y,z) to (x,z,-y); callers with y-up input must set `source_up='y'`.
The function validates shapes, finite values, index bounds and channel ranges.
It returns bytes for the caller to save. It does not assign textures from raw
voxel values: GPU rasterization, BVH reprojection and sparse interpolation must
supply the baked texture. Optional low-level rig attachment accepts a validated
`Skeleton` and normalized `(V,J)` weights and writes a glTF joint hierarchy,
inverse-bind matrices, and up to 16 influences per vertex. It does not infer a
skeleton or weights. `limit_skin_influences` performs the lossy top-k projection
explicitly, and `rewrite_skin_weights` rewrites the `JOINTS_n`/`WEIGHTS_n`
accessors of an existing GLB in place without changing geometry, materials,
animation or the joint hierarchy.

The local GLB tests check the binary header, embedded resources, exported normal
attribute, coordinate and UV mapping, color conversion, metallic/roughness channel
packing, transparency, skin attributes, buffer-view targets and invalid-payload
rejection. Four tests pass locally; trimesh emits a SciPy fallback warning on
mesh reload in this minimal local venv. Actual source-rig round trips and
Khronos validator results are recorded in `PROGRESS.md`. This is packaging
evidence, not rendered fidelity or end-to-end GPU export.
