# Static GLB preparation

Install project dependencies, then run:

```sh
.venv/bin/python -m t3d.import_glb input.glb prepared/asset.t3d --grid-size 32
```

This loads self-contained static core-glTF triangle primitives via Trimesh,
instantiates scene transforms, corrects winding under reflections, restores UV
orientation, normalizes all instances together, converts geometry to the sparse
dual grid, and samples six-channel PBR attributes at closest source triangles.
The `.import.json` report records source/prepared hashes, normalization, counts
and material ranges. Add the prepared file to a provenance-bearing dataset
manifest; the importer does not invent source/license metadata.

Unsupported articulation, morph targets, required extensions, extended materials,
vertex colors, non-repeat addressing, external resources, alternate UV sets and
texture transforms fail explicitly. The sampler uses bilinear filtering without
mipmaps; native texture filter choices are not reproduced. Normal, occlusion and
emissive maps are not represented by the blueprint's six-channel PBR tuple.
Trimesh quantizes base-color factors to uint8; factor precision is consequently
limited. Scene triangulation/import remains subject to Trimesh's core importer.
High-resolution production conversion and BVH acceleration remain pending.

## Upstream asset evidence

`artifacts/import/avocado` retains the original Avocado GLB, upstream SOURCE.md,
prepared `.t3d`, import report and single-record manifest. Source:
[Khronos Avocado](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Avocado).
The upstream README lists Microsoft and CC0-1.0. At 16³ resolution, 682 source
triangles produce 498 active voxels. All prepared bytes are verified against the
manifest digest. This is an importer integration fixture, not a curated training
corpus or a train/validation split. It has not passed VLM filtering, pose network
alignment, multi-view reconstruction QA, or a generalization benchmark.

## Crossing ambiguity gate

Import now rejects multiple distinct intersections along one stored primal edge
by default. A boolean crossing flag cannot represent two separate crossings.
Use `--allow-multiple-crossings` only for a diagnostic conversion; the import
report then records the explicit override and crossing audit. Existing pilot
files predate this gate and remain retained as known-imperfect experimental data.
The gate does not reject open surfaces simply for having boundaries.

The converter's `diagnostics` dictionary reports marked edges, distinct crossing
counts, tolerance and up to sixteen example edges. Intersection parameters within
1e-8 along the same grid edge are treated as the same location, avoiding false
duplicates from adjacent triangles. `reject_multiple_crossings=True` rejects
ambiguous input before fitting/writing the voxel asset. This test is necessary
but not sufficient for topology preservation: disconnected surface patches in
one cell, tangencies, coplanar edges and floating-point degeneracies still require
additional handling. No crossings or components are deleted to force a pass.

`python -m tools.audit_conversion` reproduces the 16³ source audit. Original
exactly welded sources have 38/264 boundary edges and zero nonmanifold edges
(Avocado/WaterBottle). Prepared extraction has 28/112 nonmanifold edges, while
the new audit detects 3/84 multiple-crossing primal edges. These counts establish
representation ambiguity, not a one-to-one attribution of every topology defect.
