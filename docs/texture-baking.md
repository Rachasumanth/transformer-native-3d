# GPU sparse PBR texture baking

`t3d.texture_baking.bake_pbr` connects prepared UV geometry to raw sparse
six-channel PBR. It rasterizes UV triangles in 100K-face chunks with nvdiffrast,
interpolates world-space texel positions, reprojects those positions onto the
reference CuMesh BVH, and samples sparse materials using FlexGEMM trilinear
interpolation in 100K-query chunks. A seventh constant support channel detects
missing neighborhoods instead of silently baking black. Float32 Telea inpainting
fills uncovered UV texels per channel, with the resulting values clipped to [0,1].
The returned image is linear RGB/metal/rough/alpha for `t3d.glb.pack_glb`, together
with the pre-inpainting raster mask.

`PreparedMesh` now retains the exact reference vertex/face arrays alongside its
BVH so barycentric projection uses the matching geometry. Sampling follows the
upstream exporter coordinate convention `(point-aabb_min)/voxel_size` without
adding a half-cell offset. This convention and missing-support behavior need
explicit integration probes on nonconstant textures and sharp boundaries.

Dependencies/API references:
[TRELLIS.2 export](https://github.com/microsoft/TRELLIS.2/blob/main/o-voxel/o_voxel/postprocess.py),
[FlexGEMM sampling](https://github.com/JeffreyXiang/FlexGEMM/tree/main/flex_gemm/ops/grid_sample),
[nvdiffrast](https://github.com/NVlabs/nvdiffrast).

This implementation is pending GPU execution. The current Kaggle version 15
predates it and builds/probes CuMesh only. Installing the pinned remaining
dependencies and an end-to-end baking/GLB check are required next. No rendered
material fidelity, UV seam quality or production export claim is established.

Upstream kernel inspection establishes that FlexGEMM locates samples at
`integer_coord + 0.5`, using `floor(query - 0.5)` for neighbor selection. Thus
world-to-grid queries stay `(world-aabb_min)/voxel_size` with no *additional*
caller offset. The pending independent affine oracle now expects
`(query - 0.5)/3` when stored dense features are `integer_coord/3` and queries lie
inside the sample-center bounds. The earlier description of samples at integer
coordinates was inaccurate; cell centers are the correct convention.

Atlas padding now lives in CPU-only `t3d.texture_padding.pad_pbr`. A six-channel
constant-atlas regression exposed strong Telea drift for float32 [0,1] inputs in
local OpenCV 5.0.0 (some gaps clipped to zero/one). Byte-domain inpainting alone
still drifted by several byte values. The padding stage now runs Telea on byte
maps, bounds each filled channel by its covered source extrema, and restores
covered texels exactly. This preserves constant channels and excludes invented
material extremes. Two local tests pass for six distinct constant channel values,
input immutability, covered-texel preservation, full coverage and empty-mask
rejection. This change is newer than queued Kaggle version 16. The next export
probe checks the entire atlas as well as the original covered pixels.

The pending export probe now also uses a planar UV atlas with analytic RGB
varying with world x/y, so expected values can be checked independently at texel
centers. `raster_chunk_size` defaults to 100,000 and may be reduced to force
multiple chunks: the probe compares one triangle per chunk with a single chunk
to catch incorrect global triangle-ID offsets. These gradient/chunk checks are
syntax-checked but have not yet run on CUDA.
