# Multi-view curation renders

Use Blender with GLB input to produce transparent color PNGs and floating-point
OpenEXR depth and normal passes from a set of azimuths and fields of view:

```sh
blender -b --factory-startup --python-exit-code 1 \
  --python tools/render_curation_views.py -- \
  --input /path/to/asset.glb --output /path/to/render-output \
  --resolution 128 --azimuths 0 90 180 270 --fovs 35 55
```

The manifest records the input hash, bounding box, Blender version, camera
positions and output hashes. The camera fits a bounding sphere, so individual
objects may occupy less of the frame than an object-specific fit. EXR files
preserve Blender render-pass values, including the large background depth
sentinel; transparent color alpha can identify background for downstream
processing. The normal pass should be interpreted using the Blender version's
render-pass conventions. The default lighting and eight samples are suitable
for a pipeline probe, not a validated aesthetic-scoring setup. This renderer
does not perform VLM filtering, pose canonicalization, PBR verification or
license checks. Only Blender 5.2.2 has been exercised so far.

`tools/score_curation_views.py` can score the color renders against JSON prompt
categories using a pinned CLIP model. Its optional `--crop-foreground` crops to
the alpha bounds with a margin and pads to a square before CLIP preprocessing.
The scorer reports cosine similarities only; it does not classify or delete
assets. A four-example initial probe with uncropped views misclassified a cube
and a sphere by the highest score, so it is not an established quality filter.
