# Three-stage latent flow orchestration

`t3d.generation.ThreeStageFlow` composes three independently parameterized sparse
DiTs. Stage 1 flows a scalar occupancy value at every cell of a bounded coarse
grid; thresholding selects the active support. Stage 2 generates shape and optional
co-located rig latents at those coordinates. Stage 3 generates material latents
conditioned on the generated clean shape and the same prompt tokens. Empty
occupancy returns explicitly empty streams without calling sparse models on an
invalid empty token set. The pipeline returns tensors, not a decoded GLB.

Training `losses` builds dense coarse occupancy from source latent coordinates,
then returns separate occupancy, geometry and material rectified-flow losses.
Material training uses clean source shape latents; generation uses Stage 2 output.
All stages accept external conditioning tokens and classifier-free dropout.
`t3d.conditioning.PromptConditioner` optionally loads frozen Hugging Face
DINOv2 image and CLIP text backbones at explicit revisions and projects their
token sequences to a shared width. Transformers and the pretrained weights are
optional dependencies. Its projections and all flow weights require training;
one pretrained-encoder integration probe does not establish aligned 3D output.
The geometry/material stages share sparse coordinate support. Optional rig
channels belong to the geometry stream. With `separate_skeleton=True`, Stage 1
has shape and skeleton streams with zero-initialized bidirectional adapters at
each block; training accepts `skeleton_coords` and returns a separate skeleton
occupancy loss, and generation returns `skeleton_coords`.

The default `skeleton_containment="aabb"` limits skeleton voxels to the shape
support's axis-aligned bounding box. `skeleton_containment="enclosed"` uses
6-connected flood fill on a complete grid to retain surface and enclosed cells;
open shells leak, and the operation rejects nonconvergent fills. Both are
coarse voxel constraints, not guarantees that generated bone segments lie
inside the decoded continuous mesh. The optional branch and both masks are
research foundations, not trained or benchmarked production components.

Z/TZ/H/TH permutations are computed once per stage and reused across flow steps.
The integer mapping implementation is CPU-based, then transferred to the model
device; no production-throughput claim is made. One asset is processed per call.
The complete coarse grid is capped before allocation (default 262144 tokens,
matching 64³); attention is bounded in local patches but the occupancy scaffold
is still dense. Default 50 Euler steps and CFG 3 follow the plan's reference
settings, not measured quality-optimal settings for this untrained pipeline.

Known-velocity tests check stage ordering and shape-to-material conditioning;
neural tests check finite separate losses and gradients. These tests do not
establish generated asset fidelity. Still required are image/text-aligned data and trained projections, real-data validation of the coupled occupancy
branches and adapters, trained stage checkpoints, latent normalization/provenance, decoder
integration, rendering supervision and research-scale quality validation.
The earlier VAE reconstruction and source topology failures remain unresolved.
