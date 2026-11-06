# Rectified-flow sparse transformer foundations

`t3d.flow` implements the linear rectified-flow path
`x(t)=(1-t)*noise+t*clean`, targeting velocity `clean-noise` with MSE. Inference
uses Euler steps from t=0 toward 1 and classifier-free guidance
`v_unconditional + guidance*(v_conditional-v_unconditional)`. Training can drop
prompt conditioning while retaining aligned clean shape conditioning.

`SparseFlowDiT` uses sinusoidal timestep features, adaLN-Zero gates, bounded
serialized-patch self-attention, prompt-token cross-attention and a pointwise FFN.
Each attention head learns two 3D vectors (six scalars), orthonormalizes them via
Gram-Schmidt with finite degenerate-vector fallbacks, and applies axial rotary
embeddings over the rotated coordinates. Head dimensions must be divisible by 6.
The output projection starts at zero. Clean shape latents may be added through a
separate projection for a future shape-conditioned material model.

One asset is processed per forward call. Pass same-device integer permutations
in `orders` to cycle Z/TZ/H/TH across layers; default identity assumes the caller
has already ordered rows spatially. Each self-attention patch is bounded by
`patch_size`; gathering patch tensors still copies data as required by PyTorch.
This is not a claim of zero physical reordering cost or production GPU throughput.
Prompt cross-attention sees supplied condition tokens, not an internally encoded
image/text string. Sparse coordinate layout is supplied by the caller.

These foundations do not implement the complete three-stage generative system.
Still pending: coarse occupancy training, separate shape/skeleton support and
adapters, DINO/CLIP encoders and dataset alignment, trained geometry/rig/material
flows, rendering losses, end-to-end synthesis, and research-scale validation.
Synthetic neural tests establish arithmetic/gradient execution only. Poor VAE
reconstruction evidence from earlier pilot runs remains unresolved.
