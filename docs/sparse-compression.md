# Step 4 sparse compression foundations

PyTorch modules run in the Kaggle notebook; the laptop runs CPU preprocessing.

`t3d.sparse_compression` uses integer coordinates `(batch,x,y,z)` and floating
features `(N,C)` on the same device. Octants are `4*x_bit+2*y_bit+z_bit`.
Downsampling gathers eight children with zero-filled missing slots, stacks them
in octant/channel order, and averages contiguous groups into output channels.
Upsampling duplicates channel groups, unpacks octants, and retains only occupied
children. Teacher masks align to lexicographically sorted parent coordinates;
decoder coordinates are sorted again at every level. Four stages reduce spatial
resolution by 16. Shortcuts are intentionally lossy when channel averaging occurs.

`SparseDownsample` adds a learned linear residual to the shortcut.
`SparseUpsample` adds a learned linear residual and predicts eight occupancy
logits per parent. Teacher masks are accepted for reconstruction training;
predicted masks threshold sigmoid probabilities during generation. Binary mask
selection is discrete, so occupancy heads need separate supervised losses.

`t3d.sparse_blocks.SubmanifoldConv3d` computes one 3×3×3 spatial convolution at
active voxels only, treating absent neighbors as zero. It builds batch-aware
integer keys and uses sorted searches for neighbor rows. `SparseConvNeXtBlock`
adds LayerNorm and a wide pointwise GELU MLP with learned residual scaling.
These are differentiable reference implementations without custom Triton kernels.
Repeated coordinate validation/search and Python offset loops are expensive;
production-scale memory and runtime have not been established.

The full SC-VAE still needs the composed encoder/decoder, posterior moments and
sampling, KL loss, decoupled shape/material training, attribute reconstruction
and rendering losses, checkpoint/resume, and real-asset dataset training. SkinAE
and generative flow models are also pending. These modules do not establish any
compression-quality or throughput target from the research blueprint.

## Composed variational model

`t3d.sc_vae.SparseCompressionVAE` composes four encoder/downsample stages and
four decoder/upsample stages with ConvNeXt blocks. A learned posterior predicts
mean/log-variance on the 16× coarser support, and samples by reparameterization.
Decoder support uses teacher occupancy during reconstruction training or predicted
masks during inference. There is no geometry-feature skip around the bottleneck.
Shape and material instances have independent parameters.

Shape input/output channels are local dual xyz, three edge flags, and split
weight (7). Tensor losses are sigmoid dual-coordinate MSE, edge-logit BCE, and
split-weight MSE. Material channels are linear RGB, metallic, roughness and alpha
(6), trained with sigmoid-output L1. Both add four-level occupancy BCE and latent
KL. Output tensors from `decode` are raw; sigmoid is applied by the relevant
attribute loss. Rendered depth/normal/color losses are not yet implemented.

`tools.train_compression_probe` runs ten deterministic posterior-mean updates
per model on the cube fixture in Kaggle, recording attribute/occupancy/KL terms
and model/optimizer checkpoints. This verifies execution only. The tiny model,
single analytic asset, constant material, and short run cannot support research
quality or dataset generalization claims. Checkpointing here records the state;
a production resume/data-loader training loop is still pending.
