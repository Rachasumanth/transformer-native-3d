# VAE reconstruction and surface measurements

`t3d.reconstruction.reconstruct_asset` encodes a supplied raw asset with the
shape VAE and decodes its posterior mean. `support='predicted'` uses decoder
occupancy logits; `support='teacher'` explicitly supplies source masks as a
diagnostic. Neither mode is generative synthesis. Dual offsets are sigmoid
outputs, edge flags threshold sigmoid probabilities, and split weights are raw.
Generated cells outside the source grid are excluded. Optional material decoding
uses its own posterior and follows the reconstructed shape's support; it does not
copy source PBR attributes. Rig decoding is still unsupported and fails explicitly.

`t3d.metrics.surface_comparison` samples triangle surfaces by area and computes
bidirectional point-to-triangle distances in input world units. `mean_distance`
is the mean of the two directional unsquared distance means. F-score uses a
stated tolerance, sample count and RNG seed. Empty surfaces, degenerate triangles
or nonfinite area yield a failed measurement with a reason. Boundary/nonmanifold
incidence is computed after welding exactly coincident vertex positions, so GLB
UV seam duplicates do not appear as open topology. Total boundary edge length is
reported alongside edge count; winding has not been repaired.

These definitions are deliberately explicit: they do not reproduce the
blueprint's named Mesh Distance, Dora rendered Sharp Normal Error, or material
relighting protocols. Those suites still need their exact reference conventions.
The pilot compares reconstruction against mesh extraction from the prepared
source `.t3d`, which isolates VAE error but does not measure original-GLB conversion
error. Original-surface fidelity, rendered supervision and benchmark suites remain
pending. Saved OBJ outputs carry geometry only; `.t3d` carries decoded PBR.

Version 8 measured predicted-support pilot reconstructions at F-score zero
(tolerance 0.01), with only 6 train and 8 validation triangles. These outputs are
not usable-quality reconstructions. `metrics.valid` only indicates that the
measurement could be computed. All four reconstruction artifacts and reports
are retained under `artifacts/kaggle-v8`; no green unit-test result overrides this
failed geometry-quality evidence.
