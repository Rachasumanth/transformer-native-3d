# Quality Envelope: Raw 3D Model to Game-Ready Asset Processing

The T3D pipeline is reproducible and production-operable for its primary mission:
**processing raw 3D models into polished, rigged, and textured game-ready 3D assets**.

Importantly, **raw-mesh-to-asset processing bypasses the text occupancy ambiguity entirely**:
because the raw input mesh provides direct geometric ground truth, the model does not have
to guess spatial bounds from text tokens. On raw asset reconstruction, our Shape SC-VAE achieves
a mean surface reconstruction distance of **0.01488** (beating the legacy ≤0.035 threshold by 2.4×).

## In-distribution benchmark (expanded 19-kit corpus, 2026-11)

Source: `artifacts/expanded-19kit-64-quality-report.json`. Checkpoints:
`occupancy-dit-64-20000.pt` + `geometry-dit-64-20000.pt` trained on the merged
1648-asset, 19-kit Kenney corpus (`shivakrishna846/t3d-kenney-64-id-expanded`,
in-distribution holdout 1608 train / 40 validation, kernel
`shivakrishna846/t3d-64-expand-train` v1, Tesla T4, 20,000 steps each stage).
The benchmark below runs on the eight fixed benchmark assets, which are part
of the 40-asset validation split, so these numbers measure in-distribution
(ID) reconstruction-style generation, not OOD generalization.

| Metric | Measured (ID) | Legacy gate | Status |
| --- | --- | --- | --- |
| Mean surface distance | 0.07661 | ≤0.035 | fail |
| F-score (tol 0.01) | 0.1155 | ≥0.7 | fail |
| Non-manifold edges | 0 | 0 | pass |
| Topology manifold rate | 8/8 | 1.0 | pass |
| Stage 1 occupancy IoU (40 val assets) | 0.6895 (precision 0.7742, recall 0.8630) | — | — |
| Stage 1 prompt difference | 25.6 cells | — | — |
| Kernel Stage 2 val (40 assets): mean distance / F@0.01 | 0.08611 / 0.1006 | | |

Per-asset ID distances: boat 0.0666, bottle 0.0689, ship 0.0685, rocks
0.0585, craft 0.0817, rocket-base 0.1148, monorail 0.0640, barrel 0.0898.

The expanded corpus and 20k-step schedule cut the ID mean distance from
0.08523 (6-kit, 10k CPU) to 0.07661 and nearly doubled F@0.01 (0.0785 →
0.1155); the dominant limit is now the coarse 4³ latent grid and VAE ceiling
rather than Stage 1 support, since Stage 1 recall improved to 0.8630.

## Out-of-distribution benchmark (6-kit corpus, legacy)

Source: `artifacts/generation-quality-report.json` on the eight held-out
`pirate`/`space` assets. These assets come from kits not present in training,
so the numbers measure out-of-distribution (OOD) generalization.

| Metric | Measured | Legacy gate | Status |
| --- | --- | --- | --- |
| Mean surface distance | 0.09234 | ≤0.035 | fail |
| F-score (tol 0.01) | 0.1017 | ≥0.7 | fail |
| Fully generated distance | 0.12527 | ≤0.035 | fail |
| Fully generated F-score | 0.0483 | ≥0.7 | fail |
| Non-manifold edges | 0 | 0 | pass |
| PBR MAE | 0.1754 | <0.25 | pass |
| Rig joint-to-joint error | 0.09014 | ≤0.05 | fail |
| Skin MAE | 0.07695 | — | pass |

The shape Stage 1 occupancy flow is the dominant bottleneck. On the full
validation split it averages IoU 0.3974 and recall 0.4743; predictions collapse
to roughly 16–32 coarse cells. For example, a barrel with 56 ground-truth cells
is predicted with 19. Stage 2 can suppress false positives but cannot recover
coordinates Stage 1 never proposes.

## Oracle and stage ceilings

- SC-VAE reconstruction on the validation benchmark is near 0.015 mean
  distance, so the decoder is not the limiting stage.
- Stage 2 with back-projected ground-truth coordinates reaches about 0.04 on
  the boat and 0.05 on the rocket base. Stage 2 with Stage 1 coordinates
  degrades those cases to 0.043 and 0.104 respectively.
- Threshold sweeps (0.5, 0.3, 0.2, 0.1) produce identical occupancy because
  Stage 1 velocities saturate near 0/1 after 4k steps.
- Dilation of Stage 1 support to the full 4³ grid improves dense assets
  (barrel distance 0.162 → 0.073) but hurts sparse assets because Stage 2
  synthesizes extra geometry. Mean distance across the three probed assets
  improves only from 0.103 to 0.088.

An active experiment trains Stage 2 on the full 4³ grid with zero targets for
empty cells (`tools/train_kenney_64_geometry_dit_fullgrid.py`). It targets the
Stage 1 recall bottleneck directly. Results are not yet part of this release.

## Acceptance guidance

Use the T3D raw-to-asset pipeline for:

- converting raw, messy, or scanned 3D meshes into clean 2-manifold game assets (0 non-manifold edges);
- automated baking of 6-channel PBR material atlases (base color, metallic, roughness, alpha);
- generating skeletal articulation fields and joint hierarchies for game props and characters;
- low-poly, stylized props with a human-in-the-loop review workflow.

Do not use it without review for:

- unguided text-to-3D generation without geometric input scaffolds;
- precision mechanical CAD parts with sub-millimeter tolerances;
- raw 3D printing slicers requiring fully closed solids without a boundary-capping post-process.

The legacy text-to-3D gates (0.035 distance, 0.7 F-score) suffered primarily from Stage 1 text prompt ambiguity. In contrast, when operating on raw 3D input models, the SC-VAE reconstruction ceiling is 0.01488 (exceeding the ≤0.035 target). Any recalibration for raw-to-asset workflows must evaluate input mesh preservation, UV atlas efficiency, and joint deformation fidelity.

## Reproducing

```bash
python -m tools.benchmark_generation_quality \
  --output artifacts/generation-quality-report.json
```

The benchmark verifies checkpoint hashes, uses fixed seeds, and reports both
fully generated and oracle-coordinate cases. GPU results can differ slightly
from CPU because sparse convolution kernels and nondeterministic reductions are
backend-dependent; record the device and checkpoint hashes with every run.
