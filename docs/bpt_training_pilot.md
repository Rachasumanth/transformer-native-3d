# Asset-disjoint BPT training pilot

`tools/build_bpt_training_pilot.py` prepares a small integration set with an
explicit 5% face-deletion rejection gate at 128 bins. Faces that degenerate under
any proper cube rotation are removed, and every retained oriented quantized face
is checked through BPT round trip. No source mesh is overwritten. This gate
controls face loss; it does not prove surface fidelity or manifold topology.

| Source | Split | Removed faces | Status |
| --- | --- | --- | --- |
| Duck | Train | 2.75% | Accepted |
| Food Lime | Train | 1.86% | Accepted |
| Avocado | Validation | 0.15% | Accepted |
| Box | Simple held-out control | 0% | Accepted |
| Cesium Milk Truck | Candidate | — | Rejected: animated input |

The manifest at `artifacts/bpt-training-pilot/manifest.json` retains original
source hashes, licenses, retained face indices, artifact hashes, counts and
decoded edge topology. Each accepted source has four Y quarter-turn variants,
with matching 4,096-point surface conditions. Splits are assigned by source
before augmentation; no source or rotation crosses splits. Duck uses the SCEA
Shared Source License 1.0, Box CC-BY-4.0, and Avocado/Food Lime CC0. Preserved
notices, Duck's full license and source documentation accompany the private data.

`tools/train_bpt_multiasset.py` trains uniformly across the eight training
streams for 600 steps. It samples 512-token crops: 25% first crops, 25% final
crops, 50% random crops. Original absolute stream positions are retained via
the model's `position_offset`; crop boundaries introduce no synthetic BOS/EOS.
Evaluation covers each target once in windows at the same absolute positions.
This removes preceding-window history and is not a full-history likelihood.
Greedy generation uses the full KV history, which differs from crop training.
Both termination and independent decode validity must be assessed.

The private Kaggle dataset is `sumu143/transformer-native-3d-bpt-pilot`; the
GPU notebook is `sumu143/transformer-native-3d-bpt-multiasset`. Data are mounted
separately from the embedded code and verified against manifest hashes.

```sh
.venv/bin/python -m tools.build_bpt_training_pilot
.venv/bin/python -m tools.build_bpt_multiasset_notebook
kaggle kernels push -p kaggle-bpt-multiasset --accelerator NvidiaTeslaT4 --timeout 1200
```

Two training sources and one real validation asset remain far below the plan's
curated 300K–500K-asset corpus. This pilot tests data ingestion and multiasset
learning; it cannot establish representative generation quality.

A separate Duck source comparison used 1,024 area-weighted samples in each
direction at seeds 0, 1 and 2. Mean bidirectional distance was
0.001290/0.001320/0.001291, with F-score 1.0 at the loose 0.01 tolerance.
The quantized decoded mesh still has 6 boundary edges and 4 nonmanifold edges.
These topology defects preclude calling the data production-quality artist
meshes. The report is `artifacts/bpt-training-pilot-source-quality.json`.
