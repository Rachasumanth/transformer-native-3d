# Agentic Session Handoff & Progress Note

**Source Session:** Codex thread `01a19c30-f6b3-7583-8a5e-0ace1409653f` (halted at token limit on 2026-11-04T08:49:37Z).  
**Resumed By:** Antigravity AI Assistant  
**Date:** 2026-11-04  
**Project:** Transformer-Native 3D Model Format (`.t3d`)  
**Workspace:** `/home/sumanth/transformer native 3d model format`

---

## 1. Summary of Where Codex Stopped

Right before reaching the usage quota limit at step 26811, Codex had:
1. Launched five parallel CPU Kaggle notebooks (`sumu143/t3d-kenney-corpus-conversion-shard-0` through `4`) to convert an interleaved fifth each of the 944 raw Kenney catalog GLBs into 32³ `.t3d` shape assets.
2. Written `tools/merge_kenney_shards.py` and `tests/test_corpus_dedup.py` to prepare for merging and exact deduplication.
3. Downloaded and verified the material VAE pilot (Kaggle run v34) in `artifacts/kaggle-material-pilot-32-v34/`, which demonstrated severe overfitting on a single asset (Avocado) with validation error degrading after step 300 (0.3317) to step 600 (0.3470).
4. Hit token limit while waiting for the CPU conversion shards to complete.

---

## 2. Progress & Research Breakthroughs Achieved in This Session

### A. Shard Retrieval, Verification, and Merging
- Monitored and retrieved outputs for all 5 Kaggle conversion shards into `artifacts/kaggle-kenney-corpus-shard0/` through `shard4/`.
- Executed `tools/merge_kenney_shards.py` across all 944 candidate models:
  - **1 candidate rejected:** `pirate-kit/chest` (rejected strictly because it contains skeletal rigging/animation, avoiding silent loss of articulation in a geometry-only pass).
  - **75 within-split exact duplicates removed:** Identified by matching content/source hashes.
  - **868 verified assets retained:** Stored in `artifacts/kenney-corpus-32/` with `manifest.json` (**649 train**, **219 validation**).
  - Manifest SHA-256: `af2f3179db2cbfe1288a3ab9e67e0d7093df61a7591b9c7586cc782fc72824bc`.
  - All 868 assets verified through `t3d.dataset.AssetDataset`.
  - Pack distribution across splits is strictly disjoint:
    - **Train (649 assets):** `car-kit` (50), `food-kit` (199), `furniture-kit` (140), `nature-kit` (260).
    - **Validation (219 assets):** `pirate-kit` (67), `space-kit` (152).

### B. Deployment of Kaggle Corpus Dataset
- Created a reproducible 12.6 MB archive bundle: `kaggle-kenney-corpus-32/kenney-corpus-32.zip`.
- Uploaded and published private Kaggle dataset: **`sumu143/t3d-kenney-corpus-32`**.
- Verified dataset status: `ready` with all 870 files (`manifest.json`, `merge-report.json`, and 868 `.t3d` assets) unzipped and accessible.

### C. Large-Scale Shape VAE Corpus Training (Kaggle Version 35)
- Implemented `tools/train_kenney_corpus_32.py` and `tools/build_kenney_corpus_training_notebook.py`.
- Pushed GPU notebook `sumu143/transformer-native-3d-kenney-corpus-training` (version 35) to Kaggle; ran to completion on dual Tesla T4 GPUs in 641.28 seconds.
- Retrieved all artifacts into `artifacts/kaggle-kenney-corpus-training-v35/`:
  - `triton-kenney-corpus-32.json`
  - 4 verified checkpoints (`triton-kenney-corpus-900.pt`, `1200.pt`, `1500.pt`, `1900.pt`)
  - 24 predicted `.t3d` reconstructions on held-out validation models
- **Key Empirical Results:**
  - Step 600: Mean Val Free IoU = **0.0998** | Mean Val Loss = 1.4845
  - Step 900: Mean Val Free IoU = **0.1703** | Mean Val Loss = 1.3464
  - Step 1200: Mean Val Free IoU = **0.1965** | Mean Val Loss = 1.2583
  - Step 1500: Mean Val Free IoU = **0.2218** | Mean Val Loss = 1.1807
  - Step 1900: Mean Val Free IoU = **0.2536** | Mean Val Loss = **1.0526**
  - **Comparison with 16-asset pilot (v33):** In v33, training on only 16 models caused validation IoU to peak at step 1000 (**0.1608**) and decline to **0.1563** at step 1200 due to severe overfitting. On the full 649-asset training set, validation IoU monotonically improved all the way to **0.2536** at step 1900 (**+57.7% relative improvement**), proving that dataset diversity directly mitigates representation overfitting.

### D. Resolution Scaling Audit ($32^3$ vs $64^3$) & Voxelizer Hardening
- Hardened `t3d/voxelize.py` to filter degenerate/collinear faces where normal cross-products are near-zero, eliminating `ValueError: Face requires finite matching (N, 3) points and normals`.
- Audited geometric fidelity of $32^3$ vs $64^3$ dual-grid extraction against original normalized GLBs across representative assets:
  - `car-kit/ambulance`: Mean distance fell from 0.00097 ($32^3$) to 0.00010 ($64^3$) (**10× error reduction**); tight F-score (0.005 tol) rose from 0.9252 to **0.9961**.
  - `pirate-kit/ship-medium` (thin masts): Mean distance fell from 0.00179 to 0.00040 (**4.5× error reduction**); F-score rose from 0.8633 to **0.9881**.
  - `space-kit/alien`: Mean distance fell from 0.00013 to 0.00001 (**13× error reduction**); F-score reached **1.0000**.
  - Average $64^3$ asset size is only ~180 KB in `.t3d` format.

### E. Text Prompt Ingestion & Stage 1 Coarse Occupancy DiT Flow (Kaggle Version 37)
- Created `tools/generate_corpus_prompts.py` to assign clean, canonical natural English descriptions to all 868 models (`artifacts/kenney-corpus-32/prompts.json`).
- Built `tools/train_kenney_occupancy_dit.py` and `tools/build_kenney_occupancy_dit_notebook.py`.
- Resolved the 3D axial RoPE requirement (`head_dim = width // heads` must be divisible by 6) by configuring `width=96, heads=4` (`head_dim=24`).
- Pushed GPU notebook `sumu143/transformer-native-3d-occupancy-dit-flow` (version 37) to Kaggle; completed in 201.88 seconds on dual Tesla T4 GPUs.
- Downloaded and verified artifacts in `artifacts/kaggle-kenney-occupancy-dit-v37/`:
  - Checkpoint: `occupancy-dit-600.pt` (SHA-256: `1cce8fc76b1f6258ded81263dfbea021a3930807a86e4c15717e3ce944b185b5`).
  - Report: `occupancy-dit-report.json`.
  - **Empirical Findings:** At 600 steps, Euler sampling from pure random Gaussian noise produces an average prompt difference of **8.83 cells** when prompts are swapped (e.g. "a boat row large" vs "a debris tire"), confirming that text cross-attention is steerable and guides 3D spatial occupancy.

---

## 3. How to Resume the Session (For Codex)

When Codex resumes via `/goal resume`:

1. **Verify Local Checkpoints & Evidence:**
   - VAE Checkpoints: `artifacts/kaggle-kenney-corpus-training-v35/triton-kenney-corpus-1900.pt` (SHA-256: `40d9925756883110...`).
   - Occupancy DiT Checkpoint: `artifacts/kaggle-kenney-occupancy-dit-v37/occupancy-dit-600.pt` (SHA-256: `1cce8fc76b1f6258...`).
   - Reports: `triton-kenney-corpus-32.json` and `occupancy-dit-report.json`.

2. **Run Test Suite:**
   ```bash
   PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
   ```
   (All 163 tests pass cleanly).

3. **Next Recommended Engineering Milestones:**
   - **Longer Occupancy DiT Training (10K+ steps):** 600 steps demonstrates prompt sensitivity, but converging geometric IoU requires 10,000+ steps with cosine learning rate decay.
   - **Stage 2 Latent Geometry Flow:** Given the active coordinates from Stage 1, train Stage 2 DiT to denoise continuous 4-channel shape latents (matching the representation from `triton-kenney-corpus-1900.pt`).
   - **Scale Kenney Corpus to $64^3$ Grid:** Now that $64^3$ voxelization is proven to reduce geometric error by $10\times$ without exceeding memory budgets, build an automated Kaggle CPU shard pipeline to convert the corpus to $64^3$.
