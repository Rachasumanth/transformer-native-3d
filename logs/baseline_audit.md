# T3D Baseline Architectural Audit & Pre-Refactor Snapshot

**Date:** 2026-11-07  
**Git Baseline Commit:** `e01211ec295857b9ab550881cf13b981689475bf`  
**Git Baseline Tag:** `pre-refactor-baseline`  

---

## 1. Executive Summary & Existing Architecture Map

The existing T3D codebase is structured around a two-stage cascaded diffusion/flow architecture coupled with sparse convolutional autoencoding:

### Component Map
1. **Stage 1: Coupled Occupancy DiT (`t3d/coupled_occupancy.py`)**
   - Operates on a coarse grid (`32^3` or `64^3`), predicting binary shape occupancy ($o_i \in \{-1, +1\}$) and optional skeleton containment mask.
   - Bottleneck: Two-stage cascade introduces false positives and boundary misalignments (recall ~0.86).

2. **Stage 2: Continuous Latent Geometry Flow (`t3d/flow.py`, `t3d/generation.py`)**
   - Flow matching network conditioned on active occupancy voxels and prompt embeddings.
   - Outputs continuous SC-VAE latent features ($z_i \in \mathbb{R}^C$) only for predicted occupied cells.
   - Bottleneck: Stage 2 is completely uncoupled from Stage 1 during training; errors in Stage 1 propagate irreversibly into Stage 2.

3. **Sparse Compression VAE & Triton Decoder (`t3d/sc_vae.py`, `t3d/triton_sparse.py`, `t3d/sparse_blocks.py`)**
   - 4-stage hierarchical sparse ConvNeXt autoencoder compressing volumetric Hermite data (dual vertices, intersected flags, split weights) into continuous latents.
   - Triton-accelerated submanifold convolution kernels (`_submanifold_forward`) for fast inference.
   - Bottleneck: Lacks an explicit Hermite surface normal head, causing Dual Contouring to clip razor-sharp low-poly features and producing non-manifold edge artifacts.

4. **Dual Contouring & Topology Repair (`t3d/extraction.py`, `t3d/extraction_torch.py`, `t3d/qef.py`, `t3d/topology_repair.py`)**
   - Dual-grid transverse quad extraction with split-weight diagonal decision and QEF mass-point centering.
   - Local fan repair (`repair_local_fans`) welded duplicate sheets and closed local one-rings, but baseline generated meshes still output 4–11 non-manifold edges.

5. **Skeleton Flow & Skinning (`t3d/articulation.py`, `t3d/skin_ae.py`, `t3d/import_rig.py`)**
   - Rigging based on field inference and `SkinAE` (latent skinning autoencoder).
   - Major Bottlenecks: `SkinAE` suffers from joint Chamfer error (~0.0901 to 0.50), containment violations (joints floating outside the mesh), and mesh tearing under animation retargeting.

---

## 2. Baseline Performance vs. Production Acceptance Gates

| Metric | Pre-Refactor Baseline | Production Gate Target | Status |
| :--- | :--- | :--- | :--- |
| **Watertightness & Topology** | 4–11 non-manifold edges / asset | **0 non-manifold edges, 0 self-intersections (100% 2-manifold)** | **FAILED** |
| **Coarse Occupancy Recall** | 0.8600 | **>= 0.9800** | **FAILED** |
| **Mean Surface Distance (MSD)** | 0.0766 (ID) / 0.0923 (OOD) | **<= 0.0350** | **FAILED** |
| **F-Score (@ 0.01 tolerance)** | 0.1155 | **>= 0.7000** | **FAILED** |
| **Rigging Joint-to-Joint Error**| 0.0901 (Chamfer up to 0.50) | **<= 0.0500 with zero containment violations** | **FAILED** |
| **PBR Material Export** | RGBA only mapped into COLOR_0 | **Full 6-channel PBR (Basecolor, Metallic, Roughness, Normal)** | **FAILED** |

---

## 3. Root Cause Analysis
1. **Decoupled Two-Stage DiT:** Compounding errors between binary occupancy classification and continuous latent flow.
2. **Missing Canonicalization & Synthetic Corruption Engine:** Ingesting raw web 3D data without universal watertight canonicalization and boundary margin enforcement causes clipping at grid edges.
3. **Flawed Rigging Architecture (`skin_ae`):** Unconstrained learned skinning embeddings disregard surface geodesic distance, creating floating joints and animation tears.
4. **Lack of Decoder-in-the-Loop Loss:** Latent MSE alone does not penalize geometric surface distortions and normal misalignments.

This baseline audit confirms the urgent necessity of the complete data-and-architecture overhaul defined in Milestones 1–5.
