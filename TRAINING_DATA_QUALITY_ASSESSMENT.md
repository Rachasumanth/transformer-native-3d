# T3D Training Data Quality Assessment & Production Readiness Report

**Assessment Date:** 2026-11-07  
**Repository:** `transformer-native-3d`  
**Current Evaluation:** **Remediation In Progress — Hybrid Data Model Implemented**

---

## 1. Executive Summary & Quality Status

A thorough audit of the training data infrastructure, dataset schemas, manifests, and artifact corpora was performed against the blueprint specifications in [`plan.md`](plan.md) (Step 1) and production deployment requirements.

### Key Audit Findings:
1. **Manifest Schema Divergence (Resolved):**
   - **Legacy Shard Manifest (`data/shards_manifest.jsonl`):** Contains shard-level entries (`shard_id`, `mesh_shard_repo`, `latent_shard_repo`, `watertight_ratio`) used by `StreamingLatentsDataset`. While ideal for ephemeral streaming and remote GitHub Release caching, it lacks per-asset records required by `t3d/dataset.py:AssetDataset`.
   - **Asset-Level Manifest (`data/manifest.json`):** Now integrated directly into `data/manifest.json`, providing **868 verified CC0-1.0 assets** from the Kenney corpus with source-disjoint splits (`train`: 649, `validation`: 219), SHA-256 integrity hashes, and provenance metadata. Tested and verified against `t3d/dataset.py:AssetDataset`.

2. **Dataset Scale & Source Diversity:**
   - **Active Production Corpus:** 868 asset records in `data/` and 1 streaming shard in `data/shards_manifest.jsonl`.
   - **Target Scale:** 300,000–500,000+ assets across diverse domains (Aether, Objaverse-XL, Toys4K, PartNet-Mobility, Articulation-XL 2.0, RigNetv1, HSSD, ABO, and 3D-FUTURE).
   - **Status:** The data curation and canonicalization *engine* is production-grade; bulk streaming ingestion across additional external repositories is the remaining scaling phase.

3. **Curation & Filtering Pipelines Implemented:**
   - **VLM Aesthetic & Geometric Filtering ([`t3d/data/refinement/vlm_filter.py`](t3d/data/refinement/vlm_filter.py)):** Multi-view rendering (Front, Right, Top, Isometric depth and normal maps), dihedral angle variance analysis, ground plane and pedestal detection, and PBR material completeness gates.
   - **Pose & Scale Canonicalization ([`t3d/data/refinement/pose_canonicalize.py`](t3d/data/refinement/pose_canonicalize.py)):** Bounding box centering at $(0, 0, 0)$, scaling strictly to $[-0.45, 0.45]^3$ with a mandatory 10% empty boundary margin inside $[-0.5, 0.5]^3$, upright $+Y$ and front $+Z$ PCA alignment, and $90^\circ$ axial rotation data augmentation.
   - **Watertight 2-Manifold Canonicalizer ([`t3d/data/refinement/canonicalize_geom.py`](t3d/data/refinement/canonicalize_geom.py)):** $128^3$ UDF flood-fill remeshing, QEM crease-preserving decimation to $1,500$–$5,000$ triangles, guaranteed $0$ non-manifold edges.
   - **Skeleton & Skinning Pipeline ([`t3d/data/refinement/canonicalize_rig.py`](t3d/data/refinement/canonicalize_rig.py)):** Dead-bone pruning ($W < 0.02$), DFS spatial ordering, barycentric BVH weight transfer, glTF top-4 truncation.

---

## 2. Dataset Schema & Manifest Architecture

To bridge the dual requirements of **in-memory asset datasets** (`t3d/dataset.py`) and **out-of-core streaming micro-batches** (`t3d/data/streaming_loader.py`), T3D employs a **dual manifest architecture**:

```
data/
├── manifest.json              # Asset-level manifest (version 1) for AssetDataset
└── shards_manifest.jsonl      # Shard-level manifest for StreamingLatentsDataset
```

### Schema A: Asset-Level Manifest (`data/manifest.json`)
Conforms to `t3d/dataset.py:AssetDataset`:
```json
{
  "version": 1,
  "assets": [
    {
      "asset_id": "kenney-car-kit-ambulance",
      "path": "kenney-car-kit-ambulance.t3d",
      "sha256": "f94f163a3413fcc5204591aa9693d00b4d350c300f08ee26795dc0e5acbabbfc",
      "split": "train",
      "source": "https://raw.githubusercontent.com/Hidencod/tge-assets/.../ambulance.glb",
      "license": "CC0-1.0; Kenney; https://kenney.nl/assets/car-kit",
      "source_sha256": "d35740315ab0560156a3357f70f6d96333734225e4506e21788120811ab0b758",
      "source_group": "kenney-car-kit"
    }
  ]
}
```
- **Validation:** 868 records (649 train, 219 validation). Zero split leakage across source groups or hashes.

### Schema B: Shard-Level Manifest (`data/shards_manifest.jsonl`)
Conforms to `t3d/data/streaming_loader.py:StreamingLatentsDataset`:
```json
{"shard_id": "shard_aether_0001", "dataset": "aether", "shard_idx": 1, "num_assets": 3, "mesh_shard_repo": "Rachasumanth/t3d-shards-pbr", "latent_shard_repo": "Rachasumanth/t3d-latents-cache", "watertight_ratio": 1.0, "nonmanifold_edges_avg": 0.0}
```
- **Validation:** Backed by private remote GitHub Releases with rolling tags (`shards-v1`, `latents-v1`) to prevent local disk bloating on the 50 GB disk budget.

---

## 3. Component Readiness & Gap Analysis

| Component | Blueprint Specification | Implementation Status | Test Coverage |
| :--- | :--- | :--- | :--- |
| **Asset-Level Manifest** | Version 1 JSON with license & provenance | ✅ Implemented (`data/manifest.json`) | `tests/test_dataset.py` |
| **Streaming Shard Manifest** | Micro-batch remote storage index | ✅ Implemented (`data/shards_manifest.jsonl`) | `tests/test_streaming_loader.py` |
| **VLM Multi-View Filter** | Color/depth/normal multi-view render & prompt | ✅ Implemented (`t3d/data/refinement/vlm_filter.py`) | `tests/test_vlm_filter.py` |
| **Pose Canonicalizer** | Centered $[-0.45, 0.45]^3$, upright $+Y$, front $+Z$ | ✅ Implemented (`t3d/data/refinement/pose_canonicalize.py`) | `tests/test_pose_canonicalize.py` |
| **Geometry Canonicalizer** | $128^3$ UDF, watertight 2-manifold, QEM decimation | ✅ Implemented (`t3d/data/refinement/canonicalize_geom.py`) | `tests/test_canonicalize_geom.py` |
| **Rigging Canonicalizer** | Dead-bone pruning, DFS order, BVH skin transfer | ✅ Implemented (`t3d/data/refinement/canonicalize_rig.py`) | `tests/test_canonicalize_rig.py` |
| **PBR UV Baker** | Single UV atlas, debaked $[0.04, 0.96]$, seam dilation | ✅ Implemented (`t3d/data/refinement/canonicalize_pbr.py`) | `tests/test_canonicalize_pbr.py` |
| **Latent Pre-Encoder** | Continuous latents, Hermite normals, surface ratio | ✅ Implemented (`t3d/data/refinement/encode_latents.py`) | `tests/test_encode_latents.py` |
| **Acceptance Gate Harness**| 5 automated gates (Topology, Recall, MSD, F-Score, Rig) | ✅ Implemented (`scripts/eval_acceptance.py`) | `tests/test_eval_acceptance.py` |

---

## 4. Remediation Roadmap for Large-Scale Production Scaling

1. **Phase 1 (Complete):**
   - Established dual-track manifest interoperability (`data/manifest.json` + `data/shards_manifest.jsonl`).
   - Integrated the 868-asset Kenney corpus as the baseline clean ground-truth manifest.
   - Built and tested VLM filtering, multi-view projection rendering, and pose canonicalization modules.

2. **Phase 2 (Micro-Batch Scaling across Target Repositories):**
   - Run `scripts/ephemeral_ingestion.py` in 500-asset batches against target sources:
     - Aether Assets (1,460 game-ready models)
     - RigNetv1 (2,703 rigged models)
     - PartNet-Mobility (2,000+ articulated models)
     - Toys4K Meshes (4,000 stylized models)
     - Articulation-XL 2.0 (48,000+ models)
   - Filter each micro-batch through `VLMQualityFilter.filter()` before canonicalization.
   - Append new verified shards to `data/shards_manifest.jsonl` and commit.
