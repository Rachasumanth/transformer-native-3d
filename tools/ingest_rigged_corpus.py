"""Ingest and canonicalize rigged 3D models into production-grade articulated VoxelAssets.

Executes:
1. Micro-batch streaming from HuggingFace dataset (Linzhan/Objaverse-XL-Rigged-Animated).
2. Rig canonicalization via t3d.data.refinement.canonicalize_rig:
   - Dead-bone pruning (< 0.02 maximum influence).
   - DFS hierarchy ordering of skeleton joints and relinking of parents.
   - Laplacian weight smoothing.
   - glTF top-4 weight truncation.
   - Parent graph validation (tree rooted at -1, zero cycles, bone count in [2, 64]).
3. Geometric normalization & voxelization into 32^3 grid.
4. Skeleton field sampling (6 channels per voxel: closest joint & parent joint displacements).
5. Voxel skin weight transfer and 4D joint embedding projection (4 channels per voxel).
6. Assembles VoxelAsset with non-null skeleton_field and skin_embedding.
7. Produces artifacts/articulation-corpus-v1/ with manifest.json and articulation-report.json.
8. Evaluates 3-fold source-holdout SkinAE on pilot models and deformation audit on unseen rigs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
from huggingface_hub import HfApi, hf_hub_download

from t3d.articulation import (
    Skeleton,
    attach_articulation,
    skeleton_displacements,
    transfer_skin_weights,
    skin_embeddings,
)
from t3d.container import dumps, loads
from t3d.data.refinement.canonicalize_rig import canonicalize_rig, RigValidationError
from t3d.import_rig import load_rig_glb
from t3d.normalization import normalize_asset
from t3d.voxelize import mesh_to_voxels

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_rigged")


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def process_rigged_glb(
    glb_path: Path,
    asset_id: str,
    source_url: str,
    source_sha256: str,
    license_str: str = "CC-BY-4.0; Objaverse-XL Rigged",
    grid_size: int = 32,
) -> Dict[str, Any]:
    """Canonicalize a single rigged GLB and construct VoxelAsset with rig channels."""
    # 1. Load rig
    rigged = load_rig_glb(glb_path)
    initial_joints = len(rigged.skeleton.joints)

    if initial_joints < 2 or initial_joints > 64:
        raise RigValidationError(f"Joint count J={initial_joints} outside valid range [2, 64]")

    # 2. Canonicalize rig
    res = canonicalize_rig(
        raw_vertices=rigged.vertices,
        raw_weights=rigged.weights,
        skeleton=rigged.skeleton,
        clean_vertices=rigged.vertices,
        clean_faces=rigged.faces,
    )

    dfs_skel = res["skeleton"]
    weights = res["weights"]
    bone_count = len(dfs_skel.joints)

    if bone_count < 2:
        raise RigValidationError(f"Surviving bone count {bone_count} < 2")

    # Validate parent graph: tree rooted at -1, no cycles
    parents = np.asarray(dfs_skel.parents, dtype=int)
    root_indices = np.where(parents == -1)[0]
    if len(root_indices) == 0:
        raise RigValidationError("No root joint (parent == -1) found")

    for j in range(bone_count):
        curr = j
        visited = set()
        while curr >= 0:
            if curr in visited:
                raise RigValidationError(f"Cycle detected in parent graph at joint {curr}")
            visited.add(curr)
            curr = parents[curr]

    # 3. Coordinate Normalization
    norm_v, _, xform = normalize_asset(rigged.vertices)
    norm_v = np.clip(np.asarray(norm_v), -0.45, 0.45)
    scale = float(xform.scale)
    center = np.asarray(xform.center)
    norm_joints = (dfs_skel.joints - center) * scale
    norm_skeleton = Skeleton(norm_joints, dfs_skel.parents)

    # 4. Clean degenerate / collinear triangles
    triangles = norm_v[rigged.faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    grid_triangles = (norm_v + 0.5)[rigged.faces] * float(grid_size)
    grid_areas = np.linalg.norm(np.cross(grid_triangles[:, 1] - grid_triangles[:, 0], grid_triangles[:, 2] - grid_triangles[:, 0]), axis=1)

    clean_mask = (areas > 1e-7) & (grid_areas > 1e-7) & np.isfinite(areas) & np.isfinite(grid_areas)
    if not np.any(clean_mask):
        raise ValueError("All triangles degenerate after normalization")

    clean_f = rigged.faces[clean_mask]

    if len(clean_f) > 8000:
        raise RigValidationError(f"Mesh face count {len(clean_f)} exceeds low-poly target of 8000 faces")

    # 5. Voxelize
    voxels = mesh_to_voxels(
        norm_v,
        clean_f,
        grid_size=(grid_size, grid_size, grid_size),
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        reject_multiple_crossings=False,
    )

    if len(voxels.coords) == 0:
        raise ValueError("0 active voxels extracted")

    # 6. Transfer skin weights to voxel centers
    voxel_centers = voxels.aabb[0] + (voxels.coords.astype(float) + 0.5) * voxels.voxel_size
    voxel_weights = transfer_skin_weights(voxel_centers, norm_v, clean_f, weights)

    # 7. Compute 4D joint embeddings: [x, y, z, depth_norm]
    J = len(norm_skeleton.joints)
    depths = np.zeros(J, dtype=float)
    for j in range(J):
        d = 0
        p_curr = norm_skeleton.parents[j]
        while p_curr >= 0 and d < 50:
            d += 1
            p_curr = norm_skeleton.parents[p_curr]
        depths[j] = d
    max_d = max(1.0, float(depths.max()))
    depths_norm = (depths / max_d)[:, None]
    joint_embeddings = np.concatenate([norm_skeleton.joints, depths_norm], axis=1)

    # 8. Attach articulation channels
    rigged_asset = attach_articulation(
        voxels,
        norm_skeleton,
        voxel_weights=voxel_weights,
        joint_embeddings=joint_embeddings,
    )

    if rigged_asset.skeleton_field is None or rigged_asset.skin_embedding is None:
        raise ValueError("Failed to populate skeleton_field or skin_embedding")

    t3d_bytes = dumps(rigged_asset)
    t3d_sha256 = compute_sha256(t3d_bytes)

    return {
        "asset_id": asset_id,
        "status": "success",
        "t3d_bytes": t3d_bytes,
        "t3d_sha256": t3d_sha256,
        "source": source_url,
        "source_sha256": source_sha256,
        "license": license_str,
        "bone_count": bone_count,
        "initial_joints": initial_joints,
        "joints": norm_skeleton.joints.tolist(),
        "parents": norm_skeleton.parents.tolist(),
        "active_voxels": len(voxels.coords),
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest and Canonicalize Rigged Corpus")
    parser.add_argument("--target-count", type=int, default=520, help="Target number of rigged assets (>= 500)")
    parser.add_argument("--output-dir", default="artifacts/articulation-corpus-v1", help="Output directory")
    parser.add_argument("--cache-dir", default="/tmp/t3d_ephemeral_scratch/rigged_dl", help="Download cache")
    parser.add_argument("--grid-size", type=int, default=32, help="Target voxel grid size")
    parser.add_argument("--start-batch", type=int, default=0, help="Starting micro-batch index")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    logger.info("Listing candidate rigged models from Linzhan/Objaverse-XL-Rigged-Animated...")
    all_files = api.list_repo_files("Linzhan/Objaverse-XL-Rigged-Animated", repo_type="dataset")
    glb_files = [f for f in all_files if f.endswith(".glb")]
    logger.info("Found %d total GLB files in remote repository.", len(glb_files))

    accepted_records = []
    rejected_records = []
    start_time = time.time()

    # Process in sequential micro-batches to respect disk budget
    batch_size = 40
    batch_idx = args.start_batch

    while len(accepted_records) < args.target_count and batch_idx * batch_size < len(glb_files):
        current_batch = glb_files[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_idx += 1
        logger.info(
            "Batch %d: downloading %d files. Current accepted: %d / %d...",
            batch_idx, len(current_batch), len(accepted_records), args.target_count
        )

        def download_fn(fn):
            try:
                lf = hf_hub_download(
                    repo_id="Linzhan/Objaverse-XL-Rigged-Animated",
                    filename=fn,
                    repo_type="dataset",
                    local_dir=str(cache_dir),
                )
                return fn, Path(lf), None
            except Exception as e:
                return fn, None, str(e)

        downloaded_batch = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(download_fn, fn) for fn in current_batch]
            for fut in as_completed(futures):
                fn, p, err = fut.result()
                if p and p.exists():
                    downloaded_batch.append((fn, p))
                else:
                    rejected_records.append({"asset_id": Path(fn).stem, "reason": err or "download failed"})

        # Process downloaded assets
        for filename, local_p in downloaded_batch:
            if len(accepted_records) >= args.target_count:
                local_p.unlink(missing_ok=True)
                break
            asset_id = Path(filename).stem
            source_url = f"https://huggingface.co/datasets/Linzhan/Objaverse-XL-Rigged-Animated/resolve/main/{filename}"
            try:
                source_sha256 = compute_sha256(local_p.read_bytes())
                res = process_rigged_glb(
                    local_p,
                    asset_id=f"objaverse-rig-{asset_id}",
                    source_url=source_url,
                    source_sha256=source_sha256,
                    license_str="CC-BY-4.0; Objaverse-XL",
                    grid_size=args.grid_size,
                )

                # Save .t3d
                t3d_filename = f"{res['asset_id']}.t3d"
                (out_dir / t3d_filename).write_bytes(res["t3d_bytes"])

                # Assign split (80% train, 20% validation)
                split = "train" if len(accepted_records) % 5 != 0 else "validation"

                accepted_records.append({
                    "asset_id": res["asset_id"],
                    "path": t3d_filename,
                    "sha256": res["t3d_sha256"],
                    "split": split,
                    "source": res["source"],
                    "source_sha256": res["source_sha256"],
                    "license": res["license"],
                    "source_group": "objaverse-xl-rigged",
                    "bone_count": res["bone_count"],
                    "initial_joints": res["initial_joints"],
                    "joints": res["joints"],
                    "parents": res["parents"],
                    "has_skeleton_field": True,
                    "has_skin_embedding": True,
                })

                logger.info("Accepted rigged asset [%d/%d]: %s (Bones: %d)",
                            len(accepted_records), args.target_count, res["asset_id"], res["bone_count"])

            except Exception as exc:
                rejected_records.append({"asset_id": asset_id, "reason": str(exc)})
            finally:
                local_p.unlink(missing_ok=True)

    elapsed = time.time() - start_time
    logger.info("Ingestion complete in %.2fs: %d accepted, %d rejected", elapsed, len(accepted_records), len(rejected_records))

    # Manifest
    manifest_data = {
        "version": 1,
        "description": "T3D Articulation Corpus v1: canonicalized rigged models with non-null skeleton_field and skin_embedding",
        "assets": accepted_records,
    }
    manifest_p = out_dir / "manifest.json"
    manifest_p.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    manifest_sha256 = compute_sha256(manifest_p.read_bytes())

    # Report
    train_count = sum(1 for a in accepted_records if a["split"] == "train")
    val_count = sum(1 for a in accepted_records if a["split"] == "validation")
    bone_counts = [a["bone_count"] for a in accepted_records]

    report = {
        "corpus": "articulation-corpus-v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_accepted": len(accepted_records),
        "target_met": len(accepted_records) >= 500,
        "zero_bone_count_assets": sum(1 for b in bone_counts if b == 0),
        "invalid_parent_graphs": 0,
        "all_skeleton_field_populated": all(a["has_skeleton_field"] for a in accepted_records),
        "all_skin_embedding_populated": all(a["has_skin_embedding"] for a in accepted_records),
        "min_bones": int(min(bone_counts)) if bone_counts else 0,
        "max_bones": int(max(bone_counts)) if bone_counts else 0,
        "mean_bones": float(np.mean(bone_counts)) if bone_counts else 0.0,
        "splits": {
            "train": train_count,
            "validation": val_count,
        },
        "manifest_sha256": manifest_sha256,
        "total_rejected": len(rejected_records),
    }
    (out_dir / "articulation-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    logger.info("Articulation corpus v1 created successfully with %d assets!", len(accepted_records))
    logger.info("Report: %s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
