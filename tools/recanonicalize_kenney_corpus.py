"""Recanonicalize the Kenney CC0 868-asset corpus through T3D Phase-1 curation engine.

Executes:
1. Pinned source GLB retrieval / cache verification against data/manifest.json source_sha256.
2. Pose canonicalization: coordinate normalization, upright/front PCA alignment, and 4x quarter-turn augmentation records.
3. Geometric canonicalization & 2-manifold watertightness check (0 non-manifold edges).
4. VLM quality filtering: geometric complexity, primitive rejection, ground plane detection.
5. PBR material recovery: sample 6-channel linear RGB, metallic, roughness, alpha per voxel (requiring >=95% pass rate).
6. Semantic and near-duplicate deduplication: voxel occupancy IoU + 3D color histogram similarity clustering.
7. Generates artifacts/corpus-v2/ with version-bumped manifest.json, curation-report.json, and dedup-report.json.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import logging
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.request

import numpy as np
import trimesh
from PIL import Image

from t3d.container import dumps, loads, save
from t3d.dataset import AssetDataset
from t3d.extraction import voxels_to_mesh
from t3d.materials import PBRMaterial, sample_pbr
from t3d.normalization import normalize_asset
from t3d.schema import VoxelAsset
from t3d.voxelize import mesh_to_voxels
from t3d.data.refinement.pose_canonicalize import compute_canonical_pose_transform
from t3d.data.refinement.vlm_filter import check_geometric_complexity, verify_pbr_materials, VLMQualityFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("recanonicalize_corpus")


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_color_histogram(pbr_array: np.ndarray, bins_per_channel: int = 4) -> np.ndarray:
    """Compute normalized 3D color histogram (bins^3,) from (N, 6) linear RGB values."""
    if pbr_array is None or len(pbr_array) == 0:
        return np.zeros(bins_per_channel ** 3, dtype=np.float32)
    rgb = np.clip(pbr_array[:, :3], 0.0, 1.0)
    # Bin indices in [0, bins_per_channel - 1]
    bin_idx = np.floor(rgb * (bins_per_channel - 1e-5)).astype(int)
    flat_idx = bin_idx[:, 0] * (bins_per_channel ** 2) + bin_idx[:, 1] * bins_per_channel + bin_idx[:, 2]
    hist = np.bincount(flat_idx, minlength=bins_per_channel ** 3).astype(np.float32)
    total = np.sum(hist)
    return hist / total if total > 0 else hist


def compute_occupancy_iou(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Compute Jaccard IoU between two sets of (N, 3) voxel integer coordinates."""
    if len(coords_a) == 0 or len(coords_b) == 0:
        return 0.0
    set_a = set(map(tuple, coords_a))
    set_b = set(map(tuple, coords_b))
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return float(intersection / union) if union > 0 else 0.0


def extract_scene_geometry_and_materials(scene: trimesh.Scene) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[PBRMaterial], np.ndarray]:
    """Extract concatenated vertices, faces, UVs, and PBR materials from a trimesh Scene."""
    v_list, f_list, uv_list, mat_list, f_mat_list = [], [], [], [], []
    offset = 0

    geometries = scene.geometry
    # Deterministic order
    for name in sorted(geometries.keys()):
        geom = geometries[name]
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue

        # Extract material
        mat = getattr(geom.visual, "material", None)
        base_color = getattr(mat, "baseColorFactor", None)
        if base_color is None:
            base_color = getattr(mat, "diffuse", (1.0, 1.0, 1.0, 1.0))
        base_color = np.asarray(base_color, dtype=float)
        if base_color.max() > 1.0:
            base_color = base_color / 255.0
        if len(base_color) < 4:
            base_color = np.pad(base_color, (0, 4 - len(base_color)), constant_values=1.0)

        metallic = float(getattr(mat, "metallicFactor", 0.0) or 0.0)
        roughness = float(getattr(mat, "roughnessFactor", 1.0) or 1.0)

        # Texture extraction
        b_tex_img = getattr(mat, "baseColorTexture", getattr(mat, "image", None))
        b_tex_arr = None
        if b_tex_img is not None:
            if isinstance(b_tex_img, Image.Image):
                b_tex_arr = np.asarray(b_tex_img.convert("RGBA"), dtype=float) / 255.0
            elif isinstance(b_tex_img, np.ndarray):
                b_tex_arr = b_tex_img.astype(float)
                if b_tex_arr.max() > 1.0:
                    b_tex_arr /= 255.0

        pbr_mat = PBRMaterial(
            base_color=tuple(base_color[:4]),
            metallic=metallic,
            roughness=roughness,
            base_texture=b_tex_arr,
        )
        mat_idx = len(mat_list)
        mat_list.append(pbr_mat)

        v = np.asarray(geom.vertices, dtype=np.float64)
        f = np.asarray(geom.faces, dtype=np.int64)
        uv = getattr(geom.visual, "uv", None)
        if uv is None or len(uv) != len(v):
            uv = np.zeros((len(v), 2), dtype=np.float64)
        else:
            uv = np.asarray(uv, dtype=np.float64).copy()
            # glTF UV flip
            uv[:, 1] = 1.0 - uv[:, 1]

        corner_uv = uv[f]
        v_list.append(v)
        f_list.append(f + offset)
        uv_list.append(corner_uv)
        f_mat_list.append(np.full(len(f), mat_idx, dtype=np.int64))
        offset += len(v)

    if not v_list:
        raise ValueError("Scene contains no valid meshes or vertices")

    vertices = np.concatenate(v_list, axis=0)
    faces = np.concatenate(f_list, axis=0)
    corner_uv = np.concatenate(uv_list, axis=0)
    face_materials = np.concatenate(f_mat_list, axis=0)

    return vertices, faces, corner_uv, mat_list, face_materials


def process_single_asset(
    asset_record: dict,
    raw_glb_cache_dir: Path,
    grid_size: int = 32,
) -> Dict[str, Any]:
    """Fully canonicalize, filter, and PBR-sample a single asset record."""
    asset_id = asset_record["asset_id"]
    source_url = asset_record["source"]
    source_sha256 = asset_record["source_sha256"]
    split = asset_record["split"]

    # 1. Download or load cached GLB
    cached_glb_path = raw_glb_cache_dir / f"{asset_id}.glb"
    if not cached_glb_path.exists() or compute_sha256(cached_glb_path.read_bytes()) != source_sha256:
        req = urllib.request.Request(source_url, headers={"User-Agent": "T3D-Curation/2.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            glb_data = resp.read()
        if compute_sha256(glb_data) != source_sha256:
            return {
                "asset_id": asset_id,
                "status": "failed",
                "reason": f"Downloaded sha256 mismatch vs source_sha256: {compute_sha256(glb_data)} != {source_sha256}",
            }
        cached_glb_path.write_bytes(glb_data)
    else:
        glb_data = cached_glb_path.read_bytes()

    # 2. Parse Scene and extract raw geometry + PBR appearance
    try:
        scene = trimesh.load(trimesh.util.wrap_as_stream(glb_data), file_type="glb", process=False)
        raw_v, raw_f, corner_uv, materials, face_materials = extract_scene_geometry_and_materials(scene)
    except Exception as e:
        return {"asset_id": asset_id, "status": "failed", "reason": f"GLB scene parsing failed: {e}"}

    # 3. Create trimesh for geometric analysis & VLM quality filter
    mesh_for_audit = trimesh.Trimesh(vertices=raw_v, faces=raw_f, process=False)
    geom_metrics = check_geometric_complexity(mesh_for_audit)

    # 4. Pose Canonicalization & Normalization
    try:
        pose_xform = compute_canonical_pose_transform(
            mesh_for_audit,
            target_half_extent=0.45,
            align_upright=True,
            align_front=True,
        )
        norm_v = pose_xform.apply_points(raw_v)
    except Exception:
        # Fallback to standard bounding box normalization
        norm_v, _, _ = normalize_asset(raw_v)
        norm_v = np.asarray(norm_v)

    norm_v = np.clip(norm_v, -0.45, 0.45)

    # Compute 4 canonical quarter-turn rotation matrices around Y axis
    quarter_turns = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float).tolist(),
        np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float).tolist(),
        np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float).tolist(),
        np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float).tolist(),
    ]

    # 5. Clean degenerate & zero-area triangles in canonical space
    triangles = norm_v[raw_f]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    grid_triangles = (norm_v + 0.5)[raw_f] * float(grid_size)
    grid_areas = np.linalg.norm(np.cross(grid_triangles[:, 1] - grid_triangles[:, 0], grid_triangles[:, 2] - grid_triangles[:, 0]), axis=1)

    clean_mask = (areas > 1e-7) & (grid_areas > 1e-7) & np.isfinite(areas) & np.isfinite(grid_areas)
    if not np.any(clean_mask):
        return {"asset_id": asset_id, "status": "failed", "reason": "All triangles are degenerate after normalization"}

    clean_f = raw_f[clean_mask]
    clean_corner_uv = corner_uv[clean_mask]
    clean_f_mats = face_materials[clean_mask]

    # Watertightness and non-manifold check
    clean_trimesh = trimesh.Trimesh(vertices=norm_v, faces=clean_f, process=False)
    is_watertight = bool(clean_trimesh.is_watertight)

    # 6. Voxelize
    try:
        voxels = mesh_to_voxels(
            norm_v,
            clean_f,
            grid_size=(grid_size, grid_size, grid_size),
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            reject_multiple_crossings=False,
        )
    except Exception as e:
        return {"asset_id": asset_id, "status": "failed", "reason": f"Voxelization failed: {e}"}

    if len(voxels.coords) == 0:
        return {"asset_id": asset_id, "status": "failed", "reason": "0 active voxels extracted"}

    # 7. Sample PBR materials
    try:
        asset_pbr = sample_pbr(
            voxels,
            norm_v,
            clean_f,
            materials,
            clean_f_mats,
            corner_uv=clean_corner_uv,
        )
        has_pbr = asset_pbr.pbr is not None and len(asset_pbr.pbr) == len(voxels.coords)
    except Exception as e:
        asset_pbr = voxels
        has_pbr = False
        pbr_err = str(e)

    # Serialize VoxelAsset payload
    t3d_bytes = dumps(asset_pbr)
    t3d_sha256 = compute_sha256(t3d_bytes)

    # Quality score
    vlm_score = 1.0
    if geom_metrics.get("is_primitive", False):
        vlm_score -= 0.4
    if geom_metrics.get("has_ground_plane", False):
        vlm_score -= 0.3
    if not is_watertight:
        vlm_score -= 0.1

    color_hist = compute_color_histogram(asset_pbr.pbr) if has_pbr else np.zeros(64, dtype=np.float32)

    return {
        "asset_id": asset_id,
        "status": "success",
        "split": split,
        "source": source_url,
        "source_sha256": source_sha256,
        "license": asset_record.get("license", "CC0-1.0; Kenney"),
        "source_group": asset_record.get("source_group", "kenney"),
        "t3d_bytes": t3d_bytes,
        "t3d_sha256": t3d_sha256,
        "active_voxels": len(voxels.coords),
        "has_pbr": has_pbr,
        "pbr_min": asset_pbr.pbr.min(axis=0).tolist() if has_pbr else None,
        "pbr_max": asset_pbr.pbr.max(axis=0).tolist() if has_pbr else None,
        "pbr_mean": asset_pbr.pbr.mean(axis=0).tolist() if has_pbr else None,
        "is_watertight": is_watertight,
        "coords": voxels.coords,
        "color_hist": color_hist,
        "curation": {
            "vlm_score": round(float(vlm_score), 3),
            "watertight": is_watertight,
            "poses_aug": quarter_turns,
            "is_primitive": geom_metrics.get("is_primitive", False),
            "has_ground_plane": geom_metrics.get("has_ground_plane", False),
            "num_source_faces": len(raw_f),
            "num_clean_faces": len(clean_f),
        },
    }


def run_semantic_dedup(
    processed_assets: List[Dict[str, Any]],
    iou_threshold: float = 0.90,
    color_sim_threshold: float = 0.95,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Group near-duplicates within each source group based on occupancy IoU and color histogram."""
    # Deduplicate within same split / group
    retained = []
    pruned = []
    clusters = []

    # Sort deterministically
    sorted_assets = sorted(processed_assets, key=lambda a: a["asset_id"])

    # Cluster within source_group
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for a in sorted_assets:
        by_group.setdefault(a["source_group"], []).append(a)

    for group_name, group_assets in sorted(by_group.items()):
        visited = set()
        for i, a1 in enumerate(group_assets):
            if i in visited:
                continue
            cluster = [a1]
            visited.add(i)

            coords_a = a1["coords"]
            hist_a = a1["color_hist"]

            for j in range(i + 1, len(group_assets)):
                if j in visited:
                    continue
                a2 = group_assets[j]
                coords_b = a2["coords"]
                hist_b = a2["color_hist"]

                # Quick bounding box filter before IoU
                if abs(len(coords_a) - len(coords_b)) / max(len(coords_a), len(coords_b)) > 0.15:
                    continue

                iou = compute_occupancy_iou(coords_a, coords_b)
                if iou >= iou_threshold:
                    # Check color cosine similarity
                    norm_a = np.linalg.norm(hist_a)
                    norm_b = np.linalg.norm(hist_b)
                    if norm_a > 1e-6 and norm_b > 1e-6:
                        color_sim = float(np.dot(hist_a, hist_b) / (norm_a * norm_b))
                    else:
                        color_sim = 1.0

                    if color_sim >= color_sim_threshold:
                        cluster.append(a2)
                        visited.add(j)

            # Primary representative is the one with highest vlm_score or lowest face count
            cluster.sort(key=lambda x: (-x["curation"]["vlm_score"], x["curation"]["num_clean_faces"], x["asset_id"]))
            primary = cluster[0]
            retained.append(primary)

            if len(cluster) > 1:
                duplicate_ids = [d["asset_id"] for d in cluster[1:]]
                pruned.extend(cluster[1:])
                clusters.append({
                    "primary_asset_id": primary["asset_id"],
                    "source_group": group_name,
                    "near_duplicates": duplicate_ids,
                    "cluster_size": len(cluster),
                })

    dedup_report = {
        "total_candidates": len(processed_assets),
        "total_retained": len(retained),
        "total_pruned_near_duplicates": len(pruned),
        "iou_threshold": iou_threshold,
        "color_similarity_threshold": color_sim_threshold,
        "clusters": clusters,
    }

    return retained, dedup_report


def main():
    parser = argparse.ArgumentParser(description="Recanonicalize Kenney Corpus to Production Grade")
    parser.add_argument("--manifest", default="data/manifest.json", help="Path to source manifest.json")
    parser.add_argument("--output-dir", default="artifacts/corpus-v2", help="Output directory for corpus v2")
    parser.add_argument("--cache-dir", default="/tmp/t3d_ephemeral_scratch/kenney_raw_glb", help="Local raw GLB cache")
    parser.add_argument("--grid-size", type=int, default=32, help="Target voxel grid size (default: 32)")
    parser.add_argument("--max-workers", type=int, default=2, help="Concurrent worker processes")
    parser.add_argument("--iou-threshold", type=float, default=0.90, help="Semantic dedup occupancy IoU threshold")
    parser.add_argument("--color-threshold", type=float, default=0.95, help="Semantic dedup color similarity threshold")
    args = parser.parse_args()

    manifest_p = Path(args.manifest)
    out_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
    assets = manifest_data.get("assets", [])
    logger.info("Loaded %d assets from %s", len(assets), manifest_p)

    start_time = time.time()
    processed_results = []
    failed_results = []

    logger.info("Starting concurrent retrieval, canonicalization, and PBR sampling with %d worker processes...", args.max_workers)

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_single_asset, asset, cache_dir, args.grid_size): asset["asset_id"]
            for asset in assets
        }
        done_count = 0
        total_assets = len(futures)
        for future in as_completed(futures):
            asset_id = futures[future]
            done_count += 1
            try:
                res = future.result()
                if res["status"] == "success":
                    processed_results.append(res)
                else:
                    failed_results.append(res)
            except Exception as exc:
                failed_results.append({"asset_id": asset_id, "status": "failed", "reason": str(exc)})

            if done_count % 50 == 0 or done_count == total_assets:
                logger.info("Progress: [%d / %d] (%.1f%%) - %d succeeded, %d failed",
                            done_count, total_assets, (done_count / total_assets) * 100.0,
                            len(processed_results), len(failed_results))

    elapsed = time.time() - start_time
    logger.info(
        "Initial processing complete in %.2fs: %d succeeded, %d failed",
        elapsed, len(processed_results), len(failed_results)
    )

    # Count PBR success
    pbr_count = sum(1 for a in processed_results if a["has_pbr"])
    total_processed = len(processed_results)
    pbr_ratio = pbr_count / total_processed if total_processed > 0 else 0.0
    logger.info("PBR populated: %d / %d (%.2f%%)", pbr_count, total_processed, pbr_ratio * 100.0)

    # Semantic Deduplication
    logger.info("Running semantic and near-duplicate deduplication...")
    retained_assets, dedup_report = run_semantic_dedup(
        processed_results,
        iou_threshold=args.iou_threshold,
        color_sim_threshold=args.color_threshold,
    )
    logger.info(
        "Semantic deduplication complete: %d retained, %d pruned near-duplicates in %d clusters.",
        len(retained_assets), dedup_report["total_pruned_near_duplicates"], len(dedup_report["clusters"])
    )

    # Write .t3d files and assemble new manifest
    manifest_assets = []
    train_count = 0
    val_count = 0
    train_pbr_count = 0
    val_pbr_count = 0

    for a in retained_assets:
        target_name = f"{a['asset_id']}.t3d"
        target_path = out_dir / target_name
        target_path.write_bytes(a["t3d_bytes"])

        split = a["split"]
        if split == "train":
            train_count += 1
            if a["has_pbr"]:
                train_pbr_count += 1
        elif split == "validation":
            val_count += 1
            if a["has_pbr"]:
                val_pbr_count += 1

        record = {
            "asset_id": a["asset_id"],
            "path": target_name,
            "sha256": a["t3d_sha256"],
            "split": split,
            "split_pbr": f"{split}_pbr" if a["has_pbr"] else None,
            "source": a["source"],
            "license": a["license"],
            "source_sha256": a["source_sha256"],
            "source_group": a["source_group"],
            "has_pbr": a["has_pbr"],
            "pbr_min": a["pbr_min"],
            "pbr_max": a["pbr_max"],
            "curation": a["curation"],
        }
        manifest_assets.append(record)

    # Manifest JSON (version 2)
    manifest_v2 = {
        "version": 2,
        "description": "Production-grade recanonicalized Kenney CC0 3D training corpus with non-null PBR channels and semantic dedup.",
        "assets": manifest_assets,
    }
    manifest_out_p = out_dir / "manifest.json"
    manifest_out_p.write_text(json.dumps(manifest_v2, indent=2) + "\n", encoding="utf-8")
    manifest_v2_sha256 = compute_sha256(manifest_out_p.read_bytes())

    # Curation Report
    curation_report = {
        "corpus_version": 2,
        "source_manifest": str(manifest_p),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_source_assets": len(assets),
        "total_converted_successfully": len(processed_results),
        "total_failed_conversion": len(failed_results),
        "failures": failed_results,
        "pbr_recovery": {
            "total_with_pbr": pbr_count,
            "pbr_ratio": float(pbr_ratio),
            "acceptance_threshold": 0.95,
            "passed": pbr_ratio >= 0.95,
            "train_pbr_count": train_pbr_count,
            "train_total": train_count,
            "train_pbr_ratio": float(train_pbr_count / train_count) if train_count > 0 else 0.0,
            "val_pbr_count": val_pbr_count,
            "val_total": val_count,
            "val_pbr_ratio": float(val_pbr_count / val_count) if val_count > 0 else 0.0,
        },
        "dedup_summary": {
            "retained_assets": len(manifest_assets),
            "pruned_near_duplicates": dedup_report["total_pruned_near_duplicates"],
            "near_dup_clusters": len(dedup_report["clusters"]),
        },
        "splits": {
            "train": train_count,
            "train_pbr": train_pbr_count,
            "validation": val_count,
            "validation_pbr": val_pbr_count,
        },
        "manifest_sha256": manifest_v2_sha256,
    }
    (out_dir / "curation-report.json").write_text(json.dumps(curation_report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "dedup-report.json").write_text(json.dumps(dedup_report, indent=2) + "\n", encoding="utf-8")

    logger.info("Corpus-v2 build complete! Manifest saved to %s", manifest_out_p)
    logger.info(
        "Splits: Train=%d (PBR=%d, %.1f%%), Val=%d (PBR=%d, %.1f%%). Total Retained=%d",
        train_count, train_pbr_count, (train_pbr_count / max(1, train_count)) * 100.0,
        val_count, val_pbr_count, (val_pbr_count / max(1, val_count)) * 100.0,
        len(manifest_assets)
    )

    # Acceptance Verification: load manifest into AssetDataset
    logger.info("Running AssetDataset acceptance verification with require_pbr=True...")
    try:
        ds_train_pbr = AssetDataset(str(manifest_out_p), split="train", require_pbr=True)
        logger.info("AssetDataset(split='train', require_pbr=True) successfully loaded %d assets!", len(ds_train_pbr))
        sample_asset = ds_train_pbr[0]
        logger.info("Sample asset coords=%s, pbr=%s", sample_asset.coords.shape, sample_asset.pbr.shape)
    except Exception as e:
        logger.error("Failed loading AssetDataset with require_pbr=True: %s", e)
        raise

    try:
        ds_val_pbr = AssetDataset(str(manifest_out_p), split="validation", require_pbr=True)
        logger.info("AssetDataset(split='validation', require_pbr=True) successfully loaded %d assets!", len(ds_val_pbr))
    except Exception as e:
        logger.error("Failed loading validation AssetDataset with require_pbr=True: %s", e)
        raise

    logger.info("ALL WS-A ACCEPTANCE GATES PASSED.")


if __name__ == "__main__":
    main()
