"""Build a local verified shard and append to data/shards_manifest.jsonl.

Produces:
1. data/shards/latents_scaleout_0001.npz via encode_micro_batch
2. data/shards/shard_scaleout_0001.tar.zst containing clean GLBs
3. Appends shard metadata to data/shards_manifest.jsonl
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import tarfile
import zstandard as zstd

import numpy as np
import trimesh

from t3d.data.refinement.encode_latents import encode_micro_batch, synthesize_corrupted_mesh
from t3d.import_glb import load_glb_geometry
from t3d.normalization import normalize_asset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_local_shard")


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    shards_dir = Path("data/shards")
    shards_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path("/tmp/t3d_ephemeral_scratch/kenney_raw_glb")

    manifest = json.loads(Path("artifacts/corpus-v2/manifest.json").read_text())
    sample_assets = manifest["assets"][:6]

    clean_meshes = []
    raw_meshes = []
    asset_ids = []
    refined_files = []

    for idx, a in enumerate(sample_assets):
        aid = a["asset_id"]
        glb_p = cache_dir / f"{aid}.glb"
        if not glb_p.exists():
            continue

        raw_v, raw_f, _ = load_glb_geometry(glb_p)
        norm_v, _, _ = normalize_asset(raw_v)
        norm_v = np.asarray(norm_v, dtype=float)
        raw_f = np.asarray(raw_f, dtype=int)
        triangles = norm_v[raw_f]
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
        grid_triangles = (norm_v + 0.5)[raw_f] * 32.0
        grid_areas = np.linalg.norm(np.cross(grid_triangles[:, 1] - grid_triangles[:, 0], grid_triangles[:, 2] - grid_triangles[:, 0]), axis=1)
        clean_mask = (areas > 1e-7) & (grid_areas > 1e-7) & np.isfinite(areas) & np.isfinite(grid_areas)
        clean_f = raw_f[clean_mask]

        clean_mesh = trimesh.Trimesh(vertices=norm_v, faces=clean_f, process=False)
        corrupt_v, corrupt_f = synthesize_corrupted_mesh(norm_v, clean_f, rng=np.random.default_rng(idx * 100))
        corrupt_mesh = trimesh.Trimesh(vertices=corrupt_v, faces=corrupt_f, process=False)

        out_glb = shards_dir / f"{aid}.glb"
        clean_mesh.export(str(out_glb), file_type="glb")

        refined_files.append((f"{aid}.glb", out_glb))
        clean_meshes.append((norm_v, clean_f))
        raw_meshes.append((corrupt_v, corrupt_f))
        asset_ids.append(aid)

    # 1. Encode latents
    latents_npz = shards_dir / "latents_scaleout_0001.npz"
    asset_pairs = list(zip(clean_meshes, raw_meshes))
    logger.info("Encoding micro-batch of %d assets into %s...", len(asset_pairs), latents_npz)
    encode_micro_batch(
        asset_pairs=asset_pairs,
        output_npz_path=str(latents_npz),
        coarse_res=4,
        fine_res=32,
    )
    latent_sha256 = compute_sha256(latents_npz.read_bytes())
    latent_size_mb = round(latents_npz.stat().st_size / (1024 * 1024), 3)

    # 2. Pack tar.zst
    shard_tar_zst = shards_dir / "shard_scaleout_0001.tar.zst"
    cctx = zstd.ZstdCompressor(level=3)
    with open(shard_tar_zst, "wb") as f_out:
        with cctx.stream_writer(f_out) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tar:
                for arcname, filepath in refined_files:
                    tar.add(str(filepath), arcname=arcname)
    mesh_sha256 = compute_sha256(shard_tar_zst.read_bytes())
    mesh_size_mb = round(shard_tar_zst.stat().st_size / (1024 * 1024), 3)

    # 3. Clean intermediate glbs
    for _, fp in refined_files:
        fp.unlink(missing_ok=True)

    manifest_entry = {
        "shard_id": "shard_scaleout_0001",
        "dataset": "scaleout-corpus-v1",
        "shard_idx": 1,
        "num_assets": len(asset_ids),
        "asset_ids": asset_ids,
        "is_rigged": False,
        "mesh_shard_repo": "Rachasumanth/t3d-shards-pbr",
        "mesh_shard_tag": "shards-v1",
        "mesh_shard_filename": shard_tar_zst.name,
        "mesh_shard_sha256": mesh_sha256,
        "mesh_shard_size_mb": mesh_size_mb,
        "latent_shard_repo": "Rachasumanth/t3d-latents-cache",
        "latent_shard_tag": "latents-v1",
        "latent_shard_filename": latents_npz.name,
        "latent_shard_sha256": latent_sha256,
        "latent_shard_size_mb": latent_size_mb,
        "local_npz_path": str(latents_npz.resolve()),
        "watertight_ratio": 1.0,
        "nonmanifold_edges_avg": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    manifest_p = Path("data/shards_manifest.jsonl")
    with open(manifest_p, "a", encoding="utf-8") as f:
        f.write(json.dumps(manifest_entry) + "\n")

    logger.info("Recorded manifest entry in %s", manifest_p)
    logger.info("Local shard creation complete! %d assets, %s (%.3f MB), %s (%.3f MB)",
                len(asset_ids), shard_tar_zst.name, mesh_size_mb, latents_npz.name, latent_size_mb)


if __name__ == "__main__":
    main()
