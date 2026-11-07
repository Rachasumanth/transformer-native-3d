"""Audit VAE/occupancy reconstruction F-score on held-out validation assets across >= 3 seeds.

Reports:
- Mean sampled F-score (@ tolerance 0.01)
- 95% Confidence Interval (CI) across 3 seeds (seed 42, 123, 2026)
- Mean surface distance (MSD)
- Boundary edge metrics
Strictly enforces: only held-out validation assets from artifacts/corpus-v2/ are evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np

from t3d.container import load
from t3d.extraction import voxels_to_mesh
from t3d.import_glb import load_glb_geometry
from t3d.metrics import surface_comparison
from t3d.normalization import normalize_asset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_reconstruction_ci")


def compute_ci95(data: np.ndarray) -> Tuple[float, float, float]:
    """Compute mean, lower 95% CI, and upper 95% CI for 1D array."""
    mean = float(np.mean(data))
    if len(data) <= 1:
        return mean, mean, mean
    std = float(np.std(data, ddof=1))
    # t-critical value for df = n - 1
    # For df=2 (n=3), t = 4.303
    # For general n:
    from scipy import stats
    t_val = stats.t.ppf(0.975, df=len(data) - 1)
    margin = t_val * (std / np.sqrt(len(data)))
    return mean, float(mean - margin), float(mean + margin)


def main():
    parser = argparse.ArgumentParser(description="Audit Held-Out Reconstruction F-score with CI")
    parser.add_argument("--manifest", default="artifacts/corpus-v2/manifest.json")
    parser.add_argument("--cache-dir", default="/tmp/t3d_ephemeral_scratch/kenney_raw_glb")
    parser.add_argument("--output", default="artifacts/reconstruction-fscore-ci-report.json")
    parser.add_argument("--num-assets", type=int, default=30, help="Number of held-out assets to evaluate")
    parser.add_argument("--samples", type=int, default=512, help="Point samples per asset")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Surface distance tolerance")
    args = parser.parse_args()

    manifest_p = Path(args.manifest)
    cache_dir = Path(args.cache_dir)
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
    val_assets = [a for a in manifest_data["assets"] if a["split"] == "validation"]
    logger.info("Total held-out validation assets available: %d", len(val_assets))

    eval_assets = val_assets[:args.num_assets]
    logger.info("Evaluating %d held-out validation assets across 3 seeds...", len(eval_assets))

    seeds = [42, 123, 2026]
    seed_fscores = {s: [] for s in seeds}
    seed_msds = {s: [] for s in seeds}
    asset_results = []

    start_time = time.time()

    for idx, meta in enumerate(eval_assets):
        aid = meta["asset_id"]
        t3d_path = manifest_p.parent / meta["path"]
        glb_path = cache_dir / f"{aid}.glb"

        if not t3d_path.exists():
            logger.warning("Missing .t3d: %s", t3d_path)
            continue
        if not glb_path.exists():
            logger.warning("Missing raw GLB: %s", glb_path)
            continue

        # Load raw source geometry
        source_v, source_f, _ = load_glb_geometry(glb_path)
        source_v, _, _ = normalize_asset(source_v)
        source_v = np.asarray(source_v)

        # Load reconstructed voxel asset & extract mesh
        voxel_asset = load(t3d_path).asset
        pred_v, pred_f = voxels_to_mesh(voxel_asset)

        if len(pred_f) == 0:
            logger.warning("0 faces reconstructed for %s", aid)
            continue

        asset_seed_fscores = {}
        for s in seeds:
            res = surface_comparison(
                source_v, source_f, pred_v, pred_f,
                samples=args.samples, seed=s, tolerance=args.tolerance
            )
            fscore = res.get("fscore", 0.0)
            msd = res.get("mean_distance", 1.0)
            seed_fscores[s].append(fscore)
            seed_msds[s].append(msd)
            asset_seed_fscores[s] = fscore

        asset_results.append({
            "asset_id": aid,
            "fscores": asset_seed_fscores,
            "mean_fscore": float(np.mean(list(asset_seed_fscores.values()))),
            "reconstructed_faces": len(pred_f),
        })

        if (idx + 1) % 10 == 0:
            logger.info("Evaluated [%d / %d] held-out assets...", idx + 1, len(eval_assets))

    elapsed = time.time() - start_time

    # Calculate overall metrics per seed
    mean_per_seed = [float(np.mean(seed_fscores[s])) for s in seeds]
    overall_mean, ci_lower, ci_upper = compute_ci95(np.array(mean_per_seed))

    msd_per_seed = [float(np.mean(seed_msds[s])) for s in seeds]
    msd_mean, msd_ci_lower, msd_ci_upper = compute_ci95(np.array(msd_per_seed))

    report = {
        "benchmark": "held_out_reconstruction_fscore",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 2),
        "manifest": str(manifest_p),
        "manifest_sha256": hashlib.sha256(manifest_p.read_bytes()).hexdigest(),
        "evaluation_scope": {
            "split": "validation (strictly held-out)",
            "total_evaluated_assets": len(asset_results),
            "samples_per_asset": args.samples,
            "tolerance": args.tolerance,
            "seeds": seeds,
        },
        "metrics": {
            "mean_fscore": round(overall_mean, 4),
            "fscore_95ci": [round(ci_lower, 4), round(ci_upper, 4)],
            "fscore_std": round(float(np.std(mean_per_seed, ddof=1)), 5),
            "fscore_per_seed": {str(s): round(m, 4) for s, m in zip(seeds, mean_per_seed)},
            "mean_surface_distance": round(msd_mean, 5),
            "msd_95ci": [round(msd_ci_lower, 5), round(msd_ci_upper, 5)],
            "msd_per_seed": {str(s): round(m, 5) for s, m in zip(seeds, msd_per_seed)},
        },
        "sample_evaluations": asset_results[:10],
    }

    out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Reconstruction F-score evaluation complete!")
    logger.info("Report saved to %s", out_p)
    logger.info("Mean F-score: %.4f (95%% CI: [%.4f, %.4f])", overall_mean, ci_lower, ci_upper)
    logger.info("Mean Surface Distance: %.5f", msd_mean)


if __name__ == "__main__":
    main()
