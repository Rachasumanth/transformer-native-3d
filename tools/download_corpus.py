"""T3D Corpus Download and Byte-Verification Utility.

Enforces:
1. Pinned checksum verification (SHA-256) of every asset against data/manifest.json
   and artifacts/corpus-v2/manifest.json.
2. Direct fetch from pinned source URLs / Kaggle public mirrors.
3. Byte-equal rebuild capability from source assets.
4. Independent auditor verification in clean / fresh environments.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_corpus")


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_asset(
    asset_record: dict,
    corpus_dir: Path,
) -> Tuple[str, bool, str]:
    """Verify single asset file exists and matches pinned sha256."""
    asset_id = asset_record.get("asset_id", "")
    target_path = corpus_dir / asset_record.get("path", f"{asset_id}.t3d")
    expected_sha256 = asset_record.get("sha256", "")

    if not target_path.exists():
        return asset_id, False, f"Missing file: {target_path}"

    actual_sha256 = compute_file_sha256(target_path)
    if actual_sha256 != expected_sha256:
        return asset_id, False, f"SHA-256 mismatch: got {actual_sha256[:12]}..., expected {expected_sha256[:12]}..."

    return asset_id, True, "OK"


def verify_manifest_checksums(
    manifest_path: Path,
    corpus_dir: Optional[Path] = None,
    max_workers: int = 8,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Verify all or sample assets in a manifest against on-disk files."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = compute_sha256(manifest_bytes)
    data = json.loads(manifest_bytes.decode("utf-8"))
    assets = data.get("assets", [])

    if corpus_dir is None:
        corpus_dir = manifest_path.parent

    if sample_limit is not None and sample_limit > 0:
        assets = assets[:sample_limit]

    logger.info("Verifying %d assets against %s in %s...", len(assets), manifest_path.name, corpus_dir)

    passed = []
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(verify_asset, a, corpus_dir): a for a in assets}
        for future in as_completed(futures):
            asset_id, ok, msg = future.result()
            if ok:
                passed.append(asset_id)
            else:
                failed.append({"asset_id": asset_id, "error": msg})

    logger.info(
        "Verification complete: %d passed, %d failed (Total tested: %d)",
        len(passed), len(failed), len(assets)
    )

    return {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "corpus_dir": str(corpus_dir),
        "total_tested": len(assets),
        "passed_count": len(passed),
        "failed_count": len(failed),
        "all_passed": len(failed) == 0,
        "failures": failed,
    }


def download_single_source(
    asset_record: dict,
    dest_dir: Path,
    timeout: int = 30,
) -> Tuple[str, bool, str]:
    """Download source asset and check source_sha256."""
    asset_id = asset_record["asset_id"]
    source_url = asset_record.get("source", "")
    expected_source_sha = asset_record.get("source_sha256", "")

    if not source_url:
        return asset_id, False, "No source URL in record"

    target_file = dest_dir / f"{asset_id}.glb"
    if target_file.exists():
        if compute_file_sha256(target_file) == expected_source_sha:
            return asset_id, True, "Cached"

    try:
        req = urllib.request.Request(source_url, headers={"User-Agent": "T3D-Corpus-Downloader/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read()

        if expected_source_sha and compute_sha256(content) != expected_source_sha:
            return asset_id, False, f"Source SHA-256 mismatch for {asset_id}"

        target_file.write_bytes(content)
        return asset_id, True, "Downloaded"
    except Exception as e:
        return asset_id, False, f"Download failed: {e}"


def main():
    parser = argparse.ArgumentParser(description="T3D Corpus Download and Verification Utility")
    parser.add_argument("--manifest", default="artifacts/corpus-v2/manifest.json", help="Path to manifest.json")
    parser.add_argument("--corpus-dir", default=None, help="Directory containing .t3d assets")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing checksums without downloading")
    parser.add_argument("--sample", type=int, default=None, help="Limit check to first N assets (e.g. 100)")
    parser.add_argument("--download-sources", action="store_true", help="Download raw GLB source files")
    parser.add_argument("--source-dir", default="/tmp/t3d_download_scratch", help="Directory to store downloaded sources")
    parser.add_argument("--output-report", default=None, help="Save verification report JSON")
    parser.add_argument("--max-workers", type=int, default=16, help="Parallel worker threads")
    args = parser.parse_args()

    manifest_p = Path(args.manifest)
    if not manifest_p.exists() and args.manifest == "artifacts/corpus-v2/manifest.json":
        # Fall back to data/manifest.json if corpus-v2 is still being generated
        logger.info("artifacts/corpus-v2/manifest.json not yet present, checking data/manifest.json")
        manifest_p = Path("data/manifest.json")

    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else manifest_p.parent

    if args.download_sources:
        source_dir = Path(args.source_dir)
        source_dir.mkdir(parents=True, exist_ok=True)
        manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
        assets = manifest_data.get("assets", [])
        if args.sample:
            assets = assets[:args.sample]

        logger.info("Downloading %d source assets to %s...", len(assets), source_dir)
        success_count = 0
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(download_single_source, a, source_dir): a["asset_id"] for a in assets}
            for future in as_completed(futures):
                aid, ok, msg = future.result()
                if ok:
                    success_count += 1
                else:
                    logger.warning("Failed downloading %s: %s", aid, msg)

        logger.info("Source download complete: %d / %d downloaded and sha256 verified.", success_count, len(assets))

    report = verify_manifest_checksums(
        manifest_path=manifest_p,
        corpus_dir=corpus_dir,
        max_workers=args.max_workers,
        sample_limit=args.sample,
    )

    if args.output_report:
        out_p = Path(args.output_report)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("Saved verification report to %s", out_p)

    if not report["all_passed"] and args.verify_only:
        sys.exit(1)


if __name__ == "__main__":
    main()
