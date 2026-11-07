#!/usr/bin/env python3
"""T3D Local Disk Watchdog & Cache Controller.

Enforces a strict 35 GB active ceiling (15 GB safety buffer on 50 GB partition).
Redirects ephemeral scratch directories and auto-evicts on threshold breach.
"""

import os
import shutil
import sys
import time
import logging
from pathlib import Path

# Ephemeral scratch paths
EPHEMERAL_SCRATCH = Path("/tmp/t3d_ephemeral_scratch")
RING_BUFFER_DIR = EPHEMERAL_SCRATCH / "ring_buffer"
HF_CACHE_DIR = EPHEMERAL_SCRATCH / "huggingface"
TORCH_CACHE_DIR = EPHEMERAL_SCRATCH / "torch"

MAX_ALLOWED_GB = 35.0


def init_ephemeral_env():
    """Redirect environment variables to ephemeral scratch."""
    EPHEMERAL_SCRATCH.mkdir(parents=True, exist_ok=True)
    RING_BUFFER_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    os.environ["TMPDIR"] = str(EPHEMERAL_SCRATCH)
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR)
    os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)


def get_disk_usage(path="/"):
    """Return (used_gb, free_gb, total_gb, percent_used)."""
    usage = shutil.disk_usage(path)
    used_gb = usage.used / (1024 ** 3)
    free_gb = usage.free / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)
    percent_used = (usage.used / usage.total) * 100.0
    return used_gb, free_gb, total_gb, percent_used


def evict_ephemeral_cache():
    """Safely purge all contents in the ephemeral scratch directory."""
    logging.warning("Initiating emergency cache eviction in %s...", EPHEMERAL_SCRATCH)
    if EPHEMERAL_SCRATCH.exists():
        for item in EPHEMERAL_SCRATCH.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            except Exception as e:
                logging.error("Failed to remove %s: %s", item, e)
    init_ephemeral_env()
    used_gb, _, _, _ = get_disk_usage("/")
    logging.info("Cache eviction completed. Current used disk: %.2f GB", used_gb)


def assert_disk_budget(max_gb=MAX_ALLOWED_GB, auto_evict=True):
    """Enforce disk budget assertion.

    Args:
        max_gb: Hard limit in gigabytes (default 35.0).
        auto_evict: If True, triggers cache eviction before raising or failing.

    Returns:
        float: Current used gigabytes.
    """
    used_gb, free_gb, total_gb, percent = get_disk_usage("/")
    if used_gb >= max_gb:
        logging.error(
            "DISK USAGE CRITICAL: %.2f GB used exceeds ceiling %.2f GB (Total: %.2f GB)",
            used_gb, max_gb, total_gb
        )
        if auto_evict:
            evict_ephemeral_cache()
            used_gb, _, _, _ = get_disk_usage("/")
            if used_gb >= max_gb:
                raise RuntimeError(
                    f"Disk usage {used_gb:.2f} GB still exceeds {max_gb:.2f} GB after eviction!"
                )
        else:
            raise RuntimeError(
                f"Disk usage {used_gb:.2f} GB exceeds maximum budget of {max_gb:.2f} GB!"
            )
    return used_gb


def run_daemon(interval=15.0):
    """Run persistent watchdog loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [DISK-WATCHDOG] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Starting T3D Disk Watchdog Daemon. Max Budget: %.1f GB, Polling: %.1fs", MAX_ALLOWED_GB, interval)
    init_ephemeral_env()

    while True:
        try:
            used_gb, free_gb, total_gb, pct = get_disk_usage("/")
            logging.info("Disk Status: Used: %.2f GB / %.2f GB (%.1f%%) | Free: %.2f GB", used_gb, total_gb, pct, free_gb)
            if used_gb >= MAX_ALLOWED_GB:
                logging.warning("Usage %.2f GB >= %.2f GB budget limit! Triggering eviction.", used_gb, MAX_ALLOWED_GB)
                evict_ephemeral_cache()
        except Exception as e:
            logging.error("Exception in watchdog daemon loop: %s", e)
        time.sleep(interval)


if __name__ == "__main__":
    init_ephemeral_env()
    if "--daemon" in sys.argv:
        run_daemon()
    else:
        used, free, total, pct = get_disk_usage("/")
        print(f"Disk: {used:.2f} GB used / {total:.2f} GB total ({pct:.1f}%), {free:.2f} GB free")
        assert_disk_budget()
        print("Disk budget check PASSED.")
