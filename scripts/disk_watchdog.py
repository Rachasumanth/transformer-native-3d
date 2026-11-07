#!/usr/bin/env python3
"""T3D Local Disk Watchdog & Ephemeral Workspace Scratch Manager.

Monitors local disk usage against the 35 GB budget.
Configures Hugging Face, PyTorch, and general temporary cache paths
to /tmp/t3d_ephemeral_scratch and provides automated eviction and pause controls.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

# Ensure unbuffered / line-buffered output for daemon logging when piped to nohup
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

logger = logging.getLogger("disk_watchdog")

# Paths and configuration
EPHEMERAL_SCRATCH = Path("/tmp/t3d_ephemeral_scratch")
DEFAULT_SCRATCH_DIR = str(EPHEMERAL_SCRATCH)
MAX_ALLOWED_GB = 35.0
DEFAULT_INTERVAL_SECONDS = 15.0
PID_FILE_PATH = "/tmp/disk_watchdog.pid"
PAUSE_FLAG_FILE = "/tmp/t3d_download_paused"
SCRATCH_PAUSE_FLAG = str(EPHEMERAL_SCRATCH / ".download_paused")
ENV_SCRIPT_PATH = "/tmp/t3d_env.sh"


class DiskBudgetExceededError(RuntimeError):
    """Raised when disk workspace usage exceeds allowable budget even after eviction."""
    pass


def init_ephemeral_env(scratch_dir: str = DEFAULT_SCRATCH_DIR) -> Dict[str, str]:
    """Redirect environment variables to ephemeral scratch directory.

    Exports HF_HOME, TORCH_HOME, TMPDIR to /tmp/t3d_ephemeral_scratch.
    """
    scratch_path = Path(scratch_dir)
    scratch_path.mkdir(parents=True, exist_ok=True)

    # Subdirectories for organized caching
    (scratch_path / "huggingface").mkdir(parents=True, exist_ok=True)
    (scratch_path / "torch").mkdir(parents=True, exist_ok=True)
    (scratch_path / "ring_buffer").mkdir(parents=True, exist_ok=True)

    os.environ["TMPDIR"] = str(scratch_path)
    os.environ["HF_HOME"] = str(scratch_path)
    os.environ["TORCH_HOME"] = str(scratch_path)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(scratch_path / "huggingface")

    env_vars = {
        "TMPDIR": str(scratch_path),
        "HF_HOME": str(scratch_path),
        "TORCH_HOME": str(scratch_path),
        "HUGGINGFACE_HUB_CACHE": str(scratch_path / "huggingface"),
    }

    try:
        with open(ENV_SCRIPT_PATH, "w") as f:
            for k, v in env_vars.items():
                f.write(f"export {k}={shlex.quote(v)}\n")
    except Exception as e:
        logger.debug(f"Could not write {ENV_SCRIPT_PATH}: {e}")

    return env_vars


# Export environment variables on module import
init_ephemeral_env()


def get_disk_usage(path: str = "/") -> Tuple[float, float, float, float]:
    """Return disk usage metrics (used_gb, free_gb, total_gb, percent_used) for path."""
    usage = shutil.disk_usage(path)
    gb_factor = 1024.0 ** 3
    used_gb = usage.used / gb_factor
    free_gb = usage.free / gb_factor
    total_gb = usage.total / gb_factor
    percent_used = (usage.used / usage.total) * 100.0 if usage.total > 0 else 0.0
    return used_gb, free_gb, total_gb, percent_used


def pause_downloaders() -> None:
    """Sets pause flag files to signal active downloaders to suspend transfers."""
    try:
        with open(PAUSE_FLAG_FILE, "w") as f:
            f.write(f"{time.time()}\n")
    except Exception as e:
        logger.warning(f"Could not touch pause flag {PAUSE_FLAG_FILE}: {e}")

    try:
        if EPHEMERAL_SCRATCH.exists():
            with open(SCRATCH_PAUSE_FLAG, "w") as f:
                f.write(f"{time.time()}\n")
    except Exception as e:
        logger.warning(f"Could not touch pause flag {SCRATCH_PAUSE_FLAG}: {e}")


def resume_downloaders() -> None:
    """Removes pause flag files to signal downloaders can resume."""
    for flag_path in (PAUSE_FLAG_FILE, SCRATCH_PAUSE_FLAG):
        if os.path.exists(flag_path):
            try:
                os.remove(flag_path)
            except Exception as e:
                logger.warning(f"Could not remove pause flag {flag_path}: {e}")


def is_download_paused() -> bool:
    """Returns True if any downloader pause flag is active."""
    return os.path.exists(PAUSE_FLAG_FILE) or os.path.exists(SCRATCH_PAUSE_FLAG)


def evict_ephemeral_cache(scratch_dir: str = DEFAULT_SCRATCH_DIR) -> int:
    """Safely purge all contents in the ephemeral scratch directory.

    Equivalent to `rm -rf /tmp/t3d_ephemeral_scratch/*`.
    Returns the number of top-level items cleared.
    """
    logger.warning("Initiating emergency cache eviction in %s...", scratch_dir)
    scratch_path = Path(scratch_dir)
    removed_count = 0

    if scratch_path.exists():
        for item in scratch_path.iterdir():
            try:
                if item.is_dir() and not item.is_symlink():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
                removed_count += 1
            except Exception as e:
                logger.error("Failed to remove %s: %s", item, e)

    # Shell invocation fallback to guarantee full purge of hidden files / deep paths
    try:
        subprocess.run(
            f"rm -rf {shlex.quote(scratch_dir)}/*",
            shell=True,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.debug(f"rm -rf fallback warning: {e}")

    init_ephemeral_env(scratch_dir)
    used_gb, _, _, _ = get_disk_usage("/")
    logger.info("Cache eviction completed. Current used disk: %.2f GB", used_gb)
    return removed_count


def assert_disk_budget(
    max_gb: float = MAX_ALLOWED_GB,
    path: str = "/",
    auto_evict: bool = True,
    raise_on_error: bool = True,
) -> float:
    """Enforce disk budget assertion.

    Checks if used workspace exceeds max_gb (default 35.0 GB).
    If exceeded, pauses downloaders and triggers cache eviction (`rm -rf /tmp/t3d_ephemeral_scratch/*`).
    Returns current used GB and raises or warns appropriately.

    Args:
        max_gb: Hard limit in gigabytes (default 35.0).
        path: Monitored filesystem path (default "/").
        auto_evict: If True, triggers cache eviction on threshold breach.
        raise_on_error: If True, raises DiskBudgetExceededError if threshold is breached.

    Returns:
        float: Current used gigabytes.
    """
    init_ephemeral_env()
    used_gb, free_gb, total_gb, percent = get_disk_usage(path)

    if used_gb >= max_gb:
        msg = f"DISK USAGE ALERT: {used_gb:.2f} GB used exceeds ceiling {max_gb:.2f} GB (Total: {total_gb:.2f} GB)"
        warnings.warn(msg, RuntimeWarning)
        logger.warning(msg)

        if auto_evict:
            pause_downloaders()
            evict_ephemeral_cache()
            post_used_gb, _, _, _ = get_disk_usage(path)

            if post_used_gb >= max_gb:
                err_msg = f"Disk usage {post_used_gb:.2f} GB still exceeds {max_gb:.2f} GB after cache eviction!"
                logger.critical(err_msg)
                if raise_on_error:
                    raise DiskBudgetExceededError(err_msg)
                else:
                    warnings.warn(err_msg, RuntimeWarning)
                return post_used_gb
            else:
                logger.info("Disk usage restored to %.2f GB <= %.2f GB budget. Resuming downloaders.", post_used_gb, max_gb)
                resume_downloaders()
                return post_used_gb
        else:
            if raise_on_error:
                raise DiskBudgetExceededError(
                    f"Disk usage {used_gb:.2f} GB exceeds maximum budget of {max_gb:.2f} GB!"
                )
            else:
                warnings.warn(msg, RuntimeWarning)
            return used_gb

    return used_gb


def run_daemon(
    interval: float = DEFAULT_INTERVAL_SECONDS,
    max_gb: float = MAX_ALLOWED_GB,
    path: str = "/",
    pid_file: str = PID_FILE_PATH,
) -> None:
    """Run persistent watchdog daemon loop.

    Logs disk usage periodically (every 10-30s) and evicts
    /tmp/t3d_ephemeral_scratch/* whenever used workspace exceeds max_gb.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [DISK-WATCHDOG] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    init_ephemeral_env()
    pid = os.getpid()

    # Record PID file
    try:
        with open(pid_file, "w") as f:
            f.write(f"{pid}\n")
    except Exception as e:
        logger.warning("Could not write PID file %s: %s", pid_file, e)

    logging.info(
        "Starting T3D Disk Watchdog Daemon [PID %d]. Max Budget: %.1f GB, Polling Interval: %.1fs, Monitored Path: '%s'",
        pid, max_gb, interval, path
    )

    running = True

    def _term_handler(signum, frame):
        nonlocal running
        logging.info("Disk Watchdog received signal %d. Shutting down cleanly...", signum)
        running = False

    signal.signal(signal.SIGTERM, _term_handler)
    signal.signal(signal.SIGINT, _term_handler)

    try:
        while running:
            try:
                used_gb, free_gb, total_gb, pct = get_disk_usage(path)
                if used_gb >= max_gb:
                    logging.warning(
                        "Usage %.2f GB >= %.2f GB budget limit (%.1f%% of %.2f GB)! Triggering pause & eviction.",
                        used_gb, max_gb, pct, total_gb
                    )
                    pause_downloaders()
                    evict_ephemeral_cache()
                    post_used, _, _, post_pct = get_disk_usage(path)
                    if post_used < max_gb:
                        logging.info("Eviction restored usage to %.2f GB (%.1f%%). Resuming downloaders.", post_used, post_pct)
                        resume_downloaders()
                    else:
                        logging.critical("Usage remains critical at %.2f GB despite cache eviction!", post_used)
                else:
                    logging.info(
                        "Disk Status: Used: %.2f GB / %.2f GB (%.1f%%) | Free: %.2f GB | Budget: %.1f GB [OK]",
                        used_gb, total_gb, pct, free_gb, max_gb
                    )
            except Exception as e:
                logging.error("Exception in watchdog daemon loop: %s", e, exc_info=True)

            # Sleep in 0.5s increments to respond promptly to termination signals
            slices = max(1, int(interval * 2))
            slice_dur = interval / slices
            for _ in range(slices):
                if not running:
                    break
                time.sleep(slice_dur)
    finally:
        if os.path.exists(pid_file):
            try:
                os.remove(pid_file)
            except Exception:
                pass
        logging.info("Disk Watchdog daemon [PID %d] exited.", pid)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [DISK-WATCHDOG] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="T3D Disk Watchdog & Ephemeral Workspace Manager")
    parser.add_argument("--daemon", action="store_true", help="Run continuously as a background watchdog daemon")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SECONDS, help="Check interval in seconds (default: 15)")
    parser.add_argument("--max-gb", type=float, default=MAX_ALLOWED_GB, help="Max workspace disk budget in GB (default: 35.0)")
    parser.add_argument("--path", type=str, default="/", help="Filesystem path to monitor (default: /)")
    parser.add_argument("--check", action="store_true", help="Perform a one-time disk budget check and exit")
    parser.add_argument("--evict", action="store_true", help="Manually trigger scratch cache eviction and exit")
    parser.add_argument("--status", action="store_true", help="Display current disk usage and budget status")

    args = parser.parse_args()

    if args.evict:
        count = evict_ephemeral_cache()
        print(f"Evicted {count} items from {DEFAULT_SCRATCH_DIR}")
        return

    if args.status:
        used_gb, free_gb, total_gb, pct = get_disk_usage(args.path)
        print(f"Path:        {args.path}")
        print(f"Total:       {total_gb:.2f} GB")
        print(f"Used:        {used_gb:.2f} GB ({pct:.1f}%)")
        print(f"Free:        {free_gb:.2f} GB")
        print(f"Budget:      {args.max_gb:.2f} GB")
        print(f"Scratch:     {DEFAULT_SCRATCH_DIR} (exists: {EPHEMERAL_SCRATCH.exists()})")
        print(f"Paused:      {is_download_paused()}")
        return

    if args.check:
        used = assert_disk_budget(max_gb=args.max_gb, path=args.path, raise_on_error=False)
        print(f"Disk check: {used:.2f} GB used (ceiling: {args.max_gb:.2f} GB)")
        return

    if args.daemon:
        run_daemon(interval=args.interval, max_gb=args.max_gb, path=args.path)
    else:
        # Default behavior: run budget check and print status
        used, free, total, pct = get_disk_usage(args.path)
        print(f"Disk: {used:.2f} GB used / {total:.2f} GB total ({pct:.1f}%), {free:.2f} GB free")
        assert_disk_budget(max_gb=args.max_gb, path=args.path)
        print("Disk budget check PASSED.")


if __name__ == "__main__":
    main()
