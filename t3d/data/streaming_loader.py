"""Streaming ring-buffer dataset and DataLoader for T3D pre-encoded latents.

Streams latent micro-batches directly from GitHub Releases or local cache with a
strict 4 GB ring-buffer ceiling to conform to local storage limits.
"""
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import threading
import time
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

logger = logging.getLogger(__name__)

DEFAULT_RING_BUFFER_DIR = Path("/tmp/t3d_ephemeral_scratch/ring_buffer")
DEFAULT_MAX_CACHE_GB = 4.0
DEFAULT_LATENT_REPO = "Rachasumanth/t3d-latents-cache"
DEFAULT_LATENT_TAG = "latents-v1"


class LatentsRingBuffer:
    """Manages an ephemeral local ring-buffer cache with a strict storage limit (default 4 GB).

    Downloads latent archives (.npz) on-demand from remote GitHub Releases via `gh` CLI,
    and evicts the oldest files in FIFO order when the local ring-buffer exceeds max_cache_gb.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = DEFAULT_RING_BUFFER_DIR,
        max_cache_gb: float = DEFAULT_MAX_CACHE_GB,
        repo: str = DEFAULT_LATENT_REPO,
        tag: str = DEFAULT_LATENT_TAG,
    ):
        self.cache_dir = Path(cache_dir)
        self.max_cache_bytes = int(max_cache_gb * (1024 ** 3))
        self.repo = repo
        self.tag = tag
        self.lock = threading.Lock()

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_size_bytes(self) -> int:
        """Calculate total bytes currently used by files in cache_dir."""
        total = 0
        if not self.cache_dir.exists():
            return 0
        for entry in self.cache_dir.iterdir():
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
        return total

    def evict_if_needed(self, required_bytes: int = 0):
        """Evict oldest accessed/modified files until cache fits within budget."""
        with self.lock:
            current_size = self.get_cache_size_bytes()
            if current_size + required_bytes <= self.max_cache_bytes:
                return

            files = []
            for entry in self.cache_dir.iterdir():
                if entry.is_file() and not entry.name.startswith("."):
                    try:
                        st = entry.stat()
                        files.append((st.st_atime, st.st_size, entry))
                    except OSError:
                        pass

            # Sort by access time ascending (oldest first)
            files.sort(key=lambda x: x[0])

            for _, file_size, path in files:
                if current_size + required_bytes <= self.max_cache_bytes:
                    break
                try:
                    path.unlink(missing_ok=True)
                    current_size -= file_size
                    logger.info("Evicted ring-buffer shard: %s (freed %.2f MB)", path.name, file_size / (1024 * 1024))
                except OSError as e:
                    logger.warning("Failed to evict %s: %s", path, e)

    def get_or_fetch(
        self,
        filename: str,
        repo: Optional[str] = None,
        tag: Optional[str] = None,
        local_fallback: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Ensure the specified shard is cached locally and return its path.

        If not cached, downloads it via `gh release download` or copies from local fallback.
        """
        target_path = self.cache_dir / filename
        if target_path.exists() and target_path.stat().st_size > 0:
            # Touch access time for FIFO/LRU tracking
            try:
                target_path.touch()
            except OSError:
                pass
            return target_path

        # Check local fallback first before remote download
        if local_fallback and Path(local_fallback).exists():
            fallback_p = Path(local_fallback)
            self.evict_if_needed(fallback_p.stat().st_size)
            shutil.copy2(str(fallback_p), str(target_path))
            return target_path

        target_repo = repo or self.repo
        target_tag = tag or self.tag

        with self.lock:
            if target_path.exists() and target_path.stat().st_size > 0:
                return target_path

            self.evict_if_needed(50 * 1024 * 1024)  # Reserve ~50MB estimated shard buffer

            cmd = [
                "gh", "release", "download", target_tag,
                "-p", filename,
                "-R", target_repo,
                "-D", str(self.cache_dir),
                "--clobber",
            ]
            logger.info("Downloading remote shard %s from %s release %s...", filename, target_repo, target_tag)
            res = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
            if res.returncode != 0:
                logger.error("Failed downloading %s: %s", filename, res.stderr)
                raise FileNotFoundError(
                    f"Remote shard {filename} could not be downloaded from {target_repo}@{target_tag}: {res.stderr}"
                )

            if not target_path.exists():
                raise FileNotFoundError(f"Expected shard {filename} not found after download.")

            return target_path


class StreamingLatentsDataset(IterableDataset):
    """PyTorch IterableDataset streaming pre-encoded SC-VAE latents through the ring-buffer.

    Yields micro-batches/samples containing:
      - x_0_occ: Ground-truth coarse occupancy grid {-1, +1}
      - x_0_lat: Ground-truth continuous SC-VAE latent tokens
      - c_raw: Corrupted conditioning features [occ, latents]
      - hermite_normals: Hermite voxel surface normals for Dual Contouring
      - surface_ratio: Surface voxel intersection ratio per coarse cell
    """

    def __init__(
        self,
        manifest: Union[str, Path, List[dict]],
        ring_buffer: Optional[LatentsRingBuffer] = None,
        shuffle_shards: bool = True,
        shuffle_samples: bool = True,
        infinite: bool = False,
        required_keys: Tuple[str, ...] = (
            "x_0_occ", "x_0_lat", "c_raw", "hermite_normals", "surface_ratio"
        ),
        return_torch: bool = True,
    ):
        super().__init__()
        self.ring_buffer = ring_buffer or LatentsRingBuffer()
        self.shuffle_shards = shuffle_shards
        self.shuffle_samples = shuffle_samples
        self.infinite = infinite
        self.required_keys = required_keys
        self.return_torch = return_torch

        # Parse manifest entries
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
            self.shards = []
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.shards.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        elif isinstance(manifest, list):
            self.shards = list(manifest)
        else:
            raise TypeError(f"Unsupported manifest type: {type(manifest)}")

    def _get_worker_shards(self) -> List[dict]:
        """Split shards across multi-worker DataLoader workers."""
        worker_info = get_worker_info()
        if worker_info is None:
            # Single process
            shards = list(self.shards)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shards = [s for i, s in enumerate(self.shards) if i % num_workers == worker_id]

        if self.shuffle_shards:
            random.shuffle(shards)
        return shards

    def __iter__(self) -> Iterator[Dict[str, Union[np.ndarray, torch.Tensor]]]:
        """Iterate over samples loaded from shards."""
        while True:
            worker_shards = self._get_worker_shards()
            if not worker_shards:
                return

            for shard_meta in worker_shards:
                latent_fn = shard_meta.get("latent_shard_filename")
                if not latent_fn:
                    continue

                repo = shard_meta.get("latent_shard_repo", self.ring_buffer.repo)
                tag = shard_meta.get("latent_shard_tag", self.ring_buffer.tag)
                local_path = shard_meta.get("local_npz_path")

                try:
                    shard_file = self.ring_buffer.get_or_fetch(
                        filename=latent_fn,
                        repo=repo,
                        tag=tag,
                        local_fallback=local_path,
                    )
                except Exception as e:
                    logger.warning("Error fetching shard %s: %s", latent_fn, e)
                    continue

                # Load .npz archive
                try:
                    with np.load(str(shard_file)) as data:
                        # Extract arrays
                        extracted = {}
                        num_samples = None
                        for key in self.required_keys:
                            if key in data:
                                arr = data[key]
                                extracted[key] = arr
                                if num_samples is None:
                                    num_samples = arr.shape[0] if arr.ndim > 0 else 1
                                else:
                                    num_samples = min(num_samples, arr.shape[0] if arr.ndim > 0 else 1)

                        if not extracted or num_samples is None or num_samples == 0:
                            continue

                        sample_indices = list(range(num_samples))
                        if self.shuffle_samples:
                            random.shuffle(sample_indices)

                        for idx in sample_indices:
                            sample = {}
                            for key in self.required_keys:
                                if key in extracted:
                                    val = extracted[key][idx]
                                    if self.return_torch:
                                        val = torch.from_numpy(np.ascontiguousarray(val))
                                    sample[key] = val
                            yield sample

                except Exception as e:
                    logger.warning("Failed to unpack shard %s: %s", shard_file, e)
                    continue

            if not self.infinite:
                break


def create_streaming_dataloader(
    manifest: Union[str, Path, List[dict]],
    batch_size: int = 4,
    num_workers: int = 0,
    ring_buffer: Optional[LatentsRingBuffer] = None,
    shuffle_shards: bool = True,
    infinite: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a PyTorch DataLoader streaming pre-encoded latents with ring-buffer caching."""
    dataset = StreamingLatentsDataset(
        manifest=manifest,
        ring_buffer=ring_buffer,
        shuffle_shards=shuffle_shards,
        shuffle_samples=True,
        infinite=infinite,
        return_torch=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
