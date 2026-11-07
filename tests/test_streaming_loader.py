"""Unit tests for T3D Streaming Ring-Buffer DataLoader."""

import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from t3d.data.streaming_loader import (
    LatentsRingBuffer,
    StreamingLatentsDataset,
    create_streaming_dataloader,
)


class TestStreamingLoader(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.test_root = Path(self.tmp_dir.name)
        self.cache_dir = self.test_root / "ring_buffer"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _create_dummy_npz(self, path: Path, num_samples: int = 4):
        """Create a synthetic .npz latent bundle matching SC-VAE pre-encoder output."""
        N = 64  # coarse cells
        C = 16  # latent channels
        data = {
            "x_0_occ": np.random.choice([-1.0, 1.0], size=(num_samples, N)).astype(np.float32),
            "x_0_lat": np.random.randn(num_samples, N, C).astype(np.float32),
            "c_raw": np.random.randn(num_samples, N, C + 1).astype(np.float32),
            "hermite_normals": np.random.randn(num_samples, N, 3).astype(np.float32),
            "surface_ratio": np.random.uniform(0.0, 1.0, size=(num_samples, N)).astype(np.float32),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)
        return path

    def test_ring_buffer_eviction(self):
        """Verify ring buffer evicts oldest files when exceeding maximum allowed bytes."""
        # Set tiny limit of 50 KB
        buffer = LatentsRingBuffer(
            cache_dir=self.cache_dir,
            max_cache_gb=0.00005,  # ~50 KB
        )

        f1 = self.cache_dir / "shard_1.bin"
        f2 = self.cache_dir / "shard_2.bin"
        f3 = self.cache_dir / "shard_3.bin"

        # Write 30 KB to f1
        f1.write_bytes(b"x" * 30000)
        os.utime(str(f1), (1000, 1000))

        # Write 30 KB to f2
        f2.write_bytes(b"y" * 30000)
        os.utime(str(f2), (2000, 2000))

        # Current size is 60 KB > 50 KB limit
        buffer.evict_if_needed(0)

        # Oldest file f1 should be evicted, f2 retained
        self.assertFalse(f1.exists())
        self.assertTrue(f2.exists())

    def test_ring_buffer_local_fallback(self):
        """Verify ring buffer copies from local fallback when available."""
        buffer = LatentsRingBuffer(cache_dir=self.cache_dir, max_cache_gb=1.0)
        local_src = self.test_root / "source_shard.npz"
        self._create_dummy_npz(local_src, num_samples=2)

        fetched = buffer.get_or_fetch(
            filename="cached_shard.npz",
            local_fallback=local_src,
        )

        self.assertTrue(fetched.exists())
        self.assertEqual(fetched.name, "cached_shard.npz")
        self.assertEqual(fetched.stat().st_size, local_src.stat().st_size)

    def test_streaming_dataset_iteration(self):
        """Verify StreamingLatentsDataset correctly iterates and unpacks latent fields."""
        shard1_path = self.test_root / "shard_0001.npz"
        shard2_path = self.test_root / "shard_0002.npz"
        self._create_dummy_npz(shard1_path, num_samples=3)
        self._create_dummy_npz(shard2_path, num_samples=2)

        manifest = [
            {"latent_shard_filename": "shard_0001.npz", "local_npz_path": str(shard1_path)},
            {"latent_shard_filename": "shard_0002.npz", "local_npz_path": str(shard2_path)},
        ]

        buffer = LatentsRingBuffer(cache_dir=self.cache_dir, max_cache_gb=1.0)
        dataset = StreamingLatentsDataset(
            manifest=manifest,
            ring_buffer=buffer,
            shuffle_shards=False,
            shuffle_samples=False,
            infinite=False,
            return_torch=True,
        )

        samples = list(iter(dataset))
        self.assertEqual(len(samples), 5)  # 3 + 2

        first = samples[0]
        self.assertIn("x_0_occ", first)
        self.assertIn("x_0_lat", first)
        self.assertIn("c_raw", first)
        self.assertIn("hermite_normals", first)
        self.assertIn("surface_ratio", first)

        self.assertIsInstance(first["x_0_occ"], torch.Tensor)
        self.assertEqual(first["x_0_occ"].shape, (64,))
        self.assertEqual(first["x_0_lat"].shape, (64, 16))
        self.assertEqual(first["c_raw"].shape, (64, 17))
        self.assertEqual(first["hermite_normals"].shape, (64, 3))
        self.assertEqual(first["surface_ratio"].shape, (64,))

    def test_streaming_dataloader_batches(self):
        """Verify create_streaming_dataloader creates batched tensors."""
        shard_path = self.test_root / "shard_batch.npz"
        self._create_dummy_npz(shard_path, num_samples=6)

        manifest = [
            {"latent_shard_filename": "shard_batch.npz", "local_npz_path": str(shard_path)},
        ]
        buffer = LatentsRingBuffer(cache_dir=self.cache_dir, max_cache_gb=1.0)

        loader = create_streaming_dataloader(
            manifest=manifest,
            batch_size=2,
            num_workers=0,
            ring_buffer=buffer,
            shuffle_shards=False,
            infinite=False,
        )

        batches = list(iter(loader))
        self.assertEqual(len(batches), 3)  # 6 / 2

        batch = batches[0]
        self.assertEqual(batch["x_0_occ"].shape, (2, 64))
        self.assertEqual(batch["x_0_lat"].shape, (2, 64, 16))
        self.assertEqual(batch["c_raw"].shape, (2, 64, 17))
        self.assertEqual(batch["hermite_normals"].shape, (2, 64, 3))
        self.assertEqual(batch["surface_ratio"].shape, (2, 64))


if __name__ == "__main__":
    unittest.main()
