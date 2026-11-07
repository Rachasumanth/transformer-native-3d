"""Unit tests for T3D Disk Watchdog and assert_disk_budget."""

import os
import unittest
from pathlib import Path
import warnings

from scripts.disk_watchdog import (
    assert_disk_budget,
    get_disk_usage,
    init_ephemeral_env,
    evict_ephemeral_cache,
    pause_downloaders,
    resume_downloaders,
    is_download_paused,
    DiskBudgetExceededError,
    EPHEMERAL_SCRATCH,
    PAUSE_FLAG_FILE,
)


class TestDiskWatchdog(unittest.TestCase):
    def setUp(self):
        resume_downloaders()
        init_ephemeral_env()

    def tearDown(self):
        resume_downloaders()

    def test_environment_configuration(self):
        """Verify environment variables and ephemeral scratch directory setup."""
        env_vars = init_ephemeral_env()
        self.assertTrue(EPHEMERAL_SCRATCH.exists())
        self.assertEqual(os.environ.get("TMPDIR"), str(EPHEMERAL_SCRATCH))
        self.assertEqual(os.environ.get("HF_HOME"), str(EPHEMERAL_SCRATCH))
        self.assertEqual(os.environ.get("TORCH_HOME"), str(EPHEMERAL_SCRATCH))
        self.assertIn("HUGGINGFACE_HUB_CACHE", os.environ)

    def test_get_disk_usage(self):
        """Verify disk usage metrics return valid numerical values."""
        used_gb, free_gb, total_gb, percent = get_disk_usage("/")
        self.assertGreater(total_gb, 0.0)
        self.assertGreater(free_gb, 0.0)
        self.assertGreater(used_gb, 0.0)
        self.assertGreaterEqual(percent, 0.0)
        self.assertLessEqual(percent, 100.0)

    def test_assert_disk_budget_normal(self):
        """Under normal conditions (ceiling 35 GB), assert_disk_budget succeeds."""
        used_gb = assert_disk_budget(max_gb=35.0, path="/")
        self.assertIsInstance(used_gb, float)
        self.assertLess(used_gb, 35.0)
        self.assertFalse(is_download_paused())

    def test_pause_and_resume_downloaders(self):
        """Verify downloader pause flag toggle mechanism."""
        self.assertFalse(is_download_paused())
        pause_downloaders()
        self.assertTrue(is_download_paused())
        resume_downloaders()
        self.assertFalse(is_download_paused())

    def test_cache_eviction(self):
        """Verify eviction purges files within ephemeral scratch."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_scratch:
            test_file = Path(tmp_scratch) / "dummy_cache_file.dat"
            test_dir = Path(tmp_scratch) / "dummy_subdir"
            test_dir.mkdir(parents=True, exist_ok=True)
            with open(test_file, "w") as f:
                f.write("temporary cache content")
            with open(test_dir / "nested.dat", "w") as f:
                f.write("nested content")

            self.assertTrue(test_file.exists())
            self.assertTrue((test_dir / "nested.dat").exists())

            evict_ephemeral_cache(scratch_dir=tmp_scratch, keep_subdirs=())

            self.assertFalse(test_file.exists())
            self.assertFalse((test_dir / "nested.dat").exists())
            self.assertTrue(Path(tmp_scratch).exists())

    def test_assert_disk_budget_exceeded_raises(self):
        """When budget is artificially set lower than current usage, an exception is raised."""
        # 1.0 GB limit is well below actual disk usage (~5 GB)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_scratch:
            with self.assertRaises((DiskBudgetExceededError, RuntimeError)):
                assert_disk_budget(max_gb=1.0, path="/", scratch_dir=tmp_scratch, auto_evict=True, raise_on_error=True)

    def test_assert_disk_budget_exceeded_warns(self):
        """When raise_on_error is False, warnings are emitted without crashing."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_scratch:
            with warnings.catch_warnings(record=True) as recorded_warnings:
                warnings.simplefilter("always")
                used = assert_disk_budget(max_gb=1.0, path="/", scratch_dir=tmp_scratch, auto_evict=True, raise_on_error=False)
                self.assertGreater(used, 1.0)
                self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in recorded_warnings))


if __name__ == "__main__":
    unittest.main()
