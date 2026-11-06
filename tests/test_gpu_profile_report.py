import json
from pathlib import Path
import unittest


class GPUProfileReportTests(unittest.TestCase):
    def setUp(self):
        self.report_path = Path('artifacts/kaggle-gpu-runtime-profile-v1/gpu-runtime-profile-report.json')

    def test_report_exists_and_valid(self):
        self.assertTrue(self.report_path.exists(), "GPU runtime profile report must exist")
        data = json.loads(self.report_path.read_text())
        self.assertIn("hardware", data)
        self.assertIn("benchmarks", data)
        self.assertIn("Tesla T4", data["hardware"])

    def test_acceptance_criteria(self):
        data = json.loads(self.report_path.read_text())
        acceptance = data.get("acceptance", {})

        # Peak VRAM must be well below 4.0 GB threshold (measured ~0.411 GB)
        self.assertTrue(acceptance.get("vram_threshold_passed", False))
        self.assertLess(acceptance.get("measured_peak_vram_gb", 10.0), 1.0)

        # 10-step flow latency must be under 3.0s per asset on T4
        benchmarks = {b["steps"]: b for b in data["benchmarks"]}
        self.assertIn(10, benchmarks)
        self.assertLess(benchmarks[10]["mean_total_seconds"], 3.0)

        # Verify 100% PBR generation across all benchmark runs
        for step_count, bench in benchmarks.items():
            self.assertEqual(len(bench["runs"]), 4)
            for run in bench["runs"]:
                self.assertTrue(run["has_pbr"])
                self.assertGreater(run["mesh_vertices"], 0)
                self.assertGreater(run["mesh_triangles"], 0)
                self.assertGreater(run["glb_bytes"], 1000)
