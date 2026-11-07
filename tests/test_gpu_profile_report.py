import json
from pathlib import Path
import unittest


class GPUProfileReportTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Path('tests/fixtures/gpu-runtime-profile-schema.json')
        self.real_report = Path('artifacts/kaggle-gpu-runtime-profile-v1/gpu-runtime-profile-report.json')

    def test_schema_fixture_is_valid(self):
        data = json.loads(self.fixture.read_text())
        self.assertIn("hardware", data)
        self.assertIn("benchmarks", data)
        self.assertIn("T4", data["hardware"])
        benchmarks = {b["steps"]: b for b in data["benchmarks"]}
        self.assertIn(10, benchmarks)
        self.assertLess(benchmarks[10]["mean_total_seconds"], 3.0)
        for bench in benchmarks.values():
            self.assertEqual(len(bench["runs"]), 4)
            for run in bench["runs"]:
                self.assertTrue(run["has_pbr"])
                self.assertGreater(run["mesh_vertices"], 0)
                self.assertGreater(run["mesh_triangles"], 0)
                self.assertGreater(run["glb_bytes"], 1000)

    @unittest.skipUnless(
        Path('artifacts/kaggle-gpu-runtime-profile-v1/gpu-runtime-profile-report.json').exists(),
        'real GPU runtime profile artifact not present',
    )
    def test_measured_report_meets_acceptance_when_present(self):
        data = json.loads(self.real_report.read_text())
        self.assertIn("Tesla T4", data["hardware"])
        acceptance = data.get("acceptance", {})
        self.assertTrue(acceptance.get("vram_threshold_passed", False))
        self.assertLess(acceptance.get("measured_peak_vram_gb", 10.0), 1.0)
        benchmarks = {b["steps"]: b for b in data["benchmarks"]}
        self.assertIn(10, benchmarks)
        self.assertLess(benchmarks[10]["mean_total_seconds"], 3.0)


if __name__ == '__main__':
    unittest.main()
