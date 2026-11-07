"""Unit tests for VLM aesthetic and geometry filtering pipeline."""
import unittest
import numpy as np
import trimesh

from t3d.data.refinement.vlm_filter import (
    render_projected_views,
    check_geometric_complexity,
    verify_pbr_materials,
    VLMQualityFilter,
)


class TestVLMFilter(unittest.TestCase):
    def setUp(self):
        # Clean detailed model: torus / icosphere
        self.clean_mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
        # Flat planar ground plane
        self.flat_sheet = trimesh.creation.box(extents=[1.0, 0.001, 1.0])
        # Trivial primitive box with 12 faces
        self.cube = trimesh.creation.box(extents=[0.5, 0.5, 0.5])

    def test_multi_view_projections_rendering(self):
        views = render_projected_views(self.clean_mesh, image_size=64)
        self.assertIn("front", views)
        self.assertIn("right", views)
        self.assertIn("top", views)
        self.assertIn("iso", views)

        for view_name, data in views.items():
            self.assertEqual(data["depth"].shape, (64, 64))
            self.assertEqual(data["normal"].shape, (64, 64, 3))
            self.assertEqual(data["mask"].shape, (64, 64))
            # Normal map values in [0, 1]
            self.assertTrue(np.all(data["normal"] >= 0.0))
            self.assertTrue(np.all(data["normal"] <= 1.0))
            # Foreground mask should have active texels
            self.assertTrue(np.any(data["mask"]))

    def test_geometric_complexity_analysis(self):
        clean_metrics = check_geometric_complexity(self.clean_mesh)
        self.assertFalse(clean_metrics["is_primitive"])
        self.assertFalse(clean_metrics["has_ground_plane"])
        self.assertGreater(clean_metrics["dihedral_variance"], 0.0)

        # Flat sheet test
        flat_metrics = check_geometric_complexity(self.flat_sheet)
        self.assertTrue(flat_metrics["is_primitive"])

    def test_filter_clean_vs_primitive(self):
        filter_gate = VLMQualityFilter(min_faces=200, reject_primitives=True)
        res_clean = filter_gate.filter(self.clean_mesh)
        self.assertTrue(res_clean.passed)
        self.assertGreaterEqual(res_clean.quality_score, 0.8)

        # Cube has only 12 faces (< 200) -> should be rejected
        res_cube = filter_gate.filter(self.cube)
        self.assertFalse(res_cube.passed)
        self.assertTrue(res_cube.is_primitive or "below minimum threshold" in res_cube.reasons[0])

    def test_vlm_prompt_generation(self):
        filter_gate = VLMQualityFilter()
        payload = filter_gate.generate_vlm_prompt(self.clean_mesh, asset_id="test_icosphere")
        self.assertIn("test_icosphere", payload["prompt"])
        self.assertIn("front", payload["views"])
        self.assertIn("num_faces", payload["metrics"])


if __name__ == "__main__":
    unittest.main()
