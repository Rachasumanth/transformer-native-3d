"""Unit tests for pose and scale canonicalization module."""
import unittest
import numpy as np
import trimesh

from t3d.data.refinement.pose_canonicalize import (
    compute_canonical_pose_transform,
    canonicalize_pose_and_scale,
    random_quarter_turn_augmentation,
)


class TestPoseCanonicalize(unittest.TestCase):
    def setUp(self):
        # Create an off-center, unscaled mesh
        self.mesh = trimesh.creation.box(extents=[2.0, 4.0, 1.0])
        # Shift away from origin
        self.mesh.vertices += np.array([5.0, -3.0, 8.0])

    def test_canonicalize_centering_and_scaling(self):
        canon_mesh, transform = canonicalize_pose_and_scale(
            self.mesh,
            target_half_extent=0.45,
            margin_domain=0.50,
            align_orientation=True,
        )

        # Check bounds: must be centered and strictly inside [-0.45, 0.45]^3
        bounds = canon_mesh.bounds
        center = (bounds[0] + bounds[1]) / 2.0
        np.testing.assert_allclose(center, 0.0, atol=1e-5)

        self.assertGreaterEqual(float(bounds[0].min()), -0.45001)
        self.assertLessEqual(float(bounds[1].max()), 0.45001)

        # Maximum extent should be approximately 2 * 0.45 = 0.90
        extents = bounds[1] - bounds[0]
        self.assertAlmostEqual(float(np.max(extents)), 0.90, places=4)

    def test_transform_invertibility(self):
        _, transform = canonicalize_pose_and_scale(self.mesh)
        orig_verts = self.mesh.vertices.copy()
        
        # P -> canonical -> invert back
        canon_verts = transform.apply_points(orig_verts)
        restored_verts = transform.invert_points(canon_verts)

        np.testing.assert_allclose(restored_verts, orig_verts, atol=1e-5)

    def test_quarter_turn_augmentation(self):
        rng = np.random.default_rng(seed=42)
        aug_mesh, k = random_quarter_turn_augmentation(self.mesh, axes=("y",), rng=rng)
        
        # Vertex count and topology unchanged
        self.assertEqual(len(aug_mesh.vertices), len(self.mesh.vertices))
        self.assertEqual(len(aug_mesh.faces), len(self.mesh.faces))
        self.assertIn(k, (0, 1, 2, 3))


if __name__ == "__main__":
    unittest.main()
