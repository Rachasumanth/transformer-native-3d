"""Unit tests for automated production acceptance harness."""

import unittest
import numpy as np
import trimesh

from scripts.eval_acceptance import (
    check_topology_and_manifold,
    evaluate_coarse_occupancy,
    evaluate_surface_distance_and_fscore,
    evaluate_skeleton_rigging,
    evaluate_asset,
)


class TestEvalAcceptance(unittest.TestCase):
    def test_watertight_manifold_box(self):
        """Verify watertightness and edge counts on a closed manifold mesh."""
        mesh = trimesh.creation.box()
        # Subdivide to hit 1500-5000 face count
        while len(mesh.faces) < 1500:
            mesh = mesh.subdivide()

        is_wt, non_manifold, self_intersect, n_faces = check_topology_and_manifold(mesh)
        self.assertTrue(is_wt)
        self.assertEqual(non_manifold, 0)
        self.assertEqual(self_intersect, 0)
        self.assertGreaterEqual(n_faces, 1500)
        self.assertLessEqual(n_faces, 5000)

    def test_occupancy_recall(self):
        """Verify occupancy recall metric matches TP / (TP + FN)."""
        gt = np.array([1.0, 1.0, 1.0, -1.0, -1.0])
        pred_good = np.array([2.5, 1.1, 0.4, -2.0, -1.5])  # 3/3 active predicted
        recall = evaluate_coarse_occupancy(pred_good, gt)
        self.assertEqual(recall, 1.0)

        pred_miss = np.array([-1.0, 1.1, 0.4, -2.0, -1.5])  # 2/3 active predicted
        recall_miss = evaluate_coarse_occupancy(pred_miss, gt)
        self.assertAlmostEqual(recall_miss, 2.0 / 3.0, places=4)

    def test_surface_distance_and_fscore_identical(self):
        """Verify MSD is ~0 and F-Score is 1.0 for identical meshes."""
        mesh = trimesh.creation.icosphere(subdivisions=3)
        msd, f_score, prec, rec = evaluate_surface_distance_and_fscore(
            mesh.vertices, mesh.faces,
            mesh.vertices, mesh.faces,
            num_samples=2000,
            tolerance=0.01,
        )
        self.assertLess(msd, 1e-4)
        self.assertAlmostEqual(f_score, 1.0, places=2)

    def test_skeleton_rigging_metrics(self):
        """Verify joint-to-joint Chamfer error, containment violations, and skinning weights check."""
        gt_j = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]])
        # Small perturbation < 0.02
        pred_j = gt_j + np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])

        box = trimesh.creation.box(extents=(1.0, 2.5, 1.0))
        weights = np.ones((len(box.vertices), 3), dtype=np.float32) / 3.0

        j_err, violations, w_valid = evaluate_skeleton_rigging(
            pred_j, gt_j, mesh_vertices=box.vertices, pred_weights=weights
        )

        self.assertLess(j_err, 0.05)
        self.assertEqual(violations, 0)
        self.assertTrue(w_valid)


if __name__ == "__main__":
    unittest.main()
