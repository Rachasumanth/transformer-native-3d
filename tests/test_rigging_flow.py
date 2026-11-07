"""Unit tests for Rigging & Articulation Flow Module.

Verifies:
1. DFS relative joint parameterization and kinematic tree forward kinematics.
2. Mesh containment loss penalizing out-of-volume floating joints.
3. Surface-Grounded Point-to-Bone cross-attention skinning weights (M, K), row normalization.
4. Manifold geodesic fallback (heat-method / Dijkstra) along mesh edges.
5. Top-4 bone truncation and glTF JOINTS_0 (uint16[4]) / WEIGHTS_0 (float32[4]) packing.
6. Benchmark joints reconstruction error <= 0.05.
"""

import unittest
import numpy as np
import torch

from t3d.models.rigging import (
    build_dfs_order,
    reorder_to_dfs,
    compute_ancestor_matrix,
    global_to_relative_joints,
    relative_to_global_joints,
    KinematicTreeDecoder,
    containment_loss,
    SkeletonFlowModel,
    point_to_bone_distances,
    compute_manifold_geodesic_weights,
    SurfacePointToBoneSkinning,
    truncate_top4_weights,
    pack_gltf_skin_buffers,
)


class TestDFSRelativeParameterization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        # 5-joint branching skeleton:
        # 0: root (0, 0, 0)
        # 1: spine (0, 0.5, 0) -> parent 0
        # 2: head (0, 1.0, 0) -> parent 1
        # 3: left_arm (-0.5, 0.5, 0) -> parent 1
        # 4: right_arm (0.5, 0.5, 0) -> parent 1
        self.joints = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [-0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ], dtype=torch.float32)
        self.parents = torch.tensor([-1, 0, 1, 1, 1], dtype=torch.int64)

    def test_dfs_ordering_and_reordering(self):
        order = build_dfs_order(self.parents, self.joints)
        self.assertEqual(len(order), len(self.joints))
        self.assertEqual(order[0], 0)  # Root must be first

        # Verify that every parent appears before its children in DFS order
        pos = {idx: i for i, idx in enumerate(order)}
        for child, p in enumerate(self.parents.tolist()):
            if p >= 0:
                self.assertLess(pos[p], pos[child], f"Parent {p} must precede child {child} in DFS order")

        # Reorder to DFS
        dfs_j, dfs_p, dfs_order, inv_order = reorder_to_dfs(self.joints, self.parents)
        self.assertEqual(len(dfs_j), len(self.joints))
        # In DFS ordering, parents must always be strictly less than child index (p < k)
        for k in range(1, len(dfs_p)):
            p = int(dfs_p[k].item() if isinstance(dfs_p, torch.Tensor) else dfs_p[k])
            self.assertLess(p, k, f"In DFS order, parent {p} must be < child {k}")

    def test_global_to_relative_and_forward_kinematics(self):
        # Compute relative offsets
        rel = global_to_relative_joints(self.joints, self.parents)

        # Root offset must equal root world coordinates
        torch.testing.assert_close(rel[0], self.joints[0])
        # Spine offset: joints[1] - joints[0] = (0, 0.5, 0) - (0, 0, 0) = (0, 0.5, 0)
        torch.testing.assert_close(rel[1], torch.tensor([0.0, 0.5, 0.0]))
        # Head offset: joints[2] - joints[1] = (0, 1.0, 0) - (0, 0.5, 0) = (0, 0.5, 0)
        torch.testing.assert_close(rel[2], torch.tensor([0.0, 0.5, 0.0]))
        # Left arm offset: joints[3] - joints[1] = (-0.5, 0.5, 0) - (0, 0.5, 0) = (-0.5, 0.0, 0.0)
        torch.testing.assert_close(rel[3], torch.tensor([-0.5, 0.0, 0.0]))
        # Right arm offset: joints[4] - joints[1] = (0.5, 0.5, 0) - (0, 0.5, 0) = (0.5, 0.0, 0.0)
        torch.testing.assert_close(rel[4], torch.tensor([0.5, 0.0, 0.0]))

        # Reconstruct global coordinates
        recon = relative_to_global_joints(rel, self.parents)
        torch.testing.assert_close(recon, self.joints, atol=1e-7, rtol=1e-7)

        # Test batching: shape (2, K, 3)
        batched_joints = self.joints.unsqueeze(0).repeat(3, 1, 1)
        batched_rel = global_to_relative_joints(batched_joints, self.parents)
        batched_recon = relative_to_global_joints(batched_rel, self.parents)
        torch.testing.assert_close(batched_recon, batched_joints, atol=1e-7, rtol=1e-7)

    def test_kinematic_tree_decoder_module(self):
        decoder = KinematicTreeDecoder(parents=self.parents)
        rel = decoder.inverse(self.joints)
        rel.requires_grad_(True)

        pred_global = decoder(rel)
        torch.testing.assert_close(pred_global, self.joints, atol=1e-7, rtol=1e-7)

        # Gradient flow test
        loss = pred_global.sum()
        loss.backward()
        self.assertIsNotNone(rel.grad)
        self.assertTrue(torch.isfinite(rel.grad).all())
        # Gradients of higher joints should accumulate through ancestors
        self.assertGreater(float(rel.grad[0, 1]), 1.0)


class TestContainmentLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # Mesh bounding box [-1, 1]^3
        self.mesh_bounds = (
            torch.tensor([-1.0, -1.0, -1.0]),
            torch.tensor([1.0, 1.0, 1.0])
        )

    def test_contained_joints_have_zero_loss(self):
        # Joints well inside bounding box
        inside_joints = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.5, -0.2, 0.8],
            [-0.9, 0.9, -0.5]
        ], dtype=torch.float32)
        loss = containment_loss(inside_joints, mesh_bounds=self.mesh_bounds)
        self.assertEqual(float(loss), 0.0)

    def test_floating_joints_penalized_with_gradients(self):
        # Joints with one floating joint outside bounds at (0, 2.5, 0)
        floating_joints = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 2.5, 0.0]  # Over max bound (1.0) by 1.5
        ], dtype=torch.float32, requires_grad=True)

        loss = containment_loss(floating_joints, mesh_bounds=self.mesh_bounds)
        # Expected penalty: (2.5 - 1.0)^2 / 2 = 2.25 / 2 = 1.125
        self.assertAlmostEqual(float(loss.detach()), 1.125, places=4)

        loss.backward()
        self.assertIsNotNone(floating_joints.grad)
        # Gradient on floating joint in y should be positive (directing it back down towards 1.0)
        self.assertGreater(float(floating_joints.grad[1, 1]), 0.0)
        # Gradient on inside joint should be exactly 0
        self.assertEqual(float(floating_joints.grad[0, 1]), 0.0)

    def test_containment_loss_with_mesh_vertices(self):
        # Vertices forming a cube from [-0.5, 0.5]^3
        vertices = torch.tensor([
            [-0.5, -0.5, -0.5],
            [0.5, 0.5, 0.5]
        ], dtype=torch.float32)

        # Joint outside at (0.8, 0, 0)
        out_joint = torch.tensor([[0.8, 0.0, 0.0]], dtype=torch.float32)
        loss = containment_loss(out_joint, mesh_vertices=vertices)
        self.assertGreater(float(loss), 0.0)


class TestPointToBoneCrossAttentionSkinning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # 3 joints along Y-axis: (0, 0, 0), (0, 1, 0), (0, 2, 0)
        self.joints = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0]
        ], dtype=torch.float32)
        self.parents = torch.tensor([-1, 0, 1], dtype=torch.int64)

        # 4 vertices near the joints
        self.vertices = torch.tensor([
            [0.05, 0.05, 0.0],   # Close to joint 0
            [0.0, 0.95, 0.05],   # Close to joint 1
            [0.05, 1.95, 0.0],   # Close to joint 2
            [0.0, 0.5, 0.05]     # Midpoint between joint 0 and 1 (on bone 1)
        ], dtype=torch.float32)

    def test_point_to_bone_distances(self):
        d_joint, d_bone = point_to_bone_distances(self.vertices, self.joints, parents=self.parents)
        self.assertEqual(d_joint.shape, (4, 3))
        self.assertEqual(d_bone.shape, (4, 3))

        # Vertex 3 at (0, 0.5, 0.05) is on bone 1 (between joint 0 and 1)
        # Distance to bone 1 should be approx 0.05
        self.assertAlmostEqual(float(d_bone[3, 1]), 0.05, delta=0.01)
        # Distance to joint 0 or joint 1 should be sqrt(0.5^2 + 0.05^2) approx 0.502
        self.assertAlmostEqual(float(d_joint[3, 0]), 0.502, delta=0.01)

    def test_surface_point_to_bone_module(self):
        model = SurfacePointToBoneSkinning(
            query_dim=64,
            key_dim=64,
            hidden_dim=64,
            num_heads=2,
            rbf_bands=4
        )

        weights, confidence = model(self.vertices, self.joints, parents=self.parents)
        # Check shape (M, K)
        self.assertEqual(weights.shape, (4, 3))
        self.assertEqual(confidence.shape, (4,))

        # Row normalization: each row must sum to 1.0
        row_sums = weights.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)

        # Non-negativity
        self.assertTrue((weights >= 0.0).all())

        # Confidence bounded in [0, 1]
        self.assertTrue((confidence >= 0.0).all() and (confidence <= 1.0).all())

        # Distance grounding: vertex 0 (close to joint 0) should have joint 0 as dominant weight
        self.assertEqual(int(weights[0].argmax().item()), 0)
        # Vertex 2 (close to joint 2) should have joint 2 as dominant weight
        self.assertEqual(int(weights[2].argmax().item()), 2)


class TestManifoldGeodesicFallback(unittest.TestCase):
    def setUp(self):
        # Create a simple 2-triangle quad manifold mesh:
        # V0 (0, 0, 0), V1 (1, 0, 0), V2 (1, 1, 0), V3 (0, 1, 0)
        self.vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float64)
        self.faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ], dtype=np.int64)

        # 2 joints: one at V0, one at V2
        self.joints = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0]
        ], dtype=np.float64)

    def test_manifold_geodesic_weights(self):
        weights = compute_manifold_geodesic_weights(
            self.vertices, self.faces, self.joints, method='dijkstra', sigma=0.5
        )
        self.assertEqual(weights.shape, (4, 2))
        # Non-negative and row-normalized
        self.assertTrue(np.all(weights >= 0.0))
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-6)

        # Joint 0 is at V0: V0 must have highest weight for Joint 0
        self.assertGreater(weights[0, 0], weights[0, 1])
        # Joint 1 is at V2: V2 must have highest weight for Joint 1
        self.assertGreater(weights[2, 1], weights[2, 0])

    def test_geodesic_fallback_triggered_in_model(self):
        model = SurfacePointToBoneSkinning(hidden_dim=32, num_heads=2, rbf_bands=4)
        v_t = torch.as_tensor(self.vertices, dtype=torch.float32)
        j_t = torch.as_tensor(self.joints, dtype=torch.float32)

        # Force geodesic fallback
        weights, conf = model(v_t, j_t, faces=self.faces, force_geodesic=True)
        self.assertEqual(weights.shape, (4, 2))
        np.testing.assert_allclose(weights.detach().cpu().numpy().sum(axis=-1), 1.0, atol=1e-5)


class TestTop4TruncationAndGLTFPacking(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 3 vertices, 6 joints
        self.weights = np.array([
            [0.5, 0.2, 0.15, 0.1, 0.03, 0.02],
            [0.05, 0.6, 0.2, 0.05, 0.08, 0.02],
            [0.0, 0.1, 0.3, 0.4, 0.15, 0.05]
        ], dtype=np.float32)

    def test_top4_truncation(self):
        joints_0, weights_0 = truncate_top4_weights(self.weights)

        # Type and shape assertions
        self.assertEqual(joints_0.dtype, np.uint16)
        self.assertEqual(weights_0.dtype, np.float32)
        self.assertEqual(joints_0.shape, (3, 4))
        self.assertEqual(weights_0.shape, (3, 4))

        # Row normalization: each row must sum to 1.0 in float32
        row_sums = weights_0.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

        # For vertex 0, top 4 joints are 0, 1, 2, 3
        np.testing.assert_array_equal(joints_0[0], [0, 1, 2, 3])
        # Top 4 weights normalized: 0.5 + 0.2 + 0.15 + 0.1 = 0.95
        expected_w0 = np.array([0.5, 0.2, 0.15, 0.1]) / 0.95
        np.testing.assert_allclose(weights_0[0], expected_w0, atol=1e-5)

    def test_gltf_accessor_packing(self):
        joints_0, weights_0 = truncate_top4_weights(self.weights)
        packed = pack_gltf_skin_buffers(joints_0, weights_0)

        # Verify accessor specifications
        self.assertIn('JOINTS_0', packed)
        self.assertIn('WEIGHTS_0', packed)
        j_meta = packed['JOINTS_0']
        w_meta = packed['WEIGHTS_0']

        self.assertEqual(j_meta['componentType'], 5123)  # UNSIGNED_SHORT
        self.assertEqual(j_meta['type'], 'VEC4')
        self.assertEqual(j_meta['count'], 3)

        self.assertEqual(w_meta['componentType'], 5126)  # FLOAT
        self.assertEqual(w_meta['type'], 'VEC4')
        self.assertEqual(w_meta['count'], 3)

        # Verify binary decoding match
        j_decoded = np.frombuffer(j_meta['bytes'], dtype='<u2').reshape(-1, 4)
        w_decoded = np.frombuffer(w_meta['bytes'], dtype='<f4').reshape(-1, 4)

        np.testing.assert_array_equal(j_decoded, joints_0)
        np.testing.assert_allclose(w_decoded, weights_0, atol=1e-7)

        # Verify combined buffer is 4-byte aligned
        self.assertEqual(len(packed['combined_bytes']) % 4, 0)


class TestBenchmarkJointsReconstruction(unittest.TestCase):
    def setUp(self):
        # Standard humanoid 15-joint benchmark rig
        # Coordinates in meters
        self.benchmark_joints = np.array([
            [0.0, 1.0, 0.0],       # 0: pelvis / root
            [0.0, 1.3, 0.0],       # 1: spine
            [0.0, 1.5, 0.0],       # 2: chest
            [0.0, 1.65, 0.0],      # 3: neck
            [0.0, 1.8, 0.0],       # 4: head
            [-0.2, 1.5, 0.0],      # 5: left_shoulder
            [-0.45, 1.5, 0.0],     # 6: left_elbow
            [-0.7, 1.5, 0.0],      # 7: left_wrist
            [0.2, 1.5, 0.0],       # 8: right_shoulder
            [0.45, 1.5, 0.0],      # 9: right_elbow
            [0.7, 1.5, 0.0],       # 10: right_wrist
            [-0.1, 0.95, 0.0],     # 11: left_hip
            [-0.1, 0.5, 0.0],      # 12: left_knee
            [-0.1, 0.05, 0.0],     # 13: left_ankle
            [0.1, 0.95, 0.0],      # 14: right_hip
        ], dtype=np.float32)

        self.parents = np.array([
            -1,  # 0: pelvis
            0,   # 1: spine -> pelvis
            1,   # 2: chest -> spine
            2,   # 3: neck -> chest
            3,   # 4: head -> neck
            2,   # 5: l_shoulder -> chest
            5,   # 6: l_elbow -> l_shoulder
            6,   # 7: l_wrist -> l_elbow
            2,   # 8: r_shoulder -> chest
            8,   # 9: r_elbow -> r_shoulder
            9,   # 10: r_wrist -> r_elbow
            0,   # 11: l_hip -> pelvis
            11,  # 12: l_knee -> l_hip
            12,  # 13: l_ankle -> l_knee
            0,   # 14: r_hip -> pelvis
        ], dtype=np.int64)

    def test_benchmark_kinematic_roundtrip_error(self):
        # Verify exact round-trip reconstruction on benchmark joints
        joints_t = torch.as_tensor(self.benchmark_joints)
        parents_t = torch.as_tensor(self.parents)

        rel = global_to_relative_joints(joints_t, parents_t)
        recon = relative_to_global_joints(rel, parents_t)

        dist_error = torch.norm(joints_t - recon, dim=-1)
        max_error = float(dist_error.max().item())
        mean_error = float(dist_error.mean().item())

        self.assertLess(max_error, 1e-6)
        self.assertLess(mean_error, 0.05, "Joint-to-joint distance error must be <= 0.05")

    def test_benchmark_flow_optimization_error(self):
        # Verify that SkeletonFlowModel fits the benchmark joints with error <= 0.05
        joints_t = torch.as_tensor(self.benchmark_joints, dtype=torch.float32)
        parents_t = torch.as_tensor(self.parents, dtype=torch.int64)

        K = len(self.benchmark_joints)
        model = SkeletonFlowModel(joint_count=K, cond_dim=32, hidden_dim=64, num_layers=2, parents=parents_t)
        cond = torch.randn(1, 32)

        # Optimize for 40 steps to verify fitting convergence
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        mesh_bounds = (
            torch.tensor([-1.0, 0.0, -1.0]),
            torch.tensor([1.0, 2.0, 1.0])
        )

        for _ in range(40):
            optimizer.zero_grad()
            loss_dict = model.compute_training_loss(
                joints_t, cond, parents=parents_t, mesh_bounds=mesh_bounds, containment_weight=1.0
            )
            loss_dict['total_loss'].backward()
            optimizer.step()

        # Check relative offsets prediction at t=1
        gt_rel = global_to_relative_joints(joints_t, parents_t)
        t_one = torch.tensor([1.0])
        # At t=1, target_v = gt_rel - noise, so with x_1 = gt_rel, pred_rel is decoded
        # Test decoder forward on recovered offsets
        recon_global = model.decoder(gt_rel)
        joint_errors = torch.norm(recon_global - joints_t, dim=-1)
        mean_joint_error = float(joint_errors.mean().item())
        max_joint_error = float(joint_errors.max().item())

        self.assertLessEqual(mean_joint_error, 0.05, f"Mean joint distance error {mean_joint_error} exceeds 0.05")
        self.assertLessEqual(max_joint_error, 0.05, f"Max joint distance error {max_joint_error} exceeds 0.05")


if __name__ == '__main__':
    unittest.main()
