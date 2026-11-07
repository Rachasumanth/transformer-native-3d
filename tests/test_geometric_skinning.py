"""Unit tests for geometric skinning engine and medial axis refinement."""
import unittest
import numpy as np
import trimesh
from t3d.articulation import (
    Skeleton,
    refine_joints_medial_axis,
    compute_geometric_skin_weights,
)
from t3d.glb import pack_static_glb, validate_glb_semantics


class GeometricSkinningTests(unittest.TestCase):
    def test_refine_joints_medial_axis_snap_toward_centroid(self):
        # Create a synthetic voxel cylinder/tube along Z: x, y in [-0.1, 0.1], z in [-0.2, 0.2]
        centers = []
        for z in np.linspace(-0.2, 0.2, 5):
            for x in np.linspace(-0.1, 0.1, 5):
                for y in np.linspace(-0.1, 0.1, 5):
                    centers.append([x, y, z])
        centers = np.array(centers, dtype=float)
        voxel_size = 0.05

        # Joint 1 placed off-center at [0.05, 0.0, 0.0] with parent at [0.05, 0.0, -0.2]
        skel = Skeleton([[0.05, 0.0, -0.2], [0.05, 0.0, 0.0]], [-1, 0])
        refined = refine_joints_medial_axis(skel, centers, voxel_size)

        # The enclosing cross-section centroid is at [0.0, 0.0, 0.0]
        # Snapping 30% from 0.05 toward 0.0 brings it to 0.05 - 0.3 * 0.05 = 0.035
        np.testing.assert_allclose(refined.joints[1, 0], 0.035, atol=1e-5)
        np.testing.assert_allclose(refined.joints[1, 1], 0.0, atol=1e-5)
        np.testing.assert_allclose(refined.joints[1, 2], 0.0, atol=1e-5)

        # Joint already at the centroid [0, 0, 0] should remain unchanged
        centered_skel = Skeleton([[0.0, 0.0, -0.2], [0.0, 0.0, 0.0]], [-1, 0])
        centered_refined = refine_joints_medial_axis(centered_skel, centers, voxel_size)
        np.testing.assert_allclose(centered_refined.joints[1], [0.0, 0.0, 0.0], atol=1e-6)

    def test_refine_joints_medial_axis_multi_limb_isolation(self):
        # Two limbs: Left limb at x in [-0.4, -0.2], Right limb at x in [0.2, 0.4]
        centers = []
        for z in np.linspace(-0.2, 0.2, 5):
            for x in np.linspace(-0.4, -0.2, 5):
                for y in np.linspace(-0.1, 0.1, 5):
                    centers.append([x, y, z])
            for x in np.linspace(0.2, 0.4, 5):
                for y in np.linspace(-0.1, 0.1, 5):
                    centers.append([x, y, z])
        centers = np.array(centers, dtype=float)
        voxel_size = 0.05

        # Joint placed in the left limb off-center at [-0.25, 0, 0], bone along Z
        skel = Skeleton([[-0.25, 0.0, -0.2], [-0.25, 0.0, 0.0]], [-1, 0])
        refined = refine_joints_medial_axis(skel, centers, voxel_size)

        # Left limb centroid is at x = -0.3.
        # Snapping 30% from -0.25 toward -0.3 should reach -0.25 + 0.3 * (-0.3 - -0.25) = -0.265
        np.testing.assert_allclose(refined.joints[1, 0], -0.265, atol=1e-5)
        np.testing.assert_allclose(refined.joints[1, 1], 0.0, atol=1e-5)

    def test_refine_joints_medial_axis_edge_cases(self):
        skel = Skeleton([[0.0, 0.0, 0.0]], [-1])
        # Empty centers returns skeleton unchanged
        empty_res = refine_joints_medial_axis(skel, np.empty((0, 3)), 0.05)
        np.testing.assert_array_equal(empty_res.joints, skel.joints)

        # Invalid skeleton input
        with self.assertRaises(ValueError):
            refine_joints_medial_axis([[0, 0, 0]], [[0, 0, 0]], 0.05)

        # Vector voxel_size
        centers = np.array([[0.0, 0.0, 0.0]])
        res_vec = refine_joints_medial_axis(skel, centers, [0.05, 0.05, 0.05])
        self.assertEqual(len(res_vec.joints), 1)

    def test_compute_geometric_skin_weights_synthetic_cylinder(self):
        # Create a cylinder mesh along Y
        cylinder = trimesh.creation.cylinder(radius=0.1, height=2.0, sections=16)
        vertices = np.asarray(cylinder.vertices, dtype=np.float32)
        faces = np.asarray(cylinder.faces, dtype=np.int32)

        # 3-joint skeleton along Y: Joint 0 at y=-1, Joint 1 at y=0, Joint 2 at y=1
        skeleton = Skeleton(
            [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [-1, 0, 1]
        )

        weights = compute_geometric_skin_weights(vertices, faces, skeleton, max_influences=4)

        # Verify shape
        self.assertEqual(weights.shape, (len(vertices), 3))
        self.assertEqual(weights.dtype, np.float32)

        # Verify no NaNs or non-finite values
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights >= 0.0))

        # Verify max influences <= 4 per vertex
        influences_per_vertex = np.count_nonzero(weights > 0, axis=1)
        self.assertTrue(np.all(influences_per_vertex <= 4))

        # Verify zero out weights below 0.005
        non_zero_weights = weights[weights > 0]
        self.assertTrue(np.all(non_zero_weights >= 0.005 - 1e-7))

        # Verify exact sum to 1.0 in float32 with zero IEEE-754 drift
        row_sums = weights.sum(axis=1)
        for s in row_sums:
            self.assertEqual(s, 1.0)

        # Verify falloff characteristics:
        # Vertices near bottom (y < -0.8) should have higher Joint 0 weight than Joint 2
        bottom_mask = vertices[:, 1] < -0.8
        self.assertTrue(np.all(weights[bottom_mask, 0] > weights[bottom_mask, 2]))

        # Vertices near top (y > 0.8) should have higher Joint 2 weight than Joint 0
        top_mask = vertices[:, 1] > 0.8
        self.assertTrue(np.all(weights[top_mask, 2] > weights[top_mask, 0]))

    def test_compute_geometric_skin_weights_branching_skeleton(self):
        # 17-joint humanoid skeleton
        joints = [
            [0.0, 0.0, 0.0],       # 0: Pelvis
            [0.0, 0.3, 0.0],       # 1: Spine
            [0.0, 0.6, 0.0],       # 2: Chest
            [0.0, 0.8, 0.0],       # 3: Neck
            [0.0, 1.0, 0.0],       # 4: Head
            [-0.2, 0.6, 0.0],      # 5: L Shoulder
            [-0.4, 0.6, 0.0],      # 6: L Elbow
            [-0.6, 0.6, 0.0],      # 7: L Wrist
            [0.2, 0.6, 0.0],       # 8: R Shoulder
            [0.4, 0.6, 0.0],       # 9: R Elbow
            [0.6, 0.6, 0.0],       # 10: R Wrist
            [-0.15, -0.1, 0.0],    # 11: L Hip
            [-0.15, -0.5, 0.0],    # 12: L Knee
            [-0.15, -0.9, 0.0],    # 13: L Ankle
            [0.15, -0.1, 0.0],     # 14: R Hip
            [0.15, -0.5, 0.0],     # 15: R Knee
            [0.15, -0.9, 0.0],     # 16: R Ankle
        ]
        parents = [-1, 0, 1, 2, 3, 2, 5, 6, 2, 8, 9, 0, 11, 12, 0, 14, 15]
        skeleton = Skeleton(joints, parents)

        # Sample 100 test vertices around the skeleton
        rng = np.random.default_rng(42)
        sampled_pts = []
        for j in joints:
            noise = rng.normal(0, 0.05, size=(6, 3))
            sampled_pts.append(np.array(j) + noise)
        vertices = np.vstack(sampled_pts).astype(np.float32)
        faces = np.empty((0, 3), dtype=np.int32)

        weights = compute_geometric_skin_weights(vertices, faces, skeleton, max_influences=4)

        # Assertions
        self.assertEqual(weights.shape, (len(vertices), 17))
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights >= 0.0))
        self.assertTrue(np.all(np.count_nonzero(weights > 0, axis=1) <= 4))

        # Exact 1.0 float32 sum
        row_sums = weights.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

        # L Wrist vertices should have zero influence on R Ankle
        l_wrist_v = vertices[7 * 6 : 8 * 6]
        l_wrist_weights = weights[7 * 6 : 8 * 6]
        np.testing.assert_array_equal(l_wrist_weights[:, 16], 0.0)

    def test_compute_geometric_skin_weights_edge_cases(self):
        # Single-joint skeleton
        single_skel = Skeleton([[0.0, 0.0, 0.0]], [-1])
        verts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        weights = compute_geometric_skin_weights(verts, [], single_skel)
        self.assertEqual(weights.shape, (2, 1))
        np.testing.assert_array_equal(weights, [[1.0], [1.0]])

        # Empty vertices
        empty_w = compute_geometric_skin_weights(np.empty((0, 3)), [], single_skel)
        self.assertEqual(empty_w.shape, (0, 1))

        # Invalid inputs
        with self.assertRaises(ValueError):
            compute_geometric_skin_weights(verts, [], "not_a_skeleton")
        with self.assertRaises(ValueError):
            compute_geometric_skin_weights(verts, [], single_skel, max_influences=0)

    def test_rigged_glb_export_and_semantics_validation(self):
        # Cylinder mesh
        cylinder = trimesh.creation.cylinder(radius=0.1, height=1.0, sections=12)
        v = np.asarray(cylinder.vertices, dtype=np.float32)
        f = np.asarray(cylinder.faces, dtype=np.int32)

        skeleton = Skeleton(
            [[0.0, 0.0, -0.4], [0.0, 0.0, 0.0], [0.0, 0.0, 0.4]],
            [-1, 0, 1]
        )

        # Calling pack_static_glb without skin_weights should auto-compute geometric skin weights
        glb_bytes = pack_static_glb(v, f, skeleton=skeleton, source_up='z')
        self.assertTrue(glb_bytes.startswith(b'glTF'))

        # Validate with validate_glb_semantics
        report = validate_glb_semantics(glb_bytes)
        self.assertTrue(report['valid'])
        self.assertTrue(report['skinning']['is_skinned'])
        self.assertEqual(report['skinning']['joint_count'], 3)
        self.assertGreaterEqual(report['skinning']['skin_attribute_sets'], 1)

    def test_rigged_glb_export_with_pbr_semantics(self):
        cylinder = trimesh.creation.cylinder(radius=0.1, height=1.0, sections=12)
        v = np.asarray(cylinder.vertices, dtype=np.float32)
        f = np.asarray(cylinder.faces, dtype=np.int32)
        pbr = np.tile([0.8, 0.2, 0.2, 0.1, 0.5, 1.0], (len(v), 1))

        skeleton = Skeleton(
            [[0.0, 0.0, -0.4], [0.0, 0.0, 0.0], [0.0, 0.0, 0.4]],
            [-1, 0, 1]
        )

        glb_bytes = pack_static_glb(v, f, pbr=pbr, skeleton=skeleton, source_up='z')
        report = validate_glb_semantics(glb_bytes)
        self.assertTrue(report['valid'])
        self.assertTrue(report['six_channels_preserved'])
        self.assertTrue(report['skinning']['is_skinned'])
        self.assertEqual(report['skinning']['joint_count'], 3)

    def test_normal_visibility_penalty_and_laplacian_diffusion(self):
        # Two parallel disconnected walls (simulating two legs with an air gap)
        # Left wall at x = -0.2, right wall at x = 0.2
        # Normal of left wall points left (-X), normal of right wall points right (+X)
        v_left = np.array([[-0.2, -0.5, 0], [-0.2, 0.5, 0], [-0.2, 0.5, 1], [-0.2, -0.5, 1]], dtype=np.float32)
        f_left = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        v_right = np.array([[0.2, -0.5, 0], [0.2, 0.5, 0], [0.2, 0.5, 1], [0.2, -0.5, 1]], dtype=np.float32)
        f_right = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32) + len(v_left)

        v_all = np.vstack([v_left, v_right])
        f_all = np.vstack([f_left, f_right])

        # Skeleton has two bones: Left leg bone along X=-0.2, Right leg bone along X=0.2
        joints = np.array([[-0.2, 0.0, 0.0], [-0.2, 0.0, 1.0], [0.2, 0.0, 0.0], [0.2, 0.0, 1.0]], dtype=np.float32)
        parents = np.array([-1, 0, -1, 2], dtype=np.int32)
        skel = Skeleton(joints, parents)

        weights = compute_geometric_skin_weights(v_all, f_all, skel, max_influences=4)
        # Left wall vertices (indices 0..3) should have zero or near-zero influence on right bone (joint 3)
        self.assertTrue(np.all(weights[:4, 3] < 0.01))
        # Right wall vertices (indices 4..7) should have zero or near-zero influence on left bone (joint 1)
        self.assertTrue(np.all(weights[4:, 1] < 0.01))


if __name__ == '__main__':
    unittest.main()
