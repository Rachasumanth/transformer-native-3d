"""Unit tests for rig canonicalization, weight transfer, and URDF parsing."""
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np

from t3d.articulation import Skeleton
from t3d.data.refinement.canonicalize_rig import (
    canonicalize_rig,
    parse_partnet_urdf,
    prune_dead_bones,
    dfs_order_skeleton,
    barycentric_weight_transfer,
    laplacian_smooth_weights,
    truncate_top4_weights,
    pack_gltf_skin_accessors,
    RigValidationError,
)
from t3d.import_rig import load_rig_glb


class TestDeadBonePruning(unittest.TestCase):
    def test_prune_dead_bone_and_relink(self):
        # 4 joints: 0 (root), 1 (spine, child of 0), 2 (ik_bone, child of 1), 3 (hand, child of 2)
        joints = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
        ], dtype=np.float64)
        parents = np.array([-1, 0, 1, 2], dtype=np.int64)
        skel = Skeleton(joints, parents)

        # 3 vertices. Bone 2 is dead (max weight = 0.01 < 0.02)
        # Weights: (V, 4)
        weights = np.array([
            [0.5, 0.4, 0.01, 0.09],
            [0.1, 0.7, 0.005, 0.195],
            [0.2, 0.5, 0.01, 0.29],
        ], dtype=np.float64)

        pruned_w, pruned_skel, pruned_idx = prune_dead_bones(weights, skel, threshold=0.02)

        self.assertEqual(pruned_idx, [2])
        self.assertEqual(len(pruned_skel.joints), 3)
        # Bone 3 (hand) should be relinked to parent of bone 2, which is bone 1 (spine)
        # Old mapping: 0 -> 0, 1 -> 1, 3 -> 2
        # In new indexing:
        # Joint 0: parent -1
        # Joint 1: parent 0
        # Joint 2 (old 3): parent 1
        np.testing.assert_array_equal(pruned_skel.parents, np.array([-1, 0, 1]))

        # Residual weights of bone 2 must be transferred to bone 1
        expected_w1 = weights[:, 1] + weights[:, 2]
        np.testing.assert_allclose(pruned_w[:, 1], expected_w1, rtol=1e-5)
        # Check rows sum to 1.0
        np.testing.assert_allclose(pruned_w.sum(axis=1), np.ones(3), rtol=1e-6)

    def test_chained_dead_bones(self):
        # Hierarchy: 0 -> 1 -> 2 -> 3
        # Both 1 and 2 are dead (< 0.02). Hand (3) should relink all the way to 0 (root).
        joints = np.zeros((4, 3))
        parents = np.array([-1, 0, 1, 2])
        skel = Skeleton(joints, parents)

        weights = np.array([
            [0.6, 0.01, 0.005, 0.385],
            [0.5, 0.008, 0.012, 0.48],
        ])

        pruned_w, pruned_skel, pruned_idx = prune_dead_bones(weights, skel, threshold=0.02)
        self.assertEqual(sorted(pruned_idx), [1, 2])
        self.assertEqual(len(pruned_skel.joints), 2)
        # Joint 0: parent -1, Joint 1 (old 3): parent 0
        np.testing.assert_array_equal(pruned_skel.parents, np.array([-1, 0]))
        # All dead bone mass transferred to root (0)
        expected_w0 = weights[:, 0] + weights[:, 1] + weights[:, 2]
        np.testing.assert_allclose(pruned_w[:, 0], expected_w0, rtol=1e-5)
        np.testing.assert_allclose(pruned_w.sum(axis=1), np.ones(2), rtol=1e-6)

    def test_rejection_joint_count_bounds(self):
        # Reject J < 2
        skel1 = Skeleton(np.zeros((1, 3)), np.array([-1]))
        w1 = np.ones((5, 1))
        with self.assertRaises(RigValidationError):
            prune_dead_bones(w1, skel1)

        # Reject J > 64
        skel65 = Skeleton(np.zeros((65, 3)), np.array([-1] + [0] * 64))
        w65 = np.ones((5, 65)) / 65.0
        with self.assertRaises(RigValidationError):
            prune_dead_bones(w65, skel65)

        # Reject when pruning causes remaining joints to fall below 2
        skel2 = Skeleton(np.zeros((2, 3)), np.array([-1, 0]))
        w2 = np.array([[0.99, 0.01], [0.995, 0.005]])  # bone 1 dead -> only 1 bone left
        with self.assertRaises(RigValidationError):
            prune_dead_bones(w2, skel2)


class TestDFSOrdering(unittest.TestCase):
    def test_dfs_ordering_and_tie_breaking(self):
        # Root at (0, 0, 0)
        # Children of Root:
        # Child A at (x=10, y=1, z=0) -> (y=1, x=10, z=0)
        # Child B at (x=2,  y=2, z=0) -> (y=2, x=2,  z=0)
        # Child C at (x=1,  y=1, z=5) -> (y=1, x=1,  z=5)
        # Sibling tie-breaking lexicographically by (y, x, z):
        # For A and C: y is both 1. x for C is 1, x for A is 10.
        # Thus (1, 1, 5) < (1, 10, 0) < (2, 2, 0).
        # Expected child visit order from root: Child C, then Child A, then Child B!
        # Let's add child of C: D at (0, 5, 0)
        # DFS must visit C, then D (subtree of C), then A, then B.
        joints = np.array([
            [0.0, 0.0, 0.0],   # 0: Root
            [10.0, 1.0, 0.0],  # 1: Child A
            [2.0, 2.0, 0.0],   # 2: Child B
            [1.0, 1.0, 5.0],   # 3: Child C
            [0.0, 5.0, 0.0],   # 4: Child D (child of C)
        ], dtype=np.float64)
        parents = np.array([-1, 0, 0, 0, 3], dtype=np.int64)
        skel = Skeleton(joints, parents)

        weights = np.array([
            [0.2, 0.1, 0.3, 0.15, 0.25],
            [0.4, 0.2, 0.1, 0.2, 0.1],
        ], dtype=np.float64)

        new_skel, new_w, dfs_order = dfs_order_skeleton(skel, weights)

        # Expected order: Root (0) -> C (3) -> D (4) -> A (1) -> B (2)
        expected_order = [0, 3, 4, 1, 2]
        np.testing.assert_array_equal(dfs_order, expected_order)

        # Check permuted weights
        np.testing.assert_allclose(new_w, weights[:, expected_order])

        # Check joints
        np.testing.assert_allclose(new_skel.joints, joints[expected_order])

        # In DFS, root is index 0 (parent -1).
        # C is at index 1 (parent 0).
        # D is at index 2 (parent 1, which is C).
        # A is at index 3 (parent 0).
        # B is at index 4 (parent 0).
        expected_parents = np.array([-1, 0, 1, 0, 0], dtype=np.int64)
        np.testing.assert_array_equal(new_skel.parents, expected_parents)


class TestWeightTransferAndSmoothing(unittest.TestCase):
    def test_barycentric_weight_transfer(self):
        # A single triangle in XY plane:
        # V0: (0, 0, 0), V1: (1, 0, 0), V2: (0, 1, 0)
        raw_v = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)
        raw_f = np.array([[0, 1, 2]], dtype=np.int64)

        # 2 bones. Weights at raw vertices:
        raw_w = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float64)

        # Query clean vertices:
        # Q0 at centroid: (1/3, 1/3, 0) -> barycentric weights (1/3, 1/3, 1/3)
        # Expected weight: 1/3 * [1, 0] + 1/3 * [0, 1] + 1/3 * [0, 1] = [1/3, 2/3]
        # Q1 at V0: (0, 0, 0) -> barycentric (1, 0, 0) -> [1, 0]
        # Q2 at midpoint of edge 0-1: (0.5, 0, 0) -> barycentric (0.5, 0.5, 0) -> [0.5, 0.5]
        clean_v = np.array([
            [1.0 / 3.0, 1.0 / 3.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ], dtype=np.float64)

        clean_w = barycentric_weight_transfer(raw_v, raw_f, raw_w, clean_v)

        expected_w = np.array([
            [1.0 / 3.0, 2.0 / 3.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ])
        np.testing.assert_allclose(clean_w, expected_w, atol=1e-5)

    def test_manifold_edge_laplacian_smoothing(self):
        # Line graph of 3 vertices connected by 1 triangle:
        # 0 - 1 - 2
        faces = np.array([[0, 1, 2]])
        weights = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float64)

        # 1 step with lambda=0.5
        # Vertex 0 has neighbors {1, 2}. Average of neighbors: (w1 + w2) / 2 = [0, 1].
        # Smoothed w0 = 0.5 * [1, 0] + 0.5 * [0, 1] = [0.5, 0.5].
        # Vertex 1 has neighbors {0, 2}. Average: (w0 + w2) / 2 = ([1, 0] + [0, 1]) / 2 = [0.5, 0.5].
        # Smoothed w1 = 0.5 * [0, 1] + 0.5 * [0.5, 0.5] = [0.25, 0.75].
        # Vertex 2 has neighbors {0, 1}. Same as vertex 1: [0.25, 0.75].
        smoothed = laplacian_smooth_weights(weights, faces, steps=1, lambda_param=0.5)

        expected = np.array([
            [0.5, 0.5],
            [0.25, 0.75],
            [0.25, 0.75],
        ])
        np.testing.assert_allclose(smoothed, expected, atol=1e-5)
        np.testing.assert_allclose(smoothed.sum(axis=1), np.ones(3), rtol=1e-6)


class TestTop4TruncationAndGLTFPacking(unittest.TestCase):
    def test_top4_truncation(self):
        # 6 bones: vertex has influences across 6 bones
        weights = np.array([
            [0.40, 0.25, 0.15, 0.10, 0.06, 0.04],  # top 4: 0, 1, 2, 3 (sum = 0.90)
            [0.05, 0.50, 0.05, 0.20, 0.10, 0.10],  # top 4: 1 (0.50), 3 (0.20), 4 (0.10), 5 (0.10) (sum = 0.90)
        ], dtype=np.float64)

        res = truncate_top4_weights(weights)
        joints_0 = res['JOINTS_0']
        weights_0 = res['WEIGHTS_0']

        self.assertEqual(joints_0.dtype, np.uint16)
        self.assertEqual(weights_0.dtype, np.float32)
        self.assertEqual(joints_0.shape, (2, 4))
        self.assertEqual(weights_0.shape, (2, 4))

        # Check row 0
        np.testing.assert_array_equal(joints_0[0], [0, 1, 2, 3])
        expected_w0 = np.array([0.40, 0.25, 0.15, 0.10]) / 0.90
        np.testing.assert_allclose(weights_0[0], expected_w0, atol=1e-5)
        np.testing.assert_allclose(weights_0.sum(axis=1), [1.0, 1.0], atol=1e-6)

        # Check row 1: top indices [1, 3, 4, 5]
        np.testing.assert_array_equal(joints_0[1], [1, 3, 4, 5])
        expected_w1 = np.array([0.50, 0.20, 0.10, 0.10]) / 0.90
        np.testing.assert_allclose(weights_0[1], expected_w1, atol=1e-5)

    def test_pack_gltf_skin_accessors(self):
        j = np.array([[0, 1, 2, 3]], dtype=np.uint16)
        w = np.array([[0.5, 0.3, 0.1, 0.1]], dtype=np.float32)

        accessors = pack_gltf_skin_accessors(j, w)
        self.assertIn('JOINTS_0', accessors)
        self.assertIn('WEIGHTS_0', accessors)

        self.assertEqual(accessors['JOINTS_0']['componentType'], 5123)
        self.assertEqual(accessors['JOINTS_0']['type'], 'VEC4')
        self.assertEqual(len(accessors['JOINTS_0']['bytes']), 4 * 2)  # 4 uint16 = 8 bytes

        self.assertEqual(accessors['WEIGHTS_0']['componentType'], 5126)
        self.assertEqual(accessors['WEIGHTS_0']['type'], 'VEC4')
        self.assertEqual(len(accessors['WEIGHTS_0']['bytes']), 4 * 4)  # 4 float32 = 16 bytes


class TestPartNetURDFConverter(unittest.TestCase):
    def test_parse_partnet_urdf(self):
        urdf_content = """<?xml version="1.0" ?>
<robot name="cabinet_sample">
    <link name="base">
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="1.0 1.0 1.0"/>
            </geometry>
        </visual>
    </link>
    <link name="door">
        <visual>
            <origin xyz="0.0 0.5 0.0" rpy="0 0 0"/>
            <geometry>
                <box size="0.1 0.5 0.8"/>
            </geometry>
        </visual>
    </link>
    <link name="drawer">
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="0.4 0.4 0.2"/>
            </geometry>
        </visual>
    </link>
    <joint name="door_hinge" type="revolute">
        <parent link="base"/>
        <child link="door"/>
        <origin xyz="0.5 0.0 0.0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="0.0" upper="1.57"/>
    </joint>
    <joint name="drawer_slide" type="prismatic">
        <parent link="base"/>
        <child link="drawer"/>
        <origin xyz="0.0 0.0 0.5" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="0.0" upper="0.5"/>
    </joint>
</robot>
"""
        res = parse_partnet_urdf(urdf_content)

        self.assertIn('vertices', res)
        self.assertIn('faces', res)
        self.assertIn('weights', res)
        self.assertIn('skeleton', res)
        self.assertIn('glb_bytes', res)

        skel = res['skeleton']
        self.assertEqual(len(skel.joints), 3)  # base, door, drawer
        # Binary rigid skinning: each vertex must have weight exactly 1.0 for one joint
        w = res['weights']
        self.assertEqual(w.shape[1], 3)
        np.testing.assert_allclose(w.sum(axis=1), np.ones(len(w)))
        # Verify binary weights (each entry is either 0 or 1)
        self.assertTrue(np.all((w == 0.0) | (w == 1.0)))

        # Verify joint axes:
        # door hinge is around Z: (0, 0, 1)
        np.testing.assert_allclose(res['joint_axes'][1], [0.0, 0.0, 1.0], atol=1e-5)
        # drawer slide is along X: (1, 0, 0)
        np.testing.assert_allclose(res['joint_axes'][2], [1.0, 0.0, 0.0], atol=1e-5)

        # Verify the emitted GLB is valid and can be loaded by load_rig_glb
        with tempfile.NamedTemporaryFile(suffix='.glb') as tf:
            tf.write(res['glb_bytes'])
            tf.flush()
            rig = load_rig_glb(tf.name)
            self.assertEqual(len(rig.skeleton.joints), 3)
            self.assertEqual(len(rig.vertices), len(res['vertices']))


class TestCanonicalizeRigPipeline(unittest.TestCase):
    def test_canonicalize_rig_full_pipeline(self):
        # Setup synthetic mesh with 3 bones: 0 (root), 1 (alive), 2 (dead IK bone)
        # Cube with 8 vertices, 12 faces
        raw_v = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=np.float64)
        raw_f = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 1], [1, 4, 5],
            [2, 6, 3], [3, 6, 7],
            [0, 3, 4], [3, 7, 4],
            [1, 5, 2], [2, 5, 6]
        ], dtype=np.int64)

        # Skeleton: 0 -> 1 -> 2
        # Joint 1 at (0, 2, 0), Joint 2 at (0, 3, 0)
        skel = Skeleton(
            np.array([[0, 0, 0], [0, 2, 0], [0, 3, 0]], dtype=np.float64),
            np.array([-1, 0, 1], dtype=np.int64)
        )

        # Raw weights: bone 2 has max weight 0.005 (< 0.02) -> should be pruned!
        raw_w = np.array([
            [0.60, 0.395, 0.005],
            [0.55, 0.445, 0.005],
            [0.50, 0.495, 0.005],
            [0.65, 0.345, 0.005],
            [0.40, 0.595, 0.005],
            [0.45, 0.545, 0.005],
            [0.35, 0.645, 0.005],
            [0.30, 0.695, 0.005],
        ], dtype=np.float64)

        # Watertight clean mesh: slightly perturbed / remeshed vertices
        clean_v = raw_v + 0.01
        clean_f = raw_f.copy()

        result = canonicalize_rig(
            raw_v, raw_w, skel, clean_v, clean_f, raw_faces=raw_f,
            prune_threshold=0.02, laplacian_steps=1, laplacian_lambda=0.5
        )

        self.assertEqual(result['pruned_joints'], [2])
        final_skel = result['skeleton']
        self.assertEqual(len(final_skel.joints), 2)
        self.assertEqual(result['JOINTS_0'].shape, (len(clean_v), 4))
        self.assertEqual(result['WEIGHTS_0'].shape, (len(clean_v), 4))
        np.testing.assert_allclose(result['WEIGHTS_0'].sum(axis=1), np.ones(len(clean_v)), atol=1e-5)

        # Validate produced GLB
        with tempfile.NamedTemporaryFile(suffix='.glb') as tf:
            tf.write(result['glb_bytes'])
            tf.flush()
            rig = load_rig_glb(tf.name)
            self.assertEqual(len(rig.skeleton.joints), 2)


if __name__ == '__main__':
    unittest.main()
