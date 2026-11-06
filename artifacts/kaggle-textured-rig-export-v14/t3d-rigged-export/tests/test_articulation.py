import unittest
import numpy as np
from t3d.articulation import Skeleton, skeleton_displacements, infer_skeleton_from_field, skin_embeddings, attach_articulation, linear_blend_skinning, joint_deformation_matrices, transfer_skin_weights
from t3d.schema import VoxelAsset
from t3d.container import dumps, loads


class ArticulationTests(unittest.TestCase):
    def test_hierarchy_arbitrary_order_and_cycles(self):
        s = Skeleton([[0, 1, 0], [0, 0, 0], [1, 1, 0]], [1, -1, 0])
        np.testing.assert_allclose(s.parent_relative_vectors, [[0, 1, 0], [0, 0, 0], [1, 0, 0]])
        for parents in ([0, -1], [1, 0], [-2, -1], [2, -1]):
            with self.assertRaises(ValueError):
                Skeleton([[0, 0, 0], [1, 0, 0]], parents)

    def test_displacement_root_tie_and_chunking(self):
        s = Skeleton([[0, 0, 0], [1, 0, 0]], [-1, 0])
        points = [[.8, .2, 0], [.5, 0, 0], [0, 0, 0]]
        field, ids = skeleton_displacements(points, s, chunk_size=1)
        np.testing.assert_array_equal(ids, [1, 0, 0])
        np.testing.assert_allclose(field, [[.2, -.2, 0, -.8, -.2, 0], [-.5, 0, 0, -.5, 0, 0], [0]*6])
        np.testing.assert_allclose(skeleton_displacements(points, s)[0], field)
        self.assertEqual(skeleton_displacements(np.empty((0, 3)), s)[0].shape, (0, 6))

    def test_embedding_and_container(self):
        s = Skeleton([[0, 0, 0], [1, 0, 0]], [-1, 0])
        e = [[1, 0, 0, 0], [0, 1, 0, 1]]
        np.testing.assert_allclose(skin_embeddings([[.25, .75]], e), [[.25, .75, 0, .75]])
        a = VoxelAsset([[0, 0, 0]], [[.1, .1, .1]], np.zeros((1, 3), bool), (1,)*3, [[0]*3, [1]*3])
        rigged = attach_articulation(a, s, voxel_weights=[[.25, .75]], joint_embeddings=e)
        # Field uses cell center, not the dual vertex position.
        np.testing.assert_allclose(rigged.skeleton_field, [[-.5]*6])
        restored = loads(dumps(rigged)).asset
        np.testing.assert_allclose(restored.skin_embedding, [[.25, .75, 0, .75]])
        np.testing.assert_allclose(restored.skeleton_field, rigged.skeleton_field)

    def test_infer_skeleton_from_exact_and_noisy_fields(self):
        expected=Skeleton([[.25,.25,.25],[.75,.25,.25],[.75,.75,.25]],[-1,0,1])
        points=np.array([[x,y,z] for x in [.1,.3,.6,.8]
                         for y in [.1,.3,.6,.8] for z in [.1,.3]])
        field,_=skeleton_displacements(points,expected)
        exact=infer_skeleton_from_field(points,field,tolerance=1e-6)
        np.testing.assert_array_equal(exact.parents,expected.parents)
        np.testing.assert_allclose(exact.joints,expected.joints,atol=1e-12)
        noisy=field+np.random.default_rng(12).normal(0,.0005,field.shape)
        predicted=infer_skeleton_from_field(points,noisy,tolerance=.01)
        np.testing.assert_array_equal(predicted.parents,expected.parents)
        np.testing.assert_allclose(predicted.joints,expected.joints,atol=.001)
        with self.assertRaisesRegex(ValueError,'max_joints'):
            infer_skeleton_from_field(points,field,tolerance=1e-6,max_joints=2)

    def test_infer_skeleton_from_float16_container_precision(self):
        expected=Skeleton([[-.2,-.1,0],[.2,.1,0]],[-1,0])
        points=np.random.default_rng(7).uniform(-.5,.5,(2048,3))
        field,_=skeleton_displacements(points,expected)
        # Raw container attributes are stored as float16. The default tolerance
        # is one quarter of a 32-grid voxel over a unit AABB.
        restored_field=field.astype(np.float16).astype(float)
        predicted=infer_skeleton_from_field(points,restored_field,tolerance=1/128)
        np.testing.assert_array_equal(predicted.parents,expected.parents)
        np.testing.assert_allclose(predicted.joints,expected.joints,atol=1e-4)

    def test_joint_deformation_matrices_bind_relative(self):
        bind = np.tile(np.eye(4), (2, 1, 1))
        bind[1, :3, 3] = [1, 0, 0]
        pose = bind.copy()
        theta = np.pi / 2
        pose[0, :3, :3] = [[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        pose[0, :3, 3] = [0, 1, 0]
        deformation = joint_deformation_matrices(bind, pose)
        np.testing.assert_allclose(deformation[0] @ [1, 0, 0, 1], [0, 2, 0, 1], atol=1e-12)
        np.testing.assert_allclose(deformation[1], np.eye(4), atol=1e-12)
        for bad_bind, bad_pose in (
                (bind[:1], pose),
                (np.tile(np.zeros((4, 4)), (2, 1, 1)), pose),
                (bind, np.tile(np.eye(3), (2, 1, 1)))):
            with self.assertRaises(ValueError):
                joint_deformation_matrices(bad_bind, bad_pose)

    def test_deformation_identity_translation_rotation(self):
        matrices = np.tile(np.eye(4), (2, 1, 1))
        v, w = [[1, 0, 0]], [[.25, .75]]
        np.testing.assert_allclose(linear_blend_skinning(v, w, matrices), v)
        matrices[1, :3, 3] = [0, 2, 0]
        np.testing.assert_allclose(linear_blend_skinning(v, w, matrices), [[1, 1.5, 0]])
        matrices[1] = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        np.testing.assert_allclose(linear_blend_skinning(v, [[0, 1]], matrices), [[0, 1, 0]])

    def test_weights_validation(self):
        for w in ([[.2, .2]], [[-.1, 1.1]], [[float('nan'), 0]], [[1]]):
            with self.assertRaises(ValueError):
                skin_embeddings(w, np.ones((2, 4)))

    def test_source_weight_transfer(self):
        v = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        weights = [[1, 0], [0, 1], [0, 1]]
        result = transfer_skin_weights([[.2, .3, 1], [0, 0, 0]], v, [[0, 1, 2]], weights)
        np.testing.assert_allclose(result, [[.5, .5], [1, 0]])
        np.testing.assert_allclose(result.sum(axis=1), 1)
