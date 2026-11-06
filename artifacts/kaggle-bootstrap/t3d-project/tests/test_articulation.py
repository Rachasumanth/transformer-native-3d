import unittest
import numpy as np
from t3d.articulation import Skeleton, skeleton_displacements, skin_embeddings, attach_articulation, linear_blend_skinning, transfer_skin_weights
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
