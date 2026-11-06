import unittest
import numpy as np
from t3d.serialization import morton_keys, morton_decode, serialization_order
from t3d.schema import VoxelAsset


class SerializationTests(unittest.TestCase):
    def test_known_bits(self):
        np.testing.assert_array_equal(morton_keys([[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0]], depth=2), [1, 2, 4, 8])

    def test_exhaustive_small_grid_bijection(self):
        c = np.indices((8, 8, 8)).reshape(3, -1).T
        for transposed in (False, True):
            keys = morton_keys(c, depth=3, transposed=transposed)
            np.testing.assert_array_equal(np.sort(keys), np.arange(512))
            decoded, batch = morton_decode(keys, depth=3, transposed=transposed)
            np.testing.assert_array_equal(decoded, c)
            np.testing.assert_array_equal(batch, 0)

    def test_full_uint64_boundary(self):
        c = [[2**21-1]*3]
        keys = morton_keys(c, depth=21, batch=[1])
        self.assertEqual(int(keys[0]), 2**64-1)
        decoded, batch = morton_decode(keys, depth=21)
        np.testing.assert_array_equal(decoded, c)
        np.testing.assert_array_equal(batch, [1])
        with self.assertRaises(ValueError):
            morton_keys(c, depth=21, batch=[2])

    def test_indirect_mapping_and_duplicates(self):
        c = [[1, 1, 1], [0, 0, 0], [1, 0, 0]]
        order, inverse = serialization_order(c, depth=2)
        features = np.arange(12).reshape(3, 4)
        np.testing.assert_array_equal(features[order][inverse], features)
        with self.assertRaises(ValueError):
            serialization_order([[0, 0, 0]]*2, depth=2)
        serialization_order([[0, 0, 0]]*2, depth=2, batch=[0, 1])
        for coords in ([[4, 0, 0]], [[-1, 0, 0]], [[1., 0., 0.]]):
            with self.assertRaises(ValueError):
                morton_keys(coords, depth=2)


class SchemaTests(unittest.TestCase):
    def asset(self, **changes):
        values = dict(coords=np.array([[1, 0, 0], [0, 0, 0]]),
                      dual_vertices=np.full((2, 3), .5),
                      intersected_flag=np.array([[True, False, False], [False, True, False]]),
                      grid_size=(2, 2, 2), aabb=[[-.5]*3, [.5]*3],
                      pbr=[[1, 0, 0, 0, .3, 1], [0, 1, 0, 1, .8, .5]],
                      skeleton_field=np.arange(12).reshape(2, 6),
                      skin_embedding=np.arange(8).reshape(2, 4))
        values.update(changes)
        return VoxelAsset(**values)

    def test_sort_keeps_all_fields_aligned(self):
        a = self.asset()
        b = a.canonical()
        for field in ('coords', 'dual_vertices', 'intersected_flag', 'pbr', 'skeleton_field', 'skin_embedding'):
            np.testing.assert_array_equal(getattr(b, field), getattr(a, field)[::-1])
        np.testing.assert_allclose(a.world_vertices, [[.25, -.25, -.25], [-.25, -.25, -.25]])
        np.testing.assert_allclose(a.voxel_size, [.5]*3)

    def test_rejects_corrupt_attributes(self):
        for change in (dict(coords=[[2, 0, 0], [0, 0, 0]]),
                       dict(coords=[[0, 0, 0]]*2), dict(pbr=np.full((2, 6), 1.1)),
                       dict(dual_vertices=None), dict(dual_vertices=np.full((2, 3), np.nan)),
                       dict(intersected_flag=np.ones((2, 3))), dict(skin_embedding=np.zeros((2, 3))),
                       dict(aabb=[[0]*3, [0]*3])):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.asset(**change)

    def test_input_ownership_and_empty_asset(self):
        c = np.array([[1, 0, 0], [0, 0, 0]])
        a = self.asset(coords=c)
        c[:] = 0
        self.assertEqual(a.coords[0, 0], 1)
        with self.assertRaises(ValueError):
            a.coords[0, 0] = 0
        a = VoxelAsset(np.empty((0, 3), int), np.empty((0, 3)), np.empty((0, 3), bool),
                       (1536,)*3, [[-.5]*3, [.5]*3])
        self.assertEqual(a.canonical().world_vertices.shape, (0, 3))
