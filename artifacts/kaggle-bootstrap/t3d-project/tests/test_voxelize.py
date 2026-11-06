import unittest
import numpy as np
from t3d.voxelize import mesh_to_voxels


class VoxelizationTests(unittest.TestCase):
    def convert(self, v, f, **kwargs):
        options = dict(grid_size=(4,)*3, aabb=[[0]*3, [1]*3], regularization_weight=0)
        options.update(kwargs)
        return mesh_to_voxels(v, f, **options)

    def test_open_plane_and_edge_flags(self):
        v = [[.4, 0, 0], [.4, 1, 0], [.4, 1, 1], [.4, 0, 1]]
        a = self.convert(v, [[0, 1, 2], [0, 2, 3]])
        self.assertEqual(len(a.coords), 16)
        np.testing.assert_allclose(a.world_vertices[:, 0], .4, atol=1e-12)
        self.assertTrue(a.intersected_flag[:, 0].all())
        self.assertFalse(a.intersected_flag[:, 1:].any())
        self.assertTrue(np.all(a.coords[:, 0] == 1))

    def test_subcell_triangle_without_grid_edge_crossings(self):
        a = self.convert([[.31, .31, .35], [.4, .31, .35], [.31, .4, .35]], [[0, 1, 2]])
        self.assertEqual(len(a.coords), 1)
        self.assertFalse(a.intersected_flag.any())
        np.testing.assert_allclose(a.world_vertices[0, 2], .35)

    def test_disconnected_internal_surfaces_survive(self):
        v = [[.15, .1, .1], [.15, .2, .1], [.15, .1, .2],
             [.85, .8, .8], [.85, .9, .8], [.85, .8, .9]]
        a = self.convert(v, [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(len(a.coords), 2)
        np.testing.assert_allclose(np.sort(a.world_vertices[:, 0]), [.15, .85])

    def test_anisotropic_bounds_and_face_reversal(self):
        v = [[.4, 0, 0], [.4, 1, 0], [.4, 0, 1]]
        a = self.convert(v, [[0, 1, 2]])
        b = self.convert(v, [[2, 1, 0]])
        np.testing.assert_array_equal(a.coords, b.coords)
        np.testing.assert_allclose(a.world_vertices, b.world_vertices, atol=1e-12)
        c = self.convert(np.array(v) * [2, 3, 4], [[0, 1, 2]], aabb=[[0]*3, [2, 3, 4]])
        np.testing.assert_allclose(c.world_vertices, a.world_vertices * [2, 3, 4], atol=1e-12)

    def test_bounds_validation(self):
        with self.assertRaises(ValueError):
            self.convert([[2, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 2]])
