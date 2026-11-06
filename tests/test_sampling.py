import unittest
import numpy as np
from t3d.sampling import sample_conditioning, salient_edges, farthest_point_indices
from t3d.augmentation import quarter_turn


class SamplingTests(unittest.TestCase):
    # Two consistently oriented triangles at a right-angle crease.
    v = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    f = [(0, 1, 2), (1, 0, 3)]

    def test_dihedral_and_interpolation(self):
        np.testing.assert_array_equal(salient_edges(self.v, self.f), [[0, 1]])
        uniform, sharp = sample_conditioning(self.v, self.f, uniform_count=20, salient_count=7)
        self.assertEqual(sharp.shape, (7, 6))
        np.testing.assert_allclose(sharp[:, 1:3], 0)
        np.testing.assert_allclose(sharp[:, 0], [0, 1, 1/6, 2/6, 3/6, 4/6, 5/6])
        for points in (uniform, sharp):
            np.testing.assert_allclose(np.linalg.norm(points[:, 3:], axis=1), 1)
            self.assertTrue(np.all(points[:, :3] >= 0))
            self.assertTrue(np.all(points[:, :3].sum(axis=1) <= 1 + 1e-12))

    def test_flat_surface_has_no_sharp_samples(self):
        v = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
        uniform, sharp = sample_conditioning(v, [(0, 1, 2), (1, 3, 2)], uniform_count=10)
        self.assertEqual(sharp.shape, (0, 6))
        np.testing.assert_allclose(uniform[:, 2], 0)

    def test_seed_reproducibility_and_fps(self):
        a = sample_conditioning(self.v, self.f, uniform_count=30, salient_count=1, seed=7)
        b = sample_conditioning(self.v, self.f, uniform_count=30, salient_count=1, seed=7)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(farthest_point_indices([(0, 0, 0), (1, 0, 0), (3, 0, 0)], 3), [0, 2, 1])
        self.assertEqual(len(set(farthest_point_indices(np.zeros((3, 3)), 3))), 3)

    def test_uniform_samples_follow_area_and_centroid(self):
        # Disconnected triangles have areas in ratio 1:4.
        v = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (10, 0, 0), (12, 0, 0), (10, 2, 0)]
        p, _ = sample_conditioning(v, [(0, 1, 2), (3, 4, 5)], uniform_count=20000, salient_count=0)
        small = p[:, 0] < 2
        self.assertAlmostEqual(small.mean(), .2, delta=.015)
        np.testing.assert_allclose(p[small, :2].mean(axis=0), [1/3, 1/3], atol=.015)

    def test_invalid_input(self):
        for faces in ([(0, 1, 99)], [(0, 0, 1)], [(0., 1., 2.)]):
            with self.assertRaises(ValueError):
                sample_conditioning(self.v, faces)
        with self.assertRaises(ValueError):
            sample_conditioning(self.v, self.f, uniform_count=-1)
        with self.assertRaises(ValueError):
            salient_edges(self.v, self.f, threshold=-1)

    def test_rotations(self):
        for axis in ('x', 'y', 'z'):
            for turns in range(4):
                points = quarter_turn(self.v, turns, axis=axis)
                self.assertEqual(quarter_turn(points, -turns, axis=axis), tuple(self.v))
                np.testing.assert_allclose(np.linalg.norm(points, axis=1), np.linalg.norm(self.v, axis=1))
        self.assertEqual(quarter_turn([(1, 0, 0)], 1), ((0., 0., -1.),))
