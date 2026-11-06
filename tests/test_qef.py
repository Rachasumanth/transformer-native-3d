import unittest
import numpy as np
from t3d.qef import solve_qef


class QefTests(unittest.TestCase):
    def test_plane_crease_and_corner(self):
        for points, normals, expected in (
            ([[.2, .1, .3], [.2, .7, .9]], [[1, 0, 0]]*2, [.2, .4, .6]),
            ([[.2, .1, .3], [.1, .4, .9]], [[1, 0, 0], [0, 1, 0]], [.2, .4, .6]),
            ([[.2, 0, 0], [0, .4, 0], [0, 0, .8]], np.eye(3), [.2, .4, .8]),
        ):
            np.testing.assert_allclose(solve_qef(points, normals, regularization_weight=0), expected, atol=1e-12)

    def test_bounded_minimum_beats_clipping(self):
        # Planes x+y=1.8 and x=.1 have an unconstrained intersection outside cube.
        p = [[.9, .9, .5], [.1, .5, .5]]
        n = [[1, 1, 0], [1, 0, 0]]
        x = solve_qef(p, n, regularization_weight=0)
        np.testing.assert_allclose(x, [1/3, 1, .5], atol=1e-12)
        def error(q):
            return .5*(q[0]+q[1]-1.8)**2 + (q[0]-.1)**2
        self.assertLess(error(x), error([.1, 1, .5]))

    def test_regularizer_and_boundary_weights(self):
        p, n = [[.2, .6, .8]], [[1, 0, 0]]
        x = solve_qef(p, n, boundary_points=[[1, .6, .8]], boundary_normals=[[1, 0, 0]],
                      boundary_weight=2, regularization_weight=1)
        np.testing.assert_allclose(x, [.6, .6, .8])

    def test_normal_scaling_and_input_order(self):
        p = np.array([[.2, .1, .3], [.1, .4, .9], [.1, .2, .7]])
        n = np.eye(3)
        expected = solve_qef(p, n)
        np.testing.assert_allclose(solve_qef(p[::-1], n[::-1]), expected)
        np.testing.assert_allclose(solve_qef(p, n * np.array([1e-250, 1e250, -3])[:, None]), expected)

    def test_validation(self):
        for p, n, kwargs in (([], [], {}), ([[0, 0, 0]], [[0, 0, 0]], {}),
                             ([[2, 0, 0]], [[1, 0, 0]], {}),
                             ([[0, 0, 0]], [[1, 0, 0]], {'face_weight': -1}),
                             ([[0, 0, 0]], [[1, 0, 0]], {'boundary_points': [[0, 0, 0]]})):
            with self.assertRaises(ValueError):
                solve_qef(p, n, **kwargs)
