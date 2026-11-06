import unittest
import numpy as np
from t3d.topology import orient_faces


def assert_consistent(test, faces):
    edges = {}
    for tri in faces:
        for a, b in zip(tri, np.roll(tri, -1)):
            edges.setdefault(tuple(sorted((a, b))), []).append((a, b))
    for owners in edges.values():
        if len(owners) == 2:
            test.assertEqual(owners[0], owners[1][::-1])


class OrientationTests(unittest.TestCase):
    v = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])

    def test_scrambled_closed_component(self):
        source = self.f.copy()
        source[[0, 2]] = source[[0, 2]][:, [0, 2, 1]]
        result, report = orient_faces(self.v, source, outward=True)
        assert_consistent(self, result)
        np.testing.assert_array_equal(np.sort(result, axis=1), np.sort(source, axis=1))
        self.assertAlmostEqual(report['components'][0]['signed_volume'], 1/6)
        self.assertTrue(report['components'][0]['closed'])
        again, _ = orient_faces(self.v, result, outward=True)
        np.testing.assert_array_equal(again, result)

    def test_open_sheet_preserves_seed(self):
        v = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        source = [[0, 1, 2], [0, 3, 2]]
        f, report = orient_faces(v, source, outward=True)
        assert_consistent(self, f)
        np.testing.assert_array_equal(f[0], source[0])
        self.assertFalse(report['components'][0]['closed'])
        self.assertIsNone(report['components'][0]['signed_volume'])

    def test_disconnected_components_and_translation(self):
        v = np.concatenate([self.v, self.v + [10, 0, 0]]) + [1e8, -1e8, 1e8]
        f = np.concatenate([self.f, self.f[:, [0, 2, 1]] + 4])
        result, report = orient_faces(v, f, outward=True)
        self.assertEqual(len(report['components']), 2)
        for c in report['components']:
            self.assertAlmostEqual(c['signed_volume'], 1/6)
        assert_consistent(self, result)

    def test_duplicate_and_nonmanifold_rejection(self):
        with self.assertRaises(ValueError):
            orient_faces(self.v, np.concatenate([self.f, self.f[:1]]))
        v = np.concatenate([self.v, [[0, -1, 0]]])
        with self.assertRaises(ValueError):
            orient_faces(v, [[0, 1, 2], [1, 0, 3], [0, 1, 4]])

    def test_nonorientable_strip_rejection(self):
        # Triangulated strip with a half twist at the identified ends.
        rng = np.random.default_rng(4)
        v = rng.random((6, 3))
        f = [[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4], [4, 5, 1], [5, 0, 1]]
        with self.assertRaisesRegex(ValueError, 'non-orientable'):
            orient_faces(v, f)
