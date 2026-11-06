import unittest
from t3d.normalization import normalize_asset


class NormalizationTests(unittest.TestCase):
    def test_shared_transform_and_inverse(self):
        vertices = [(10, 0, 2), (14, 2, 2), (12, 1, 2)]
        joints = [(12, 1, 2), (12, 2, 2)]
        mesh, rig, transform = normalize_asset(vertices, joints)
        self.assertEqual(mesh, ((-.5, -.25, 0), (.5, .25, 0), (0, 0, 0)))
        self.assertEqual(rig, ((0, 0, 0), (0, .25, 0)))
        self.assertEqual(transform.inverse(mesh), tuple(vertices))
        self.assertEqual(transform.inverse(rig), tuple(joints))

    def test_vertex_permutation_does_not_change_transform(self):
        vertices = [(0, 0, 0), (3, 2, 1), (1, 0, 1)]
        a, _, transform = normalize_asset(vertices)
        b, _, other = normalize_asset(reversed(vertices))
        self.assertEqual(transform, other)
        self.assertEqual(a, tuple(reversed(b)))

    def test_rejects_invalid_geometry(self):
        for vertices in ([], [(0, 0, 0)], [(0, 0)], [(0, 0, float('nan'))],
                         [(0, 0, 0), (float('inf'), 1, 1)]):
            with self.subTest(vertices=vertices), self.assertRaises(ValueError):
                normalize_asset(vertices)

    def test_evaluation_bounds_and_unclipped_joints(self):
        mesh, rig, _ = normalize_asset([(0, 0, 0), (2, 2, 2)], [(4, 1, 1)], half_extent=1)
        self.assertEqual(mesh, ((-1, -1, -1), (1, 1, 1)))
        self.assertEqual(rig, ((3, 0, 0),))
