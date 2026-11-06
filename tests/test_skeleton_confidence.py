import unittest
import numpy as np
from t3d.articulation import Skeleton, skeleton_confidence


class ConfidenceTests(unittest.TestCase):
    def test_joint_midpoint_and_ratio(self):
        rig = Skeleton([[0, 0, 0], [2, 0, 0]], [-1, 0])
        actual = skeleton_confidence([[0, 0, 0], [1, 0, 0], [.5, 0, 0]], rig)
        np.testing.assert_allclose(actual, [1, 0, 8/9])

    def test_translation_scale_and_chunks(self):
        joints = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0.]])
        points = np.array([[.3, .4, 0], [2, 1, 0], [1, 0, 0.]])
        expected = skeleton_confidence(points, Skeleton(joints, [-1, 0, 0]))
        shifted = skeleton_confidence(points*3+7, Skeleton(joints*3+7, [-1, 0, 0]), chunk_size=1)
        np.testing.assert_allclose(shifted, expected)

    def test_coincident_joints_singleton_and_empty(self):
        duplicate = Skeleton([[0, 0, 0], [0, 0, 0]], [-1, 0])
        np.testing.assert_array_equal(skeleton_confidence([[0, 0, 0], [1, 0, 0]], duplicate), [0, 0])
        single = Skeleton([[0, 0, 0]], [-1])
        np.testing.assert_array_equal(skeleton_confidence([[0, 0, 0], [9, 0, 0]], single), [1, 1])
        self.assertEqual(skeleton_confidence(np.empty((0, 3)), single).shape, (0,))
        with self.assertRaises(ValueError):
            skeleton_confidence([[0, 0, 0]], single, chunk_size=0)


if __name__ == '__main__':
    unittest.main()
