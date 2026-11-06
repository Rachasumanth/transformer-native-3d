import unittest
import numpy as np
from t3d.bpt import encode, decode, quantize


def triangles(vertices, faces):
    q = quantize(vertices)
    result = []
    for f in faces:
        coords = [tuple(q[i]) for i in f]
        result.append(min(tuple(coords[i:]+coords[:i]) for i in range(3)))
    return sorted(result)


class BptTests(unittest.TestCase):
    def test_fan_roundtrip_and_compression(self):
        angles = np.arange(12)*2*np.pi/12
        v = np.vstack(([0, 0, 0], np.column_stack((.4*np.cos(angles), .4*np.sin(angles), np.zeros(12)))))
        f = np.array([[0, i+1, (i+1)%12+1] for i in range(12)])
        tokens = encode(v, f)
        restored_v, restored_f = decode(tokens)
        self.assertEqual(triangles(v, f), triangles(restored_v, restored_f))
        self.assertLess(len(tokens), 9*len(f))
        self.assertTrue(np.all(tokens < 5120))
        permutation = np.random.default_rng(7).permutation(len(v))
        inv = np.argsort(permutation)
        np.testing.assert_array_equal(encode(v[permutation], inv[f[::-1]]), tokens)

    def test_disconnected_and_branching_fans(self):
        v = [[0, 0, 0], [.4, 0, 0], [0, .4, 0], [0, 0, .4], [0, -.4, 0]]
        f = [[0, 1, 2], [0, 1, 3], [0, 1, 4]]
        a, b = decode(encode(v, f))
        self.assertEqual(triangles(v, f), triangles(a, b))

    def test_invalid_streams(self):
        for tokens in ([1024], [512, 1024], [0], [0, 1024], [0, 512], [0, 1024, 1024, 1024], [5120]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                decode(np.array(tokens))
        self.assertEqual(decode(np.empty(0, dtype=int))[0].shape, (0, 3))

    def test_quantization_collapse(self):
        with self.assertRaises(ValueError):
            encode([[0, 0, 0], [.001, 0, 0], [0, .001, 0]], [[0, 1, 2]])
        np.testing.assert_array_equal(quantize([[-.5]*3, [.5]*3]), [[0]*3, [127]*3])
