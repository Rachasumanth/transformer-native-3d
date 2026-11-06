import unittest
import numpy as np
from t3d.serialization import hilbert_keys, hilbert_decode, serialization_orders
from t3d.container import dumps, loads
from t3d.schema import VoxelAsset


class HilbertTests(unittest.TestCase):
    def test_exhaustive_bijection_and_adjacency(self):
        c = np.indices((8,)*3).reshape(3, -1).T
        for transposed in (False, True):
            keys = hilbert_keys(c, depth=3, transposed=transposed)
            np.testing.assert_array_equal(np.sort(keys), np.arange(512))
            decoded, batch = hilbert_decode(keys, depth=3, transposed=transposed)
            np.testing.assert_array_equal(decoded, c)
            np.testing.assert_array_equal(batch, 0)
            ordered = c[np.argsort(keys)]
            np.testing.assert_array_equal(np.abs(np.diff(ordered, axis=0)).sum(axis=1), 1)

    def test_batch_high_bits_and_empty(self):
        keys = np.array([0, 2**64-1], dtype=np.uint64)
        c, b = hilbert_decode(keys, depth=21)
        np.testing.assert_array_equal(hilbert_keys(c, depth=21, batch=b), keys)
        c, b = hilbert_decode(np.empty(0, np.uint64), depth=11)
        self.assertEqual(c.shape, (0, 3))
        self.assertEqual(hilbert_keys(c, depth=11).shape, (0,))
        with self.assertRaises(ValueError):
            hilbert_keys([[2048, 0, 0]], depth=11)

    def test_four_mappings_and_storage(self):
        c = np.indices((4,)*3).reshape(3, -1).T[::-1]
        maps = serialization_orders(c, depth=2)
        self.assertEqual(set(maps), {'z', 'tz', 'h', 'th'})
        a = VoxelAsset(c, np.full(c.shape, .5), np.zeros(c.shape, bool), (4,)*3, [[0]*3, [1]*3])
        for name, (order, inverse) in maps.items():
            np.testing.assert_array_equal(c[order][inverse], c)
            data = dumps(a, curve=name)
            restored = loads(data)
            self.assertEqual(restored.curve, name)
            np.testing.assert_array_equal(restored.asset.coords, c[order])
            self.assertEqual(dumps(restored.asset, curve=name), data)
