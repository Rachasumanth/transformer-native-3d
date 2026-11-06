import struct
import tempfile
from pathlib import Path
import unittest
import numpy as np
from t3d.container import dumps, load, loads, save
from t3d.schema import VoxelAsset


class ContainerTests(unittest.TestCase):
    def asset(self, **kw):
        fields = dict(coords=[[1, 0, 0], [0, 0, 0]], dual_vertices=[[.2]*3, [.8]*3],
                      intersected_flag=np.array([[1, 0, 0], [0, 1, 0]], bool),
                      grid_size=(4,)*3, aabb=[[-.5]*3, [.5]*3],
                      pbr=np.full((2, 6), .3), skeleton_field=np.arange(12).reshape(2, 6),
                      skin_embedding=np.arange(8).reshape(2, 4), split_weight=[[1], [2]])
        fields.update(kw)
        return VoxelAsset(**fields)

    def test_roundtrip_all_attributes_and_bpt(self):
        source = self.asset().canonical()
        data = dumps(source, bpt_tokens=[0, 5119, 512])
        self.assertEqual(data[:8], b'T3DV\x02\x00\x00\x00')
        restored = loads(data)
        for name in ('coords', 'dual_vertices', 'intersected_flag', 'pbr', 'split_weight', 'skeleton_field', 'skin_embedding'):
            np.testing.assert_allclose(getattr(restored.asset, name), getattr(source, name), atol=.0003)
        np.testing.assert_array_equal(restored.bpt_tokens, [0, 5119, 512])
        self.assertEqual(dumps(restored.asset, bpt_tokens=restored.bpt_tokens), data)

    def test_geometry_only_and_empty(self):
        a = self.asset(pbr=None, split_weight=None, skeleton_field=None, skin_embedding=None)
        self.assertIsNone(loads(dumps(a)).asset.pbr)
        a = VoxelAsset(np.empty((0, 3), int), np.empty((0, 3)), np.empty((0, 3), bool), (4,)*3, [[0]*3, [1]*3])
        self.assertEqual(loads(dumps(a)).asset.coords.shape, (0, 3))

    def test_corruption_rejected(self):
        data = dumps(self.asset())
        self.assertEqual(struct.unpack_from('<H', data, 110)[0], 3)
        changed_payload = bytearray(data)
        changed_payload[128] ^= 1
        with self.assertRaisesRegex(ValueError, 'integrity'):
            loads(changed_payload)
        # Profile 1 predates the checksum trailer and remains readable.
        legacy = bytearray(data[:-32])
        struct.pack_into('<H', legacy, 110, 1)
        self.assertEqual(len(loads(legacy).asset.coords), len(self.asset().coords))
        for length in (0, 127, 129, len(data)-1):
            with self.assertRaises(ValueError):
                loads(data[:length])
        for offset, fmt, value in ((0, '4s', b'BAD!'), (4, 'I', 99), (56, 'I', 2**32-1),
                                   (74, 'H', 1), (110, 'H', 99), (116, 'H', 1),
                                   (44, 'f', .9)):
            corrupt = bytearray(data)
            struct.pack_into('<'+fmt, corrupt, offset, value)
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                loads(corrupt)
        with self.assertRaises(ValueError):
            loads(data+b'junk')

    def test_tokens_and_precision_rejection(self):
        for tokens in ([-1], [5120], [1.5]):
            with self.assertRaises(ValueError):
                dumps(self.asset(), bpt_tokens=tokens)
        with self.assertRaises(ValueError):
            dumps(self.asset(skeleton_field=np.full((2, 6), 1e10)))

    def test_atomic_save_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'asset.t3d'
            save(path, self.asset())
            self.assertEqual(len(load(path).asset.coords), 2)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ['asset.t3d'])
