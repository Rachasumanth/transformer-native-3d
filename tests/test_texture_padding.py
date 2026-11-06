import importlib.util
import unittest
import numpy as np
from t3d.texture_padding import pad_pbr


class TexturePaddingTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('cv2'),'OpenCV required for atlas gap filling')
    def test_constant_six_channel_atlas_and_valid_texels(self):
        constants=np.array([.25,.1,.8,.3,.65,.6],dtype=np.float32)
        valid=np.zeros((32,32),bool);valid[8:24,8:24]=True
        source=np.zeros((32,32,6),dtype=np.float32);source[valid]=constants
        original=source.copy()
        padded=pad_pbr(source,valid)
        np.testing.assert_array_equal(source,original)
        np.testing.assert_array_equal(padded[valid],source[valid])
        # Atlas padding will become 8-bit maps; allow one byte error in gaps.
        np.testing.assert_allclose(padded,np.broadcast_to(constants,padded.shape),atol=1/255)
        self.assertTrue(np.isfinite(padded).all())

    def test_full_atlas_and_rejected_empty_support(self):
        image=np.random.default_rng(41).random((4,4,6),dtype=np.float32)
        np.testing.assert_array_equal(pad_pbr(image,np.ones((4,4),bool)),image)
        with self.assertRaisesRegex(ValueError,'covered'):pad_pbr(image,np.zeros((4,4),bool))
        with self.assertRaisesRegex(ValueError,'boolean'):pad_pbr(image,np.ones((4,4),int))
