import unittest
import numpy as np
from t3d.materials import PBRMaterial, bilinear, closest_surface, sample_pbr
from t3d.schema import VoxelAsset
from t3d.container import dumps, loads


class MaterialTests(unittest.TestCase):
    v = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    f = [[0, 1, 2]]

    def asset(self):
        return VoxelAsset([[0, 0, 0]], [[.25, .25, .5]], np.zeros((1, 3), bool),
                          (1,)*3, [[0]*3, [1]*3])

    def test_closest_point_face_edge_vertex(self):
        ids, bary = closest_surface([[.2, .3, 2], [.8, .8, 0], [-1, -1, 0]], self.v, self.f)
        np.testing.assert_array_equal(ids, 0)
        np.testing.assert_allclose(bary, [[.5, .2, .3], [0, .5, .5], [1, 0, 0]], atol=1e-12)

    def test_factor_channels_and_storage(self):
        a = sample_pbr(self.asset(), self.v, self.f,
                       [PBRMaterial(base_color=(.2, .4, .6, .3), metallic=.7, roughness=.8, alpha_mode='BLEND')], [0])
        np.testing.assert_allclose(a.pbr, [[.2, .4, .6, .7, .8, .3]])
        np.testing.assert_allclose(loads(dumps(a)).asset.pbr, a.pbr, atol=.0003)

    def test_texture_color_space_and_gb_channels(self):
        m = PBRMaterial(base_texture=np.full((1, 1, 4), .5),
                        metallic_roughness_texture=np.array([[[.9, .2, .8]]]),
                        metallic=.5, roughness=.5, alpha_mode='BLEND')
        a = sample_pbr(self.asset(), self.v, self.f, [m], [0], corner_uv=[[[0, 0], [1, 0], [0, 1]]])
        linear = ((.5+.055)/1.055)**2.4
        np.testing.assert_allclose(a.pbr, [[linear]*3 + [.4, .1, .5]])

    def test_sampling_wrap_and_linear_filtering(self):
        texture = np.array([[[0., 0, 0], [1., 1, 1]]])
        np.testing.assert_allclose(bilinear(texture, [[.25, .5], [1.25, .5]]), 0)
        np.testing.assert_allclose(bilinear(texture, [[0, .5]], wrap='repeat'), .5)
        np.testing.assert_allclose(bilinear(texture, [[0, .5]], wrap='clamp'), 0)
        # sRGB is decoded before interpolation.
        np.testing.assert_allclose(bilinear(texture, [[.5, .5]], srgb=True), .5)

    def test_alpha_modes_and_validation(self):
        for mode, expected in [('OPAQUE', 1), ('MASK', 0), ('BLEND', .4)]:
            a = sample_pbr(self.asset(), self.v, self.f,
                           [PBRMaterial(base_color=(1, 1, 1, .4), alpha_mode=mode)], [0])
            self.assertAlmostEqual(a.pbr[0, 5], expected)
        with self.assertRaises(ValueError):
            PBRMaterial(metallic=2)
        with self.assertRaises(ValueError):
            sample_pbr(self.asset(), self.v, self.f, [PBRMaterial(base_texture=np.ones((1, 1, 3)))], [0])

    def test_face_material_assignment(self):
        v = np.concatenate([self.v, np.array(self.v)+[0, 0, 1]])
        ids, _ = closest_surface([[.1, .1, .9]], v, [[0, 1, 2], [3, 4, 5]])
        np.testing.assert_array_equal(ids, [1])
