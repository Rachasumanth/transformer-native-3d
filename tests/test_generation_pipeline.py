import unittest
import numpy as np
import torch
from t3d.generate import generate_asset_from_latents, generate_latents_from_prompt
from t3d.sc_vae import SparseCompressionVAE
from t3d.extraction import voxels_to_mesh


class MockConditioner:
    def eval(self):
        return self

    def encode_text(self, prompt: str):
        # Return mock (1, 32) condition tokens
        return torch.randn(1, 32)


class MockDiT(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, coords, time, condition=None, orders=None, shape=None):
        # Velocity targeting: predict zero
        return torch.zeros_like(x)


class GenerationPipelineTests(unittest.TestCase):
    def test_generate_latents_from_prompt(self):
        device = 'cpu'
        occ_model = MockDiT(channels=1)
        geom_model = MockDiT(channels=4)
        conditioner = MockConditioner()

        coords, latents = generate_latents_from_prompt(
            prompt="a test spaceship",
            occupancy_model=occ_model,
            geometry_model=geom_model,
            conditioner=conditioner,
            coarse_grid_size=4,
            occupancy_steps=2,
            geometry_steps=2,
            occupancy_threshold=-10.0,  # All cells active for test
            device=device,
            seed=42
        )
        self.assertTrue(len(coords) > 0)
        self.assertEqual(coords.shape[1], 3)
        self.assertEqual(latents.shape, (len(coords), 4))

    def test_generate_asset_from_latents(self):
        vae = SparseCompressionVAE(
            mode='shape',
            channels=(4, 8, 16, 32, 64),
            latent_channels=4,
            convolution_backend='reference'
        )
        vae.eval()

        # Create 2 mock latent coordinates
        active_coords = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.long)
        shape_latents = torch.randn(2, 4)

        asset = generate_asset_from_latents(
            active_coords, shape_latents, vae,
            grid_size=(32, 32, 32), threshold=0.0  # Threshold=0.0 keeps all decoded octants active
        )
        self.assertTrue(len(asset.coords) > 0)
        self.assertEqual(asset.dual_vertices.shape, (len(asset.coords), 3))
        self.assertEqual(asset.intersected_flag.shape, (len(asset.coords), 3))

        # Test mesh extraction
        v, f = voxels_to_mesh(asset)
        self.assertTrue(len(v) > 0)
        self.assertTrue(len(f) > 0)

        # Test GLB packaging without materials
        from t3d.glb import pack_static_glb
        glb = pack_static_glb(v, f)
        self.assertTrue(len(glb) > 0)
        self.assertEqual(glb[:4], b'glTF')

        # Test paired shape + material decoding
        mat_vae = SparseCompressionVAE(
            mode='material',
            channels=(4, 8, 16, 32, 64),
            latent_channels=4,
            convolution_backend='reference'
        )
        mat_vae.eval()
        mat_asset = generate_asset_from_latents(
            active_coords, shape_latents, vae,
            grid_size=(32, 32, 32), threshold=0.0,
            material_vae_model=mat_vae
        )
        self.assertIsNotNone(mat_asset.pbr)
        self.assertEqual(mat_asset.pbr.shape, (len(mat_asset.coords), 6))
        self.assertTrue(np.all((mat_asset.pbr >= 0.0) & (mat_asset.pbr <= 1.0)))

        # Test textured GLB packaging with PBR
        mv, mf = voxels_to_mesh(mat_asset)
        pbr_glb = pack_static_glb(mv, mf, pbr=mat_asset.pbr)
        self.assertTrue(len(pbr_glb) > 0)
        self.assertEqual(pbr_glb[:4], b'glTF')

    def test_generate_64_resolution_pipeline(self):
        device = 'cpu'
        occ_model = MockDiT(channels=1)
        geom_model = MockDiT(channels=4)
        conditioner = MockConditioner()

        coords, latents = generate_latents_from_prompt(
            prompt="a 64-res test vessel",
            occupancy_model=occ_model,
            geometry_model=geom_model,
            conditioner=conditioner,
            coarse_grid_size=4,
            latent_grid_size=4,
            occupancy_steps=2,
            geometry_steps=2,
            occupancy_threshold=-10.0,
            device=device,
            seed=42
        )
        self.assertEqual(len(coords), 64)
        self.assertTrue((coords >= 0).all() and (coords < 4).all())
        self.assertEqual(latents.shape, (64, 4))

    def test_generate_rigged_pipeline(self):
        vae = SparseCompressionVAE(
            mode='shape',
            channels=(4, 8, 16, 32, 64),
            latent_channels=4,
            convolution_backend='reference'
        )
        vae.eval()
        active_coords = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.long)
        shape_latents = torch.randn(2, 4)

        rig_model = MockDiT(channels=10)
        tokens = torch.randn(1, 32)
        rig_asset = generate_asset_from_latents(
            active_coords, shape_latents, vae,
            grid_size=(32, 32, 32), threshold=0.0,
            rig_model=rig_model, tokens=tokens, rig_steps=2
        )
        self.assertIsNotNone(rig_asset.skeleton_field)
        self.assertIsNotNone(rig_asset.skin_embedding)
        self.assertEqual(rig_asset.skeleton_field.shape, (len(rig_asset.coords), 6))
        self.assertEqual(rig_asset.skin_embedding.shape, (len(rig_asset.coords), 4))





