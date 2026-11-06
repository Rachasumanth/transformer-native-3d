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
