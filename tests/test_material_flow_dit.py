"""Unit tests for Stage 3 Shape-Conditioned Material Flow DiT and generation integration."""
import unittest
import numpy as np
import torch
from t3d.flow import SparseFlowDiT, flow_loss, sample_flow
from t3d.generate import generate_asset_from_latents, generate_latents_from_prompt
from t3d.sc_vae import SparseCompressionVAE
from t3d.extraction import voxels_to_mesh


class MockPromptConditioner:
    def eval(self):
        return self

    def encode_text(self, prompt: str):
        return torch.randn(1, 32)


class MockMaterialFlowDiT(torch.nn.Module):
    def __init__(self, channels=16, shape_channels=16):
        super().__init__()
        self.channels = channels
        self.shape_channels = shape_channels
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def eval(self):
        return self

    def forward(self, x, coords, time, condition=None, orders=None, shape=None):
        if self.shape_channels > 0 and (shape is None or shape.shape != (len(x), self.shape_channels)):
            raise ValueError('Shape conditioning mismatch')
        return torch.zeros_like(x)


class MaterialFlowDiTTests(unittest.TestCase):
    def test_sparse_flow_dit_shape_conditioning(self):
        model = SparseFlowDiT(
            channels=16,
            condition_channels=32,
            width=96,
            heads=4,
            depth=2,
            patch_size=64,
            shape_channels=16
        )
        coords = torch.randint(0, 16, (12, 3))
        shape_latents = torch.randn(12, 16)
        mat_latents = torch.randn(12, 16)
        tokens = torch.randn(4, 32)

        # Loss calculation
        loss = flow_loss(model, mat_latents, coords, tokens, shape=shape_latents)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.output.weight.grad)

        # Flow sampling
        noise = torch.randn(12, 16)
        sampled = sample_flow(model, noise, coords, tokens, shape=shape_latents, steps=3)
        self.assertEqual(sampled.shape, (12, 16))
        self.assertTrue(torch.isfinite(sampled).all())

    def test_shape_conditioning_dimension_validation(self):
        model = SparseFlowDiT(
            channels=16,
            condition_channels=32,
            width=96,
            heads=4,
            depth=2,
            patch_size=64,
            shape_channels=16
        )
        coords = torch.randint(0, 16, (8, 3))
        shape_wrong = torch.randn(8, 8)  # Wrong dim: 8 instead of 16
        mat_latents = torch.randn(8, 16)
        tokens = torch.randn(4, 32)

        with self.assertRaises(ValueError):
            flow_loss(model, mat_latents, coords, tokens, shape=shape_wrong)

    def test_generate_latents_with_material_flow(self):
        device = 'cpu'
        occ_model = MockMaterialFlowDiT(channels=1, shape_channels=0)
        geom_model = MockMaterialFlowDiT(channels=16, shape_channels=0)
        mat_flow_model = MockMaterialFlowDiT(channels=16, shape_channels=16)
        conditioner = MockPromptConditioner()

        # Test backward-compatible 2-tuple return when return_material=False
        coords, latents = generate_latents_from_prompt(
            prompt="a test asset",
            occupancy_model=occ_model,
            geometry_model=geom_model,
            conditioner=conditioner,
            material_model=mat_flow_model,
            coarse_grid_size=4,
            occupancy_steps=2,
            geometry_steps=2,
            occupancy_threshold=-10.0,
            device=device,
            return_material=False
        )
        self.assertTrue(len(coords) > 0)
        self.assertEqual(latents.shape, (len(coords), 16))

        # Test 3-tuple return when return_material=True
        coords, shape_lat, mat_lat = generate_latents_from_prompt(
            prompt="a test asset",
            occupancy_model=occ_model,
            geometry_model=geom_model,
            conditioner=conditioner,
            material_model=mat_flow_model,
            coarse_grid_size=4,
            occupancy_steps=2,
            geometry_steps=2,
            occupancy_threshold=-10.0,
            device=device,
            return_material=True
        )
        self.assertTrue(len(coords) > 0)
        self.assertEqual(shape_lat.shape, (len(coords), 16))
        self.assertEqual(mat_lat.shape, (len(coords), 16))

    def test_generate_asset_from_learned_material_latents(self):
        shape_vae = SparseCompressionVAE(
            mode='shape',
            channels=(4, 8, 16, 32, 64),
            latent_channels=4,
            convolution_backend='reference'
        ).eval()

        mat_vae = SparseCompressionVAE(
            mode='material',
            channels=(4, 8, 16, 32, 64),
            latent_channels=4,
            convolution_backend='reference'
        ).eval()

        active_coords = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.long)
        shape_latents = torch.randn(2, 4)
        learned_mat_latents = torch.randn(2, 4)

        asset = generate_asset_from_latents(
            active_coords, shape_latents, shape_vae,
            grid_size=(32, 32, 32), threshold=0.0,
            material_latents=learned_mat_latents,
            material_vae_model=mat_vae
        )
        self.assertIsNotNone(asset.pbr)
        self.assertEqual(asset.pbr.shape, (len(asset.coords), 6))
        self.assertTrue(np.all((asset.pbr >= 0.0) & (asset.pbr <= 1.0)))

        # Extract mesh and package GLB
        v, f = voxels_to_mesh(asset)
        from t3d.glb import pack_static_glb
        glb = pack_static_glb(v, f, pbr=asset.pbr)
        self.assertTrue(len(glb) > 0)
        self.assertEqual(glb[:4], b'glTF')

    def test_checkpoint_sha256_and_report_metrics(self):
        import hashlib
        import json
        from pathlib import Path
        ckpt_300 = Path('artifacts/kaggle-material-flow-dit-v1/material-flow-dit-300.pt')
        ckpt_600 = Path('artifacts/kaggle-material-flow-dit-v1/material-flow-dit-600.pt')
        report_path = Path('artifacts/kaggle-material-flow-dit-v1/material-flow-dit-report.json')

        if not ckpt_300.exists() or not report_path.exists():
            self.skipTest("Material flow checkpoints or report not found")

        digest_300 = hashlib.sha256(ckpt_300.read_bytes()).hexdigest()
        self.assertEqual(digest_300, 'b9f6fb93a7d8832ff9078c1abb1846d21edfc98244458f6135b87644c55c06a4')

        digest_600 = hashlib.sha256(ckpt_600.read_bytes()).hexdigest()
        self.assertEqual(digest_600, '39f86d0cc6031ac22f57a96ae05ba326c302029c2b1042bfa926b55baf8db6d9')

        report = json.loads(report_path.read_text())
        self.assertEqual(report['model'], 'SparseFlowDiT-Stage3-Material')
        self.assertEqual(report['train_assets'], 18)
        self.assertEqual(report['validation_assets'], 8)
        self.assertLess(report['final_val_pbr_mae']['mean'], 0.25)

    def test_checkpoint_weights_load_and_sample(self):
        from pathlib import Path
        ckpt_300 = Path('artifacts/kaggle-material-flow-dit-v1/material-flow-dit-300.pt')
        if not ckpt_300.exists():
            self.skipTest("Checkpoint not found")

        data = torch.load(ckpt_300, map_location='cpu', weights_only=False)
        model = SparseFlowDiT(
            channels=int(data['config']['channels']),
            condition_channels=int(data['config']['condition_channels']),
            width=data['config']['width'],
            heads=data['config']['heads'],
            depth=data['config']['depth'],
            patch_size=data['config']['patch_size'],
            shape_channels=int(data['config']['shape_channels'])
        )
        model.load_state_dict(data['model'])
        model.eval()

        coords = torch.randint(0, 16, (4, 3))
        shape_lat = torch.randn(4, 16)
        noise = torch.randn(4, 16)
        tokens = torch.randn(2, 32)
        out = sample_flow(model, noise, coords, tokens, shape=shape_lat, steps=3)
        self.assertEqual(out.shape, (4, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_asset_id_to_prompt_boombox_mapping(self):
        from tools.train_multiasset_material_flow_dit import asset_id_to_prompt
        self.assertEqual(
            asset_id_to_prompt('boombox-rot0'),
            'a retro 1980s portable boombox cassette player stereo'
        )
        self.assertEqual(
            asset_id_to_prompt('boombox-rot180'),
            'a retro 1980s portable boombox cassette player stereo'
        )
        self.assertEqual(
            asset_id_to_prompt('box-rot0'),
            'a brown wooden storage crate shipping container'
        )
        self.assertEqual(
            asset_id_to_prompt('lantern-rot0'),
            'an antique iron metal lantern lamp with glass panels'
        )


if __name__ == '__main__':
    unittest.main()
