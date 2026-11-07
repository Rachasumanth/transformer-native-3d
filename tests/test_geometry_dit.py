import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from t3d.flow import SparseFlowDiT, flow_pair, flow_loss, sample_flow
from t3d.sc_vae import SparseCompressionVAE
from t3d.extraction import voxels_to_mesh
from t3d.schema import VoxelAsset


class GeometryDiTTests(unittest.TestCase):
    def test_geometry_flow_forward_and_loss(self):
        torch.manual_seed(42)
        device = 'cpu'
        # 16-channel shape latents matching SC-VAE latent space
        model = SparseFlowDiT(
            channels=16,
            condition_channels=32,
            width=96,
            heads=4,
            depth=2,
            patch_size=64
        ).to(device)

        n_tokens = 8
        coords = torch.randint(0, 2, (n_tokens, 3), device=device)
        clean_latents = torch.randn(n_tokens, 16, device=device)
        condition_tokens = torch.randn(1, 32, device=device)

        loss = flow_loss(model, clean_latents, coords, condition_tokens)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        # Backward pass check with unconditional drop enabled to exercise null_prompt as well
        loss_drop = flow_loss(model, clean_latents, coords, condition_tokens, drop_probability=1.0)
        (loss + loss_drop).backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Grad is None for {name}")
                self.assertTrue(torch.isfinite(param.grad).all())

    def test_geometry_flow_sampling_and_decoding(self):
        torch.manual_seed(42)
        device = 'cpu'
        model = SparseFlowDiT(
            channels=16,
            condition_channels=32,
            width=96,
            heads=4,
            depth=2,
            patch_size=64
        ).to(device)
        model.eval()

        n_tokens = 4
        latent_coords = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.long)
        noise = torch.randn(n_tokens, 16, device=device)
        condition_tokens = torch.randn(1, 32, device=device)

        pred_latents = sample_flow(model, noise, latent_coords, condition_tokens, steps=4, guidance=2.0)
        self.assertEqual(pred_latents.shape, (n_tokens, 16))
        self.assertTrue(torch.isfinite(pred_latents).all())

        # Decode into mesh through SC-VAE
        vae = SparseCompressionVAE(
            mode='shape',
            channels=(8, 16, 32, 64, 128),
            latent_channels=16,
            convolution_backend='reference'
        ).to(device).eval()

        with torch.no_grad():
            batched_coords = torch.cat([torch.zeros((n_tokens, 1), dtype=torch.long), latent_coords], dim=1)
            out, raw, _ = vae.decode(batched_coords, pred_latents, threshold=0.0)
            self.assertGreater(len(out), 0)

            coords_np = out[:, 1:].cpu().numpy().astype(int)
            dual_np = raw[:, :3].sigmoid().cpu().numpy()
            flags_np = (raw[:, 3:6].sigmoid() > 0.0).cpu().numpy()
            split_np = raw[:, 6:].cpu().numpy()

        asset = VoxelAsset(coords_np, dual_np, flags_np, (32, 32, 32), [[-0.5]*3, [0.5]*3], split_weight=split_np)
        verts, faces = voxels_to_mesh(asset)
        self.assertGreater(len(verts), 0)
        self.assertGreater(len(faces), 0)

    def test_kaggle_v38_checkpoint_verification(self):
        ckpt_path = Path('artifacts/kaggle-kenney-geometry-dit-v38/geometry-dit-600.pt')
        if not ckpt_path.exists():
            self.skipTest('Kaggle v38 checkpoint not found locally')

        digest = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        self.assertEqual(digest, '9af1686720e06b7a1137c9494cbb783f0721ebd780ae10a0c6fbe77f94d7ed78')

        data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.assertIn('model', data)
        self.assertIn('text_projection', data)
        self.assertEqual(data['latent_channels'], 16)
        self.assertIn('text_model', data['models'])
        self.assertEqual(data['models']['text_model'], 'openai/clip-vit-base-patch32')

        # Load weights into SparseFlowDiT architecture
        model = SparseFlowDiT(channels=16, condition_channels=32, width=96, heads=4, depth=4, patch_size=64)
        model.load_state_dict(data['model'])
        text_proj = torch.nn.Linear(512, 32)
        text_proj.load_state_dict(data['text_projection'])

        # Verify forward with loaded weights
        coords = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.long)
        x = torch.randn(2, 16)
        raw_clip = torch.randn(1, 512)
        cond = text_proj(raw_clip)
        out = model(x, coords, torch.tensor(0.5), cond)
        self.assertEqual(out.shape, (2, 16))
        self.assertTrue(torch.isfinite(out).all())
