"""Unit tests for Stage 2 Shape-Conditioned Rig Flow DiT and rigged asset generation."""
import unittest
import numpy as np
import torch

from t3d.articulation import Skeleton, infer_skeleton_from_field
from t3d.flow import SparseFlowDiT, flow_loss, sample_flow
from t3d.generate import generate_asset_from_latents
from t3d.glb import pack_static_glb, attach_skin
from t3d.import_rig import load_rig_glb
from tools.train_multiasset_rig_flow_dit import compute_j2j_chamfer, check_bone_containment


class MockVAE:
    def __init__(self, channels=16):
        self.channels = channels

    def eval(self):
        pass

    def decode(self, batched_coords, shape_latents, threshold=0.5):
        # Return synthetic surface voxels
        n = len(batched_coords)
        coords = torch.cat([batched_coords[:, :1], batched_coords[:, 1:] * 8], dim=1)
        raw = torch.zeros((n, 7), device=batched_coords.device, dtype=torch.float32)
        raw[:, :3] = 0.5  # dual vertices
        raw[:, 3:6] = 2.0  # intersected flags (>0 -> sigmoid > 0.5)
        raw[:, 6:] = 0.0  # split weight
        logits = [torch.zeros((n, 1), device=batched_coords.device)]
        return coords, raw, logits


class TestRigFlowDiT(unittest.TestCase):
    def test_rig_flow_dit_forward_and_loss(self):
        device = 'cpu'
        model = SparseFlowDiT(
            channels=10, condition_channels=32, shape_channels=7,
            width=96, heads=4, depth=2, patch_size=64
        ).to(device)

        n = 48
        clean_rig = torch.randn((n, 10), device=device)
        coords = torch.randint(0, 32, (n, 3), device=device)
        shape_feat = torch.randn((n, 7), device=device)
        tokens = torch.randn((5, 32), device=device)

        loss = flow_loss(model, clean_rig, coords, tokens, shape=shape_feat)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        # Verify gradient flow into trainable block parameters
        has_grad = False
        for name, p in model.named_parameters():
            if 'blocks' in name:
                self.assertIsNotNone(p.grad)
                has_grad = True
        self.assertTrue(has_grad)

    def test_rig_flow_dit_sampling(self):
        device = 'cpu'
        model = SparseFlowDiT(
            channels=10, condition_channels=32, shape_channels=7,
            width=96, heads=4, depth=2, patch_size=64
        ).to(device)

        n = 32
        coords = torch.randint(0, 32, (n, 3), device=device)
        shape_feat = torch.randn((n, 7), device=device)
        tokens = torch.randn((4, 32), device=device)

        noise = torch.randn((n, 10), device=device)
        sampled = sample_flow(model, noise, coords, tokens, shape=shape_feat, steps=5, guidance=2.0)
        self.assertEqual(sampled.shape, (n, 10))
        self.assertTrue(torch.isfinite(sampled).all())

    def test_generate_asset_with_learned_rig(self):
        device = 'cpu'
        vae = MockVAE()
        rig_model = SparseFlowDiT(
            channels=10, condition_channels=32, shape_channels=7,
            width=96, heads=4, depth=2, patch_size=64
        ).to(device)

        active_coords = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]], device=device, dtype=torch.int64)
        shape_latents = torch.randn((len(active_coords), 16), device=device)
        tokens = torch.randn((3, 32), device=device)

        asset = generate_asset_from_latents(
            active_coords, shape_latents, vae,
            rig_model=rig_model, tokens=tokens, rig_steps=3, guidance=1.0
        )

        self.assertIsNotNone(asset.skeleton_field)
        self.assertIsNotNone(asset.skin_embedding)
        self.assertEqual(asset.skeleton_field.shape, (len(asset.coords), 6))
        self.assertEqual(asset.skin_embedding.shape, (len(asset.coords), 4))
        self.assertTrue(np.isfinite(asset.skeleton_field).all())
        self.assertTrue(np.isfinite(asset.skin_embedding).all())

    def test_compute_j2j_chamfer(self):
        # Coincident joints
        joints_a = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
        joints_b = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
        self.assertAlmostEqual(compute_j2j_chamfer(joints_a, joints_b), 0.0)

        # Offset joints by known delta
        joints_c = joints_a + 0.01
        self.assertAlmostEqual(compute_j2j_chamfer(joints_a, joints_c), float(np.linalg.norm([0.01, 0.01, 0.01])), places=4)

    def test_check_bone_containment(self):
        s = Skeleton([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]], [-1, 0])
        res = check_bone_containment(s, aabb=[[-0.5]*3, [0.5]*3])
        self.assertEqual(res['violations'], 0)
        self.assertEqual(res['containment_rate'], 1.0)

        # Joint outside boundary
        s_bad = Skeleton([[0.0, 0.0, 0.0], [1.2, 0.2, 0.2]], [-1, 0])
        res_bad = check_bone_containment(s_bad, aabb=[[-0.5]*3, [0.5]*3])
        self.assertGreater(res_bad['violations'], 0)
        self.assertLess(res_bad['containment_rate'], 1.0)


if __name__ == '__main__':
    unittest.main()
