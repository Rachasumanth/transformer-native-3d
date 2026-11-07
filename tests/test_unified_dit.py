"""Unit test suite for Unified Dense DiT Architecture.

Tests cover:
- Forward pass with joint state x_t in R^(B x N x (C+1)), timestep t in R^B,
  and condition c_raw in R^(B x N x (C+1)).
- Output velocity prediction v in R^(B x N x (C+1)).
- 3D RoPE spatial invariance and gradient backpropagation.
- Classifier-Free Guidance (CFG), Interval Guidance, and unconditional token masking.
- Flow matching loss and sampling integration.
"""
import importlib.util
import math
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from t3d.models.unified_dit import (
        UnifiedDiT,
        RoPE3D,
        SelfAttention3D,
        DiTBlock,
        flow_pair,
        flow_loss,
        sample_flow,
    )


@unittest.skipUnless(HAS_TORCH, "PyTorch required for UnifiedDiT tests")
class UnifiedDiTTests(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)

    def test_instantiation_defaults_and_configs(self):
        """Verify model initialization with defaults and custom configurations."""
        # Default initialization: dim=288, depth=4, heads=6, latent_channels=16, coarse_res=4
        model = UnifiedDiT().to(self.device)
        self.assertEqual(model.dim, 288)
        self.assertEqual(model.depth, 4)
        self.assertEqual(model.heads, 6)
        self.assertEqual(model.latent_channels, 16)
        self.assertEqual(model.in_channels, 17)  # 1 occupancy logit + 16 latent channels
        self.assertEqual(model.coarse_res, 4)
        self.assertEqual(len(model.blocks), 4)

        # Custom configuration: dim=192, depth=2, heads=4, latent_channels=8, coarse_res=2
        custom = UnifiedDiT(dim=192, depth=2, heads=4, latent_channels=8, coarse_res=2).to(self.device)
        self.assertEqual(custom.dim, 192)
        self.assertEqual(custom.depth, 2)
        self.assertEqual(custom.heads, 4)
        self.assertEqual(custom.latent_channels, 8)
        self.assertEqual(custom.in_channels, 9)
        self.assertEqual(custom.coarse_res, 2)

        # Invalid configurations must raise ValueError
        with self.assertRaises(ValueError):
            UnifiedDiT(depth=0)
        with self.assertRaises(ValueError):
            UnifiedDiT(dim=100, heads=3)  # 100 not divisible by 3
        with self.assertRaises(ValueError):
            UnifiedDiT(dim=64, heads=4)  # head_dim=16 not divisible by 6
        with self.assertRaises(ValueError):
            UnifiedDiT(latent_channels=0)
        with self.assertRaises(ValueError):
            UnifiedDiT(coarse_res=0)

    def test_forward_joint_state_and_output_velocity(self):
        """Verify forward pass with joint state x_t, timestep t, condition c_raw -> velocity v."""
        B, coarse_res, C = 2, 4, 16
        N = coarse_res ** 3  # 64 tokens
        in_c = C + 1  # 17 channels

        model = UnifiedDiT(dim=96, depth=2, heads=4, latent_channels=C, coarse_res=coarse_res).to(self.device)
        model.eval()

        x_t = torch.randn(B, N, in_c, device=self.device)
        t = torch.tensor([0.25, 0.75], device=self.device)
        c_raw = torch.randn(B, N, in_c, device=self.device)

        # Forward with default grid coordinates
        v = model(x_t, t, c_raw=c_raw)
        self.assertEqual(v.shape, (B, N, in_c))
        self.assertTrue(torch.isfinite(v).all())

        # Forward with explicit custom coordinates
        coords = torch.randn(B, N, 3, device=self.device)
        v_coords = model(x_t, t, c_raw=c_raw, coords=coords)
        self.assertEqual(v_coords.shape, (B, N, in_c))
        self.assertTrue(torch.isfinite(v_coords).all())

        # Unbatched forward pass: (N, in_c), scalar t, (N, in_c) -> (N, in_c)
        v_unbatched = model(x_t[0], 0.5, c_raw=c_raw[0])
        self.assertEqual(v_unbatched.shape, (N, in_c))
        self.assertTrue(torch.isfinite(v_unbatched).all())

    def test_split_and_join_state(self):
        """Verify joint state splitting into occupancy & latent and rejoining."""
        B, N, C = 2, 64, 16
        state = torch.randn(B, N, C + 1)
        occ, latent = UnifiedDiT.split_state(state)
        self.assertEqual(occ.shape, (B, N, 1))
        self.assertEqual(latent.shape, (B, N, C))
        reconstructed = UnifiedDiT.join_state(occ, latent)
        torch.testing.assert_close(reconstructed, state)

    def test_rope_rotation_properties(self):
        """Verify 3D RoPE preserves norm, maintains orthonormal frames, and supports batching."""
        heads, head_dim = 4, 24
        rope = RoPE3D(heads=heads, head_dim=head_dim).to(self.device)

        # Orthonormal rotation frames R @ R^T == I
        rot = rope.rotation()
        self.assertEqual(rot.shape, (heads, 3, 3))
        eye = torch.eye(3, device=self.device).expand(heads, 3, 3)
        torch.testing.assert_close(torch.matmul(rot, rot.transpose(-1, -2)), eye)

        # Preserves vector L2 norm
        B, N = 3, 32
        v = torch.randn(B, heads, N, head_dim, device=self.device)
        coords = torch.randn(N, 3, device=self.device)
        v_rot = rope(v, coords)
        self.assertEqual(v_rot.shape, v.shape)
        torch.testing.assert_close(v_rot.norm(dim=-1), v.norm(dim=-1))

        # Batched coordinates support (B, N, 3)
        coords_batched = torch.randn(B, N, 3, device=self.device)
        v_rot_batched = rope(v, coords_batched)
        self.assertEqual(v_rot_batched.shape, v.shape)
        torch.testing.assert_close(v_rot_batched.norm(dim=-1), v.norm(dim=-1))

    def test_rope_spatial_invariance(self):
        """Verify 3D RoPE spatial translation invariance: Attention(p + Delta) == Attention(p)."""
        heads, head_dim = 4, 24
        rope = RoPE3D(heads=heads, head_dim=head_dim).to(self.device)

        B, N = 2, 32
        q = torch.randn(B, heads, N, head_dim, device=self.device)
        k = torch.randn(B, heads, N, head_dim, device=self.device)
        coords = torch.randn(N, 3, device=self.device)
        delta = torch.tensor([12.5, -34.8, 56.1], device=self.device)

        # Unshifted attention logits
        q_rot = rope(q, coords)
        k_rot = rope(k, coords)
        attn_orig = torch.matmul(q_rot, k_rot.transpose(-1, -2))

        # Translated attention logits
        q_rot_shifted = rope(q, coords + delta)
        k_rot_shifted = rope(k, coords + delta)
        attn_shifted = torch.matmul(q_rot_shifted, k_rot_shifted.transpose(-1, -2))

        # Translation invariance must hold within float precision
        torch.testing.assert_close(attn_orig, attn_shifted, rtol=1e-4, atol=1e-4)

        # Float64 exact translation invariance test
        rope_64 = RoPE3D(heads=heads, head_dim=head_dim).double().to(self.device)
        q_64 = q.double()
        k_64 = k.double()
        coords_64 = coords.double()
        delta_64 = delta.double()

        q_r64 = rope_64(q_64, coords_64)
        k_r64 = rope_64(k_64, coords_64)
        attn_64 = torch.matmul(q_r64, k_r64.transpose(-1, -2))

        q_r64_s = rope_64(q_64, coords_64 + delta_64)
        k_r64_s = rope_64(k_64, coords_64 + delta_64)
        attn_64_s = torch.matmul(q_r64_s, k_r64_s.transpose(-1, -2))

        torch.testing.assert_close(attn_64, attn_64_s, rtol=1e-10, atol=1e-10)

        # End-to-end SelfAttention3D invariance
        attn_layer = SelfAttention3D(dim=96, heads=4).to(self.device)
        x = torch.randn(B, N, 96, device=self.device)
        out1 = attn_layer(x, coords)
        out2 = attn_layer(x, coords + delta)
        torch.testing.assert_close(out1, out2, rtol=1e-4, atol=1e-4)

    def test_gradient_backpropagation(self):
        """Verify gradient flow through all UnifiedDiT layers, AdaLN-Zero, and 3D RoPE axes."""
        model = UnifiedDiT(dim=96, depth=2, heads=4, latent_channels=16, coarse_res=4).to(self.device)
        B, N, C = 2, 64, 17

        # 1. Direct backprop through RoPE3D
        rope = RoPE3D(heads=4, head_dim=24).to(self.device)
        q = torch.randn(B, 4, N, 24, device=self.device, requires_grad=True)
        coords = torch.randn(N, 3, device=self.device)
        rope_out = rope(q, coords)
        loss_rope = (rope_out ** 2).sum()
        loss_rope.backward()
        self.assertIsNotNone(q.grad)
        self.assertTrue((q.grad != 0).any())
        self.assertIsNotNone(rope.axes.grad)
        self.assertTrue((rope.axes.grad != 0).any())

        # 2. End-to-end multi-step optimization through UnifiedDiT with flow_loss
        clean = torch.randn(B, N, C, device=self.device)
        c_raw = torch.randn(B, N, C, device=self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for step in range(3):
            optimizer.zero_grad()
            loss = flow_loss(model, clean, c_raw=c_raw)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()

        # All key parameters must receive non-zero finite gradients
        self.assertIsNotNone(model.x_embed.weight.grad)
        self.assertTrue(torch.isfinite(model.x_embed.weight.grad).all())
        self.assertTrue((model.x_embed.weight.grad.abs().sum() > 0).item())

        self.assertIsNotNone(model.final_layer.linear.weight.grad)
        self.assertTrue(torch.isfinite(model.final_layer.linear.weight.grad).all())
        self.assertTrue((model.final_layer.linear.weight.grad.abs().sum() > 0).item())

        for i, block in enumerate(model.blocks):
            self.assertIsNotNone(block.attn.rope.axes.grad)
            self.assertTrue(torch.isfinite(block.attn.rope.axes.grad).all())
            self.assertTrue((block.attn.rope.axes.grad.abs().sum() > 0).item(), f"Block {i} RoPE axes grad was zero")

    def test_cfg_and_unconditional_token_masking(self):
        """Verify CFG, Interval Guidance, and unconditional token masking mechanics."""
        model = UnifiedDiT(dim=96, depth=2, heads=4, latent_channels=16, coarse_res=4).to(self.device)
        # Give non-zero weights for thorough testing
        for p in model.parameters():
            if p.ndim > 1:
                torch.nn.init.normal_(p, std=0.02)
        model.eval()

        B, N, C = 2, 64, 17
        x = torch.randn(B, N, C, device=self.device)
        t = torch.tensor([0.5, 0.5], device=self.device)
        c_raw = torch.randn(B, N, C, device=self.device)

        # 1. c_raw=None and drop_condition=True must yield identical outputs
        v_none = model(x, t, None)
        v_drop = model(x, t, c_raw, drop_condition=True)
        torch.testing.assert_close(v_none, v_drop)

        # 2. CFG scale = 1.0 matches conditional forward
        v_cond = model(x, t, c_raw)
        v_cfg1 = model.forward_with_cfg(x, t, c_raw, cfg_scale=1.0)
        torch.testing.assert_close(v_cond, v_cfg1)

        # 3. CFG scale = 3.0 matches formula: v_uncond + 3.0 * (v_cond - v_uncond)
        v_uncond = model(x, t, None)
        expected_cfg3 = v_uncond + 3.0 * (v_cond - v_uncond)
        v_cfg3 = model.forward_with_cfg(x, t, c_raw, cfg_scale=3.0)
        torch.testing.assert_close(expected_cfg3, v_cfg3)

        # 4. Interval guidance outside interval [0.2, 0.8] falls back to conditional
        t_early = torch.tensor([0.1, 0.1], device=self.device)
        v_early_cond = model(x, t_early, c_raw)
        v_early_cfg = model.forward_with_cfg(x, t_early, c_raw, cfg_scale=3.0, cfg_interval=(0.2, 0.8))
        torch.testing.assert_close(v_early_cond, v_early_cfg)

        t_late = torch.tensor([0.9, 0.9], device=self.device)
        v_late_cond = model(x, t_late, c_raw)
        v_late_cfg = model.forward_with_cfg(x, t_late, c_raw, cfg_scale=3.0, cfg_interval=(0.2, 0.8))
        torch.testing.assert_close(v_late_cond, v_late_cfg)

        # 5. Token-level unconditional masking
        # Mask out first half of tokens in sample 0
        cond_mask = torch.ones(B, N, 1, dtype=torch.bool, device=self.device)
        cond_mask[0, :N // 2] = False
        v_masked = model(x, t, c_raw, cond_mask=cond_mask)

        # Construct expected c_raw manually
        c_raw_expected = c_raw.clone()
        c_raw_expected[0, :N // 2] = model.null_condition.expand(1, N // 2, C)
        v_expected_mask = model(x, t, c_raw_expected)
        torch.testing.assert_close(v_masked, v_expected_mask)

        # 6. Training condition dropout with cond_drop_prob=1.0
        model.train()
        v_train_drop = model(x, t, c_raw, cond_drop_prob=1.0)
        torch.testing.assert_close(v_train_drop, v_drop)

    def test_flow_matching_pipeline(self):
        """Verify flow_pair, flow_loss, and sample_flow numerical integration."""
        B, N, C = 2, 64, 17
        clean = torch.randn(B, N, C, device=self.device)
        noise = torch.randn(B, N, C, device=self.device)
        t = torch.tensor([0.25, 0.5], device=self.device)

        # flow_pair linear interpolation
        x_t, v_t = flow_pair(clean, noise, t)
        self.assertEqual(x_t.shape, (B, N, C))
        self.assertEqual(v_t.shape, (B, N, C))
        torch.testing.assert_close(v_t, clean - noise)

        # flow_loss computation
        model = UnifiedDiT(dim=96, depth=2, heads=4, latent_channels=16, coarse_res=4).to(self.device)
        c_raw = torch.randn(B, N, C, device=self.device)
        loss = flow_loss(model, clean, c_raw=c_raw)
        self.assertTrue(torch.isfinite(loss))

        # sample_flow generation
        samples = sample_flow(model, c_raw=c_raw, batch_size=B, steps=4, cfg_scale=2.0, device=self.device)
        self.assertEqual(samples.shape, (B, N, C))
        self.assertTrue(torch.isfinite(samples).all())


if __name__ == "__main__":
    unittest.main()
