"""Comprehensive unit tests for Hermite Normal Head & Surface-Grounded PBR Flow Network."""

import unittest
import numpy as np
import torch
import torch.nn.functional as F

from t3d.sc_vae import (
    HermiteNormalHead,
    HermiteHead,
    hermite_normal_loss,
    SparseCompressionVAE,
    reconstruction_loss,
)
from t3d.models.pbr_flow import (
    SurfaceGroundedPBRFlow,
    PBRFlowNetwork,
    PBRFlowOutput,
    debake_luminance,
    pad_texture_dilation,
    dilate_uv_seams,
    format_pbr_textures,
    pbr_flow_pair,
    pbr_flow_loss,
    sample_pbr_flow,
    LUMINANCE_WEIGHTS,
)


class TestPBRHermiteFlow(unittest.TestCase):
    """Test suite verifying Hermite Normal Head, Surface-Grounded PBR Flow, and texture operations."""

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)
        np.random.seed(42)

    # --------------------------------------------------------------------------
    # 1. Hermite Normal Head Tests
    # --------------------------------------------------------------------------
    def test_hermite_head_output_shape_and_unit_normalization(self):
        """Verify hermite_head output shape (B, N, 3) and unit-vector normalization."""
        b, n, in_c = 4, 32, 64
        head = HermiteNormalHead(in_channels=in_c).to(self.device)

        # Test with alias
        self.assertIs(HermiteHead, HermiteNormalHead)

        # Feed non-zero active features
        features = torch.randn(b, n, in_c, device=self.device)
        normals = head(features)

        # Verify output shape is strictly (B, N, 3)
        self.assertEqual(normals.shape, (b, n, 3))

        # Verify unit-vector normalization: ||n_hat||_2 == 1.0 everywhere on active voxels
        lengths = torch.linalg.norm(normals, ord=2, dim=-1)
        expected_lengths = torch.ones_like(lengths)
        torch.testing.assert_close(lengths, expected_lengths, atol=1e-5, rtol=1e-5)

        # Test with 2D tensor shape (N, C)
        feat_2d = torch.randn(16, in_c, device=self.device)
        norm_2d = head(feat_2d)
        self.assertEqual(norm_2d.shape, (16, 3))
        lens_2d = torch.linalg.norm(norm_2d, ord=2, dim=-1)
        torch.testing.assert_close(lens_2d, torch.ones_like(lens_2d), atol=1e-5, rtol=1e-5)

    def test_hermite_head_void_voxels_zero_vector(self):
        """Verify that void voxels produce [0, 0, 0] while active voxels produce unit vectors."""
        b, n, in_c = 2, 10, 32
        head = HermiteNormalHead(in_channels=in_c).to(self.device)

        features = torch.randn(b, n, in_c, device=self.device)
        # Create an active mask where some voxels are void
        active_mask = torch.ones(b, n, dtype=torch.bool, device=self.device)
        active_mask[0, 1] = False
        active_mask[0, 4] = False
        active_mask[1, 7] = False

        normals = head(features, active_mask=active_mask)
        self.assertEqual(normals.shape, (b, n, 3))

        # Check void voxels are strictly zero vectors [0, 0, 0]
        zero_vec = torch.zeros(3, device=self.device)
        torch.testing.assert_close(normals[0, 1], zero_vec)
        torch.testing.assert_close(normals[0, 4], zero_vec)
        torch.testing.assert_close(normals[1, 7], zero_vec)

        # Check active voxels are unit vectors
        active_norms = torch.linalg.norm(normals[active_mask], ord=2, dim=-1)
        torch.testing.assert_close(active_norms, torch.ones_like(active_norms), atol=1e-5, rtol=1e-5)

        # Test automatic void detection when active_mask is None and features are all-zero
        features_zero = features.clone()
        features_zero[0, 3] = 0.0
        normals_auto = head(features_zero, active_mask=None)
        torch.testing.assert_close(normals_auto[0, 3], zero_vec)
        self.assertAlmostEqual(float(torch.linalg.norm(normals_auto[0, 0].detach())), 1.0, places=5)

    def test_hermite_normal_loss_computation(self):
        """Verify normal cosine loss computation: L = 1 - <n_hat, n_gt>."""
        # 1. Identical unit vectors -> loss must be exactly 0.0
        n_gt = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=self.device)
        n_pred = n_gt.clone()
        loss_ident = hermite_normal_loss(n_pred, n_gt)
        self.assertAlmostEqual(float(loss_ident.item()), 0.0, places=5)

        # 2. Opposite unit vectors -> loss must be exactly 2.0
        n_opp = -n_gt
        loss_opp = hermite_normal_loss(n_opp, n_gt)
        self.assertAlmostEqual(float(loss_opp.item()), 2.0, places=5)

        # 3. Orthogonal unit vectors -> loss must be exactly 1.0
        n_orth = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], device=self.device)
        loss_orth = hermite_normal_loss(n_orth, n_gt)
        self.assertAlmostEqual(float(loss_orth.item()), 1.0, places=5)

        # 4. Reduction modes: 'mean', 'sum', 'none'
        loss_none = hermite_normal_loss(n_orth, n_gt, reduction="none")
        self.assertEqual(loss_none.shape, (3,))
        torch.testing.assert_close(loss_none, torch.tensor([1.0, 1.0, 1.0], device=self.device))

        loss_sum = hermite_normal_loss(n_orth, n_gt, reduction="sum")
        self.assertAlmostEqual(float(loss_sum.item()), 3.0, places=6)

        # 5. Masked loss: ignore void voxels
        mask = torch.tensor([True, False, True], device=self.device)
        n_mixed = n_gt.clone()
        n_mixed[1] = -n_gt[1]  # Inactive index has opposite normal
        # Without mask, losses are [0, 2, 0] -> mean is 2/3
        # With mask selecting [0, 2], losses are [0, 0] -> mean is 0.0
        loss_masked = hermite_normal_loss(n_mixed, n_gt, active_mask=mask)
        self.assertAlmostEqual(float(loss_masked.item()), 0.0, places=5)

        # 6. Gradient propagation
        pred_var = torch.randn(5, 3, device=self.device, requires_grad=True)
        gt_var = F.normalize(torch.randn(5, 3, device=self.device), p=2, dim=-1)
        loss_grad = hermite_normal_loss(pred_var, gt_var)
        loss_grad.backward()
        self.assertIsNotNone(pred_var.grad)
        self.assertTrue(torch.isfinite(pred_var.grad).all())

    def test_sc_vae_hermite_head_integration(self):
        """Verify SC-VAE shape mode includes hermite_head and multi-task supervision works."""
        coords = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 4, 6], [0, 3, 5, 7]], device=self.device)
        features = torch.rand(4, 7, device=self.device)
        features[:, 3:6] = features[:, 3:6] > 0.5
        gt_normals = F.normalize(torch.randn(4, 3, device=self.device), p=2, dim=-1)

        model = SparseCompressionVAE(
            mode="shape",
            channels=(8, 16, 32, 64, 128),
            latent_channels=4,
        ).to(self.device)

        self.assertIsNotNone(model.hermite_head)
        self.assertIsInstance(model.hermite_head, HermiteNormalHead)

        # Test decode with return_hermite=True
        posterior = model.encode(coords, features)
        out_coords, raw, logits, hermite = model.decode(
            posterior.coords,
            posterior.mean,
            teacher_occupancy=posterior.occupancy,
            return_hermite=True,
        )
        self.assertEqual(hermite.shape, (len(out_coords), 3))
        lens = torch.linalg.norm(hermite, ord=2, dim=-1)
        torch.testing.assert_close(lens, torch.ones_like(lens), atol=1e-5, rtol=1e-5)

        # Test reconstruction_loss with hermite supervision
        loss, parts = reconstruction_loss(
            model,
            coords,
            features,
            stochastic=False,
            hermite_target=gt_normals,
            hermite_weight=1.0,
        )
        self.assertIn("hermite", parts)
        self.assertTrue(torch.isfinite(parts["hermite"]))
        loss.backward()

        # Verify all parameters in hermite_head have finite gradients
        for p in model.hermite_head.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())

    # --------------------------------------------------------------------------
    # 2. PBR Flow Network Forward Pass & Channels Tests
    # --------------------------------------------------------------------------
    def test_pbr_flow_forward_and_channels(self):
        """Verify PBR flow forward pass and output texture channels (Basecolor, Metallic-Roughness, Normal map)."""
        b, h, w = 2, 16, 16
        flow_net = SurfaceGroundedPBRFlow(
            in_channels=8,
            out_channels=8,
            geom_channels=3,
            latent_channels=16,
            channels=(16, 32, 64),
            time_dim=16,
            cond_dim=32,
        ).to(self.device)

        x = torch.randn(b, 8, h, w, device=self.device)
        time = 0.5
        geom_cond = torch.randn(b, 3, h, w, device=self.device)
        latent_cond = torch.randn(b, 16, device=self.device)

        # 1. Forward with return_dict=False returns raw velocity tensor
        vel = flow_net(x, time, geom_cond=geom_cond, latent_cond=latent_cond, return_dict=False)
        self.assertEqual(vel.shape, (b, 8, h, w))
        self.assertTrue(torch.isfinite(vel).all())

        # 2. Forward with return_dict=True returns PBRFlowOutput container
        out = flow_net(x, time, geom_cond=geom_cond, latent_cond=latent_cond, return_dict=True)
        self.assertIsInstance(out, PBRFlowOutput)

        # Check Basecolor: (B, 3, H, W) in [0, 1] with Rec. 709 debaked luminance
        self.assertEqual(out.basecolor.shape, (b, 3, h, w))
        self.assertTrue((out.basecolor >= 0.0).all())
        self.assertTrue((out.basecolor <= 1.0).all())

        # Check Metallic-Roughness: (B, 2, H, W) in [0, 1]
        self.assertEqual(out.metallic_roughness.shape, (b, 2, h, w))
        self.assertTrue((out.metallic_roughness >= 0.0).all())
        self.assertTrue((out.metallic_roughness <= 1.0).all())

        # Check Normal map: (B, 3, H, W) in [0, 1]
        self.assertEqual(out.normal_map.shape, (b, 3, h, w))
        self.assertTrue((out.normal_map >= 0.0).all())
        self.assertTrue((out.normal_map <= 1.0).all())

        # Check combined output
        self.assertEqual(out.combined.shape, (b, 8, h, w))

        # Check container indexing
        torch.testing.assert_close(out["basecolor"], out.basecolor)
        torch.testing.assert_close(out["metallic_roughness"], out.metallic_roughness)
        torch.testing.assert_close(out["normal_map"], out.normal_map)
        torch.testing.assert_close(out[0], out.velocity)

    def test_pbr_flow_neutral_normal_map(self):
        """Verify that zero / flat normal vector maps to neutral tangent normal [0.5, 0.5, 1.0]."""
        raw = torch.zeros(1, 8, 4, 4, device=self.device)
        # raw normal channels 5:8 are 0.0
        out = format_pbr_textures(raw)

        # Neutral tangent normal is [0.5, 0.5, 1.0]
        expected_normal = torch.tensor([0.5, 0.5, 1.0], device=self.device).view(1, 3, 1, 1).expand(1, 3, 4, 4)
        torch.testing.assert_close(out.normal_map, expected_normal, atol=1e-5, rtol=1e-5)

    def test_pbr_flow_training_loss_and_sampling(self):
        """Verify pbr_flow_loss, backprop optimization step, and Euler sampling."""
        b, h, w = 2, 8, 8
        net = SurfaceGroundedPBRFlow(
            in_channels=8,
            out_channels=8,
            geom_channels=3,
            latent_channels=8,
            channels=(8, 16, 32),
            time_dim=8,
            cond_dim=16,
        ).to(self.device)

        clean = torch.rand(b, 8, h, w, device=self.device)
        geom = torch.randn(b, 3, h, w, device=self.device)
        latent = torch.randn(b, 8, device=self.device)
        mask = torch.ones(b, h, w, dtype=torch.bool, device=self.device)
        mask[:, :2, :2] = False  # Some uncovered texels

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss = pbr_flow_loss(net, clean, geom_cond=geom, latent_cond=latent, valid_mask=mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()

        for p in net.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())

        # Test Euler sampling
        sampled = sample_pbr_flow(net, (1, 8, 8, 8), steps=3, device=self.device)
        self.assertEqual(sampled.basecolor.shape, (1, 3, 8, 8))
        self.assertEqual(sampled.metallic_roughness.shape, (1, 2, 8, 8))
        self.assertEqual(sampled.normal_map.shape, (1, 3, 8, 8))

    # --------------------------------------------------------------------------
    # 3. Luminance Debaking & UV Dilation Padding Tests
    # --------------------------------------------------------------------------
    def test_luminance_clamping_bounds(self):
        """Verify Rec. 709 luminance debaking clamping strictly to [0.04, 0.96]."""
        # 1. Pitch black [0, 0, 0] must receive minimum luminance 0.04
        black_tensor = torch.zeros(1, 3, 4, 4, device=self.device)
        debaked_black = debake_luminance(black_tensor, min_lum=0.04, max_lum=0.96)
        weights_t = torch.tensor([0.2126, 0.7152, 0.0722], device=self.device).view(1, 3, 1, 1)
        lum_black = (debaked_black * weights_t).sum(dim=1)
        torch.testing.assert_close(lum_black, torch.full_like(lum_black, 0.04), atol=1e-5, rtol=1e-5)

        # 2. Pure white [1, 1, 1] must be clamped down to maximum luminance 0.96
        white_tensor = torch.ones(1, 3, 4, 4, device=self.device)
        debaked_white = debake_luminance(white_tensor, min_lum=0.04, max_lum=0.96)
        lum_white = (debaked_white * weights_t).sum(dim=1)
        torch.testing.assert_close(lum_white, torch.full_like(lum_white, 0.96), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(debaked_white, torch.full_like(debaked_white, 0.96), atol=1e-5, rtol=1e-5)

        # 3. Neutral midtone [0.5, 0.5, 0.5] in [0.04, 0.96] must remain exactly unchanged
        mid_tensor = torch.full((1, 3, 4, 4), 0.5, device=self.device)
        debaked_mid = debake_luminance(mid_tensor, min_lum=0.04, max_lum=0.96)
        torch.testing.assert_close(debaked_mid, mid_tensor, atol=1e-5, rtol=1e-5)

        # 4. Deep shadow: dark red [0.01, 0.0, 0.0] -> scaled proportionally, preserving hue
        dark_red = torch.tensor([[[[0.01]], [[0.0]], [[0.0]]]], device=self.device)
        debaked_red = debake_luminance(dark_red, min_lum=0.04, max_lum=0.96)
        lum_red = float((debaked_red * weights_t).sum().item())
        self.assertAlmostEqual(lum_red, 0.04, places=4)
        # G and B remain 0 (chromaticity preserved)
        self.assertAlmostEqual(float(debaked_red[0, 1, 0, 0].item()), 0.0, places=5)
        self.assertAlmostEqual(float(debaked_red[0, 2, 0, 0].item()), 0.0, places=5)

        # 5. NumPy compatibility
        black_np = np.zeros((4, 4, 3), dtype=np.float32)
        debaked_np = debake_luminance(black_np, min_lum=0.04, max_lum=0.96)
        lum_np = np.sum(debaked_np * LUMINANCE_WEIGHTS, axis=-1)
        np.testing.assert_allclose(lum_np, 0.04, atol=1e-5)

    def test_uv_dilation_padding(self):
        """Verify distance-transform nearest-neighbor dilation padding for UV seams."""
        # Create an 8x8 texture atlas with a 4x4 active island at the center
        h, w = 8, 8
        img = torch.zeros(1, 3, h, w, device=self.device)
        # Center 4x4 is red [1.0, 0.2, 0.3]
        img[:, 0, 2:6, 2:6] = 1.0
        img[:, 1, 2:6, 2:6] = 0.2
        img[:, 2, 2:6, 2:6] = 0.3

        valid_mask = torch.zeros(1, h, w, dtype=torch.bool, device=self.device)
        valid_mask[:, 2:6, 2:6] = True

        padded = pad_texture_dilation(img, valid_mask)
        self.assertEqual(padded.shape, img.shape)

        # 1. Covered texels must be 100% strictly identical
        torch.testing.assert_close(padded[:, :, 2:6, 2:6], img[:, :, 2:6, 2:6])

        # 2. Uncovered texels must be filled with nearest neighbor (which is the center color)
        # Specifically, the border texels at (1, 2) must equal the edge texel at (2, 2)
        torch.testing.assert_close(padded[:, :, 1, 2], img[:, :, 2, 2])
        torch.testing.assert_close(padded[:, :, 0, 0], img[:, :, 2, 2])

        # 3. Test on NumPy array
        img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mask_np = valid_mask.squeeze(0).cpu().numpy()
        padded_np = pad_texture_dilation(img_np, mask_np)
        np.testing.assert_allclose(padded_np[2:6, 2:6], img_np[2:6, 2:6])
        np.testing.assert_allclose(padded_np[0, 0], img_np[2, 2])

        # 4. Edge case: all covered -> no change
        all_true = torch.ones_like(valid_mask)
        torch.testing.assert_close(pad_texture_dilation(img, all_true), img)


if __name__ == "__main__":
    unittest.main()
