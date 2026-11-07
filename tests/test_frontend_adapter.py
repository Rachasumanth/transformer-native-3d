"""Unit test suite for Multimodal Front-End Adapter.

Tests cover:
- Parameter initialization with defaults and custom configurations.
- 3D coordinate positional embedding shape, bounds [-0.45, 0.45]^3, and determinism.
- Text conditioning forward pass (sequence tokens and pooled embeddings).
- Image conditioning forward pass (patch tokens and pooled embeddings).
- Joint text + image multimodal conditioning forward pass and mixed combinations.
- Unconditional conditioning forward pass with learnable null context.
- Backward pass and gradient propagation to all projection and query layers.
- Adapter loss computation and active cell latent masking.
- Acceptance gate compliance: coarse occupancy recall on predicted logits >= 0.98.
- End-to-end integration with UnifiedDiT via c_raw conditioning.
"""

from __future__ import annotations

import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    import torch.nn.functional as F
    from t3d.models.frontend_adapter import (
        MultimodalFrontEndAdapter,
        CoordinatePositionalEmbedding,
        AdapterTransformerBlock,
        adapter_loss,
        compute_occupancy_metrics,
    )
    from t3d.models.unified_dit import UnifiedDiT


@unittest.skipUnless(HAS_TORCH, "PyTorch required for MultimodalFrontEndAdapter tests")
class TestMultimodalFrontEndAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)

    def test_instantiation_defaults_and_configs(self) -> None:
        """Verify model initialization with exact default specifications and custom configs."""
        # 1. Default initialization
        adapter = MultimodalFrontEndAdapter().to(self.device)
        self.assertEqual(adapter.text_dim, 512)
        self.assertEqual(adapter.image_dim, 768)
        self.assertEqual(adapter.embed_dim, 256)
        self.assertEqual(adapter.latent_channels, 16)
        self.assertEqual(adapter.coarse_res, 4)
        self.assertEqual(adapter.num_tokens, 64)  # 4^3 = 64
        self.assertEqual(adapter.num_layers, 4)
        self.assertEqual(adapter.num_heads, 4)
        self.assertEqual(len(adapter.blocks), 4)
        self.assertEqual(adapter.coords.shape, (64, 3))
        self.assertEqual(adapter.voxel_queries.shape, (1, 64, 256))

        # 2. Custom configuration
        custom = MultimodalFrontEndAdapter(
            text_dim=256,
            image_dim=512,
            embed_dim=128,
            latent_channels=8,
            coarse_res=2,
            num_layers=2,
            num_heads=2,
        ).to(self.device)
        self.assertEqual(custom.text_dim, 256)
        self.assertEqual(custom.image_dim, 512)
        self.assertEqual(custom.embed_dim, 128)
        self.assertEqual(custom.latent_channels, 8)
        self.assertEqual(custom.coarse_res, 2)
        self.assertEqual(custom.num_tokens, 8)  # 2^3 = 8
        self.assertEqual(custom.num_layers, 2)
        self.assertEqual(custom.coords.shape, (8, 3))

    def test_coordinate_positional_embedding_shape_and_determinism(self) -> None:
        """Verify 3D grid coordinate range, positional embedding shape, and determinism."""
        adapter = MultimodalFrontEndAdapter(coarse_res=4, embed_dim=256).to(self.device)
        coords = adapter.coords  # (64, 3)

        # Check bounds: coordinates must strictly lie within [-0.45, 0.45]^3 (10% empty margin)
        self.assertEqual(coords.shape, (64, 3))
        self.assertAlmostEqual(coords.min().item(), -0.45, places=5)
        self.assertAlmostEqual(coords.max().item(), 0.45, places=5)
        self.assertTrue((coords >= -0.45 - 1e-6).all().item())
        self.assertTrue((coords <= 0.45 + 1e-6).all().item())

        # Forward pass through CoordinatePositionalEmbedding directly
        pos_emb = adapter.pos_embed(coords)
        self.assertEqual(pos_emb.shape, (64, 256))

        # Determinism check: multiple calls with identical input must yield bitwise identical output
        pos_emb_2 = adapter.pos_embed(coords)
        self.assertTrue(torch.equal(pos_emb, pos_emb_2))

        # Batched input check: (B, N, 3) -> (B, N, embed_dim)
        batched_coords = coords.unsqueeze(0).expand(3, -1, -1)
        batched_pos = adapter.pos_embed(batched_coords)
        self.assertEqual(batched_pos.shape, (3, 64, 256))
        for b in range(3):
            self.assertTrue(torch.equal(batched_pos[b], pos_emb))

    def test_text_conditioning_forward_pass(self) -> None:
        """Verify forward pass with text sequence tokens and pooled text embeddings."""
        adapter = MultimodalFrontEndAdapter().to(self.device)
        B, L_text = 2, 12

        # 1. Sequence tokens: (B, L_text, text_dim)
        text_tokens = torch.randn(B, L_text, 512, device=self.device)
        out = adapter(text_tokens=text_tokens)

        self.assertIn("occ_logits", out)
        self.assertIn("latents", out)
        self.assertIn("c_raw", out)

        self.assertEqual(out["occ_logits"].shape, (B, 64))
        self.assertEqual(out["latents"].shape, (B, 64, 16))
        self.assertEqual(out["c_raw"].shape, (B, 64, 17))

        # Check channel semantics of c_raw: channel 0 is occ_logits, channels 1..C are latents
        self.assertTrue(torch.equal(out["c_raw"][..., 0], out["occ_logits"]))
        self.assertTrue(torch.equal(out["c_raw"][..., 1:], out["latents"]))

        # 2. Pooled text embedding: (B, text_dim)
        text_pooled = torch.randn(B, 512, device=self.device)
        out_pooled = adapter(text_pooled=text_pooled)
        self.assertEqual(out_pooled["occ_logits"].shape, (B, 64))
        self.assertEqual(out_pooled["latents"].shape, (B, 64, 16))
        self.assertEqual(out_pooled["c_raw"].shape, (B, 64, 17))

        # 3. Unbatched single text input
        single_text = torch.randn(512, device=self.device)
        out_single = adapter(text_pooled=single_text)
        self.assertEqual(out_single["occ_logits"].shape, (1, 64))
        self.assertEqual(out_single["c_raw"].shape, (1, 64, 17))

    def test_image_conditioning_forward_pass(self) -> None:
        """Verify forward pass with image patch tokens and pooled image embeddings."""
        adapter = MultimodalFrontEndAdapter().to(self.device)
        B, L_img = 3, 16

        # 1. Patch tokens: (B, L_img, image_dim)
        image_tokens = torch.randn(B, L_img, 768, device=self.device)
        out = adapter(image_tokens=image_tokens)

        self.assertEqual(out["occ_logits"].shape, (B, 64))
        self.assertEqual(out["latents"].shape, (B, 64, 16))
        self.assertEqual(out["c_raw"].shape, (B, 64, 17))

        # Verify c_raw channels
        self.assertTrue(torch.equal(out["c_raw"][..., 0], out["occ_logits"]))
        self.assertTrue(torch.equal(out["c_raw"][..., 1:], out["latents"]))

        # 2. Pooled image embedding: (B, image_dim)
        image_pooled = torch.randn(B, 768, device=self.device)
        out_pooled = adapter(image_pooled=image_pooled)
        self.assertEqual(out_pooled["occ_logits"].shape, (B, 64))
        self.assertEqual(out_pooled["latents"].shape, (B, 64, 16))
        self.assertEqual(out_pooled["c_raw"].shape, (B, 64, 17))

    def test_joint_text_image_conditioning_forward_pass(self) -> None:
        """Verify joint text + image conditioning and mixed multimodal combinations."""
        adapter = MultimodalFrontEndAdapter().to(self.device)
        B = 2

        text_tokens = torch.randn(B, 10, 512, device=self.device)
        image_tokens = torch.randn(B, 16, 768, device=self.device)
        text_pooled = torch.randn(B, 512, device=self.device)
        image_pooled = torch.randn(B, 768, device=self.device)

        # 1. Joint text_tokens + image_tokens
        out_joint = adapter(text_tokens=text_tokens, image_tokens=image_tokens)
        self.assertEqual(out_joint["occ_logits"].shape, (B, 64))
        self.assertEqual(out_joint["latents"].shape, (B, 64, 16))
        self.assertEqual(out_joint["c_raw"].shape, (B, 64, 17))

        # 2. Joint text_pooled + image_tokens
        out_mixed1 = adapter(text_pooled=text_pooled, image_tokens=image_tokens)
        self.assertEqual(out_mixed1["c_raw"].shape, (B, 64, 17))

        # 3. Joint text_tokens + image_pooled
        out_mixed2 = adapter(text_tokens=text_tokens, image_pooled=image_pooled)
        self.assertEqual(out_mixed2["c_raw"].shape, (B, 64, 17))

        # 4. Joint text_pooled + image_pooled
        out_mixed3 = adapter(text_pooled=text_pooled, image_pooled=image_pooled)
        self.assertEqual(out_mixed3["c_raw"].shape, (B, 64, 17))

        # 5. Batch size mismatch validation
        mismatched_image = torch.randn(B + 1, 16, 768, device=self.device)
        with self.assertRaises(ValueError):
            adapter(text_tokens=text_tokens, image_tokens=mismatched_image)

    def test_unconditional_forward_pass(self) -> None:
        """Verify unconditional forward pass using learnable null context token."""
        adapter = MultimodalFrontEndAdapter().to(self.device)

        # Explicit batch size
        out = adapter(batch_size=3)
        self.assertEqual(out["occ_logits"].shape, (3, 64))
        self.assertEqual(out["latents"].shape, (3, 64, 16))
        self.assertEqual(out["c_raw"].shape, (3, 64, 17))

        # Default batch size 1
        out_default = adapter()
        self.assertEqual(out_default["c_raw"].shape, (1, 64, 17))

    def test_backward_pass_and_gradient_propagation(self) -> None:
        """Verify gradient backpropagation to all projection, transformer, and query layers."""
        adapter = MultimodalFrontEndAdapter().to(self.device)
        adapter.train()

        B = 2
        text_tokens = torch.randn(B, 8, 512, device=self.device)
        image_tokens = torch.randn(B, 12, 768, device=self.device)
        target_occ = (torch.rand(B, 64, device=self.device) > 0.5).float()
        target_lat = torch.randn(B, 64, 16, device=self.device)

        out = adapter(text_tokens=text_tokens, image_tokens=image_tokens)
        loss = adapter_loss(out, target_occ, target_lat=target_lat, occ_weight=1.0, lat_weight=1.0)
        loss.backward()

        # Check gradients for all expected active parameters
        self.assertIsNotNone(adapter.voxel_queries.grad)
        self.assertTrue(torch.isfinite(adapter.voxel_queries.grad).all())

        # Coordinate positional embedding parameters
        for name, param in adapter.pos_embed.named_parameters():
            self.assertIsNotNone(param.grad, f"pos_embed.{name} grad is None")
            self.assertTrue(torch.isfinite(param.grad).all(), f"pos_embed.{name} grad is non-finite")

        # Modality projectors
        for name, param in adapter.text_projector.named_parameters():
            self.assertIsNotNone(param.grad, f"text_projector.{name} grad is None")
            self.assertTrue(torch.isfinite(param.grad).all(), f"text_projector.{name} grad is non-finite")

        for name, param in adapter.image_projector.named_parameters():
            self.assertIsNotNone(param.grad, f"image_projector.{name} grad is None")
            self.assertTrue(torch.isfinite(param.grad).all(), f"image_projector.{name} grad is non-finite")

        # Modality type indicators
        self.assertIsNotNone(adapter.text_type_embed.grad)
        self.assertTrue(torch.isfinite(adapter.text_type_embed.grad).all())
        self.assertIsNotNone(adapter.image_type_embed.grad)
        self.assertTrue(torch.isfinite(adapter.image_type_embed.grad).all())

        # Transformer blocks
        for i, block in enumerate(adapter.blocks):
            for name, param in block.named_parameters():
                self.assertIsNotNone(param.grad, f"block {i}.{name} grad is None")
                self.assertTrue(torch.isfinite(param.grad).all(), f"block {i}.{name} grad is non-finite")

        # Output heads
        for name, param in adapter.occ_head.named_parameters():
            self.assertIsNotNone(param.grad, f"occ_head.{name} grad is None")
            self.assertTrue(torch.isfinite(param.grad).all())

        for name, param in adapter.latent_head.named_parameters():
            self.assertIsNotNone(param.grad, f"latent_head.{name} grad is None")
            self.assertTrue(torch.isfinite(param.grad).all())

        # Also verify null_context gradient in unconditional mode
        adapter.zero_grad()
        out_uncond = adapter(batch_size=1)
        loss_uncond = out_uncond["occ_logits"].sum()
        loss_uncond.backward()
        self.assertIsNotNone(adapter.null_context.grad)
        self.assertTrue(torch.isfinite(adapter.null_context.grad).all())

    def test_adapter_loss_function_and_masking(self) -> None:
        """Verify adapter loss computation and active cell latent masking."""
        B, N, C = 2, 64, 16
        occ_logits = torch.randn(B, N, device=self.device, requires_grad=True)
        pred_lat = torch.randn(B, N, C, device=self.device, requires_grad=True)
        c_raw = torch.cat([occ_logits.unsqueeze(-1), pred_lat], dim=-1)

        output = {"occ_logits": occ_logits, "latents": pred_lat, "c_raw": c_raw}

        # Sparse target occupancy: only first 5 voxels active
        target_occ = torch.zeros(B, N, device=self.device)
        target_occ[:, :5] = 1.0

        target_lat = torch.randn(B, N, C, device=self.device)

        # 1. Total loss scalar
        loss = adapter_loss(output, target_occ, target_lat=target_lat, occ_weight=1.0, lat_weight=2.0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(hasattr(loss, "loss_occ"))
        self.assertTrue(hasattr(loss, "loss_lat"))

        # 2. Return dict
        loss_dict = adapter_loss(
            output, target_occ, target_lat=target_lat, return_dict=True
        )
        self.assertIn("loss", loss_dict)
        self.assertIn("loss_occ", loss_dict)
        self.assertIn("loss_lat", loss_dict)

        # 3. Active cell masking verification:
        # Gradients on pred_lat must ONLY be non-zero on active cells (first 5 voxels)
        pred_lat_test = torch.randn(B, N, C, device=self.device, requires_grad=True)
        out_test = {"occ_logits": occ_logits, "latents": pred_lat_test}
        loss_mask = adapter_loss(out_test, target_occ, target_lat=target_lat, occ_weight=0.0, lat_weight=1.0)
        loss_mask.backward()

        self.assertIsNotNone(pred_lat_test.grad)
        active_grads = pred_lat_test.grad[:, :5, :]
        inactive_grads = pred_lat_test.grad[:, 5:, :]
        self.assertTrue((active_grads.abs() > 0).all().item())
        self.assertTrue((inactive_grads == 0).all().item())

    def test_compute_occupancy_metrics(self) -> None:
        """Verify binary classification metrics helper."""
        occ_logits = torch.tensor([2.0, -2.0, 3.0, -1.0])
        target_occ = torch.tensor([1.0, 0.0, 1.0, 0.0])

        metrics = compute_occupancy_metrics(occ_logits, target_occ, threshold=0.0)
        self.assertAlmostEqual(metrics["recall"], 1.0, places=4)
        self.assertAlmostEqual(metrics["precision"], 1.0, places=4)
        self.assertAlmostEqual(metrics["f1"], 1.0, places=4)
        self.assertAlmostEqual(metrics["accuracy"], 1.0, places=4)
        self.assertAlmostEqual(metrics["iou"], 1.0, places=4)

    def test_acceptance_gate_occupancy_recall(self) -> None:
        """Verify acceptance gate compliance: occupancy recall on predicted logits >= 0.98.

        Trains an adapter instance on sample multimodal conditioning for 35 iterations
        and verifies that predicted occupancy recall strictly exceeds 0.98.
        """
        torch.manual_seed(42)
        adapter = MultimodalFrontEndAdapter(
            text_dim=512,
            image_dim=768,
            embed_dim=256,
            latent_channels=16,
            coarse_res=4,
            num_layers=4,
            num_heads=4,
        ).to(self.device)

        B = 2
        text_tokens = torch.randn(B, 10, 512, device=self.device)
        image_tokens = torch.randn(B, 16, 768, device=self.device)

        # Ground truth occupancy with ~25% occupied voxels
        target_occ = (torch.rand(B, 64, device=self.device) > 0.75).float()
        # Ensure at least some occupied voxels exist
        target_occ[:, 0:8] = 1.0
        target_lat = torch.randn(B, 64, 16, device=self.device)

        # Fast optimization on sample conditioning
        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-2)
        for _ in range(35):
            optimizer.zero_grad()
            out = adapter(text_tokens=text_tokens, image_tokens=image_tokens)
            loss = adapter_loss(
                out,
                target_occ,
                target_lat=target_lat,
                occ_weight=1.0,
                lat_weight=0.5,
                pos_weight=3.0,  # Prioritize positive occupancy recall
            )
            loss.backward()
            optimizer.step()

        # Evaluate recall
        adapter.eval()
        with torch.no_grad():
            eval_out = adapter(text_tokens=text_tokens, image_tokens=image_tokens)
            metrics = compute_occupancy_metrics(
                eval_out["occ_logits"], target_occ, threshold=0.0
            )

        recall = metrics["recall"]
        print(f"\n[Milestone 5A Acceptance Gate] Coarse Occupancy Recall: {recall:.4f} (Required: >= 0.98)")
        self.assertGreaterEqual(
            recall,
            0.98,
            f"Acceptance gate failure: coarse occupancy recall {recall:.4f} < 0.98",
        )

    def test_unified_dit_interoperability(self) -> None:
        """Verify seamless end-to-end integration between MultimodalFrontEndAdapter and UnifiedDiT."""
        torch.manual_seed(42)
        adapter = MultimodalFrontEndAdapter(
            text_dim=512,
            embed_dim=256,
            latent_channels=16,
            coarse_res=4,
        ).to(self.device)

        dit = UnifiedDiT(
            dim=288,
            depth=2,
            heads=6,
            latent_channels=16,
            coarse_res=4,
        ).to(self.device)

        B = 2
        text_tokens = torch.randn(B, 8, 512, device=self.device)

        # 1. Front-end adapter produces c_raw
        adapter_out = adapter(text_tokens=text_tokens)
        c_raw = adapter_out["c_raw"]  # (B, 64, 17)
        self.assertEqual(c_raw.shape, (B, 64, 17))

        # 2. UnifiedDiT receives c_raw as conditioning
        x_noisy = torch.randn(B, 64, 17, device=self.device)
        timestep = torch.tensor([0.4, 0.7], device=self.device)

        pred_v = dit(x_noisy, timestep, c_raw=c_raw)
        self.assertEqual(pred_v.shape, (B, 64, 17))

        # 3. End-to-end backpropagation through DiT and Adapter
        loss = pred_v.sum()
        loss.backward()

        self.assertIsNotNone(adapter.voxel_queries.grad)
        self.assertTrue(torch.isfinite(adapter.voxel_queries.grad).all())
        self.assertIsNotNone(adapter.text_projector[0].weight.grad)
        self.assertTrue(torch.isfinite(adapter.text_projector[0].weight.grad).all())


if __name__ == "__main__":
    unittest.main()
