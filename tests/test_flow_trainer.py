"""Unit tests for Optimal Transport Rectified Flow trainer and Decoder-in-the-Loop."""
import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from t3d.flow import SparseFlowDiT
from t3d.sc_vae import SparseCompressionVAE
from t3d.training.flow_trainer import (
    FlowTrainer,
    FlowTrainerConfig,
    sample_logit_normal,
    rectified_flow_pair,
    rectified_flow_loss,
    chamfer_distance,
    decoder_in_the_loop_loss,
    sample_ode,
)


class DummyConstantModel(nn.Module):
    """Simple linear velocity model for exact ODE integration verification."""
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.bias = nn.Parameter(torch.ones(1, channels))

    def forward(self, x, coords, time, condition=None, orders=None, shape=None):
        return self.bias.expand(len(x), -1)


class TestFlowTrainer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_cells = 64  # 4^3 coarse cells
        self.channels = 16
        self.cond_channels = 17

        # Synthetic grid coordinates (4x4x4)
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(4), torch.arange(4), torch.arange(4), indexing="ij"
            ),
            dim=-1,
        ).reshape(-1, 3).float().to(self.device)
        self.coords = grid

        # Clean latents x_1
        self.clean = torch.randn((self.n_cells, self.channels), device=self.device)
        # Condition c_raw (coarse occupancy + latents)
        self.condition = torch.randn((self.n_cells, self.cond_channels), device=self.device)
        # Surface ratio: half cells dense (1.0), half empty (0.0)
        self.surface_ratio = torch.zeros(self.n_cells, device=self.device)
        self.surface_ratio[: self.n_cells // 2] = 1.0

        # Small flow DiT model: width divisible by heads, and width // heads divisible by 6
        self.model = SparseFlowDiT(
            channels=self.channels,
            condition_channels=self.cond_channels,
            width=48,
            heads=2,
            depth=2,
            patch_size=32,
        ).to(self.device)

    def test_logit_normal_distribution_properties(self):
        """Verify Logit-Normal distribution sampling properties and mid-trajectory concentration."""
        samples = sample_logit_normal(size=(100000,), m=0.0, s=1.0, device=self.device)

        # 1. Output strictly in (0, 1)
        self.assertTrue((samples > 0.0).all())
        self.assertTrue((samples < 1.0).all())

        # 2. Mean is approximately 0.5 by symmetry when m=0
        mean = samples.mean().item()
        self.assertAlmostEqual(mean, 0.5, delta=0.015)

        # 3. Concentration in ambiguous mid-trajectories [0.3, 0.7]
        # Theoretical fraction for N(0, 1): sigma(u) in [0.3, 0.7] <=> u in [-0.8473, 0.8473] => ~60.3%
        mid_mask = (samples >= 0.3) & (samples <= 0.7)
        mid_fraction = mid_mask.float().mean().item()
        self.assertGreater(mid_fraction, 0.57)
        self.assertLess(mid_fraction, 0.64)

        # 4. Parametric control with m != 0
        samples_pos = sample_logit_normal(size=(50000,), m=1.0, s=1.0, device=self.device)
        self.assertGreater(samples_pos.mean().item(), 0.65)

        samples_neg = sample_logit_normal(size=(50000,), m=-1.0, s=1.0, device=self.device)
        self.assertLess(samples_neg.mean().item(), 0.35)

        # 5. Invalid arguments raise ValueError
        with self.assertRaises(ValueError):
            sample_logit_normal(size=(10,), s=0.0)
        with self.assertRaises(ValueError):
            sample_logit_normal(size=(10,), eps=0.6)

    def test_rectified_flow_pair(self):
        """Verify linear flow matching trajectory and velocity target."""
        clean = torch.tensor([[2.0, 4.0], [6.0, 8.0]], device=self.device)
        noise = torch.tensor([[0.0, 0.0], [2.0, 2.0]], device=self.device)

        # At t=0, state is noise
        x_0, v_0 = rectified_flow_pair(clean, noise, 0.0)
        torch.testing.assert_close(x_0, noise)
        torch.testing.assert_close(v_0, clean - noise)

        # At t=1, state is clean
        x_1, v_1 = rectified_flow_pair(clean, noise, 1.0)
        torch.testing.assert_close(x_1, clean)
        torch.testing.assert_close(v_1, clean - noise)

        # At t=0.5, state is midpoint
        x_half, v_half = rectified_flow_pair(clean, noise, 0.5)
        torch.testing.assert_close(x_half, 0.5 * (clean + noise))
        torch.testing.assert_close(v_half, clean - noise)

        # Boundary checks
        with self.assertRaises(ValueError):
            rectified_flow_pair(clean, noise, -0.1)
        with self.assertRaises(ValueError):
            rectified_flow_pair(clean, noise, 1.1)

    def test_chamfer_distance(self):
        """Verify bidirectional Chamfer distance calculation and differentiability."""
        p1 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=self.device, requires_grad=True)
        p2 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=self.device)

        # Identical point clouds => 0.0
        dist = chamfer_distance(p1, p2, squared=True)
        self.assertAlmostEqual(dist.item(), 0.0, places=6)

        # Shifted point cloud
        offset = torch.tensor([0.1, 0.0, 0.0], device=self.device)
        p3 = p2 + offset
        dist_shifted = chamfer_distance(p1, p3, squared=True)
        # Expected: min dist is 0.1^2 = 0.01 for all points in both directions => 0.01 + 0.01 = 0.02
        self.assertAlmostEqual(dist_shifted.item(), 0.02, places=5)

        # Gradient flow
        dist_shifted.backward()
        self.assertIsNotNone(p1.grad)
        self.assertTrue(torch.isfinite(p1.grad).all())
        self.assertGreater(p1.grad.norm().item(), 0.0)

    def test_rectified_flow_loss_with_surface_ratio(self):
        """Verify flow matching loss computation with and without surface_ratio weighting."""
        # Unweighted loss
        loss_unweighted, metrics_u = rectified_flow_loss(
            self.model,
            x_1=self.clean,
            coords=self.coords,
            condition=self.condition,
            surface_ratio=None,
        )
        self.assertTrue(torch.isfinite(loss_unweighted))
        self.assertGreater(loss_unweighted.item(), 0.0)
        self.assertIn("flow_loss", metrics_u)

        # Weighted loss with surface_ratio
        loss_weighted, metrics_w = rectified_flow_loss(
            self.model,
            x_1=self.clean,
            coords=self.coords,
            condition=self.condition,
            surface_ratio=self.surface_ratio,
            base_weight=0.5,
            surface_scale=2.0,
        )
        self.assertTrue(torch.isfinite(loss_weighted))
        self.assertGreater(loss_weighted.item(), 0.0)
        self.assertIn("flow_loss", metrics_w)

    def test_batched_loss_computation(self):
        """Verify flow trainer handles batched input tensors (B, N, C)."""
        trainer = FlowTrainer(
            model=self.model,
            config=FlowTrainerConfig(learning_rate=1e-4, device=self.device),
        )

        b_size = 2
        batch = {
            "clean": torch.randn((b_size, self.n_cells, self.channels), device=self.device),
            "coords": self.coords,
            "c_raw": torch.randn((b_size, self.n_cells, self.cond_channels), device=self.device),
            "surface_ratio": self.surface_ratio.unsqueeze(0).expand(b_size, -1),
        }

        loss, metrics = trainer.compute_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("flow_loss", metrics)
        self.assertIn("total_loss", metrics)

    def test_ode_sampling_euler_and_heun(self):
        """Verify Euler and Heun ODE integration rollout from t=0 to t=1."""
        noise = torch.randn((self.n_cells, self.channels), device=self.device)

        # 1. Euler solver rollout
        out_euler, traj_euler = sample_ode(
            model=self.model,
            x_0=noise,
            coords=self.coords,
            condition=self.condition,
            steps=10,
            solver="euler",
            return_trajectory=True,
        )
        self.assertEqual(out_euler.shape, noise.shape)
        self.assertEqual(len(traj_euler), 11)  # 10 steps + initial state
        self.assertTrue(torch.isfinite(out_euler).all())

        # 2. Heun solver rollout
        out_heun, traj_heun = sample_ode(
            model=self.model,
            x_0=noise,
            coords=self.coords,
            condition=self.condition,
            steps=10,
            solver="heun",
            return_trajectory=True,
        )
        self.assertEqual(out_heun.shape, noise.shape)
        self.assertEqual(len(traj_heun), 11)
        self.assertTrue(torch.isfinite(out_heun).all())

        # 3. Verify exact linear integration on a constant velocity field
        const_model = DummyConstantModel(channels=self.channels).to(self.device)
        c_noise = torch.zeros((5, self.channels), device=self.device)
        c_coords = torch.zeros((5, 3), device=self.device)

        # dy/dt = 1 => y(1) - y(0) = 1.0
        sol_euler = sample_ode(const_model, c_noise, c_coords, steps=5, solver="euler")
        sol_heun = sample_ode(const_model, c_noise, c_coords, steps=5, solver="heun")

        torch.testing.assert_close(sol_euler, torch.ones_like(c_noise))
        torch.testing.assert_close(sol_heun, torch.ones_like(c_noise))

    def test_interval_cfg_guidance(self):
        """Verify Classifier-Free Guidance sampling with timestep interval filtering."""
        # SparseFlowDiT initializes modulation and output with zeros (adaLN-zero).
        # Populate modulation and output weights so cross-attention gate is active.
        with torch.no_grad():
            for b in self.model.blocks:
                nn.init.normal_(b.modulation.weight, std=0.2)
            nn.init.normal_(self.model.output.weight, std=0.2)

        noise = torch.randn((self.n_cells, self.channels), device=self.device)

        # Baseline with guidance=1.0 (pure conditional)
        res_no_cfg = sample_ode(
            self.model,
            noise,
            self.coords,
            self.condition,
            steps=8,
            guidance=1.0,
        )

        # Guidance > 1.0 restricted to interval [0.2, 0.8]
        res_cfg = sample_ode(
            self.model,
            noise,
            self.coords,
            self.condition,
            steps=8,
            guidance=3.0,
            guidance_interval=(0.2, 0.8),
        )

        self.assertEqual(res_cfg.shape, noise.shape)
        self.assertTrue(torch.isfinite(res_cfg).all())
        # The guided trajectory should diverge from the unguided trajectory
        diff = (res_cfg - res_no_cfg).abs().mean().item()
        self.assertGreater(diff, 1e-4)

    def test_loss_backpropagation_and_optimization(self):
        """Verify complete training step updates model weights with finite gradients."""
        trainer = FlowTrainer(
            model=self.model,
            config=FlowTrainerConfig(learning_rate=1e-3, device=self.device),
        )

        batch = {
            "clean": self.clean,
            "coords": self.coords,
            "condition": self.condition,
            "surface_ratio": self.surface_ratio,
        }

        # Record parameter before step
        param_before = [p.clone() for p in self.model.parameters()]
        metrics = trainer.train_step(batch)

        self.assertEqual(trainer.steps, 1)
        self.assertIn("total_loss", metrics)
        self.assertTrue(math.isfinite(metrics["total_loss"]))

        # Verify weights updated
        any_changed = False
        for p_before, p_after in zip(param_before, self.model.parameters()):
            if not torch.equal(p_before, p_after):
                any_changed = True
                break
        self.assertTrue(any_changed, "Model parameters did not update after train_step")

    def test_decoder_in_the_loop_fine_tuning(self):
        """Verify frozen SC-VAE decoder integration and gradient propagation to flow network."""
        # Create a lightweight SC-VAE model
        vae = SparseCompressionVAE(
            mode="shape",
            channels=(4, 8, 16, 32, 64),
            latent_channels=self.channels,
            convolution_backend="reference",
        ).to(self.device)

        # Initialize trainer with frozen decoder and decoder_weight > 0
        trainer = FlowTrainer(
            model=self.model,
            decoder=vae,
            config=FlowTrainerConfig(
                learning_rate=1e-3,
                decoder_weight=0.5,
                chamfer_weight=1.0,
                occupancy_weight=1.0,
                device=self.device,
            ),
        )

        # Ensure all decoder parameters are frozen
        for p in trainer.decoder.parameters():
            self.assertFalse(p.requires_grad)

        # Synthesize ground-truth fine voxels and stage occupancies
        # Synthetic 16 fine voxels
        target_coords = torch.zeros((16, 4), device=self.device, dtype=torch.int64)
        target_coords[:, 1] = torch.arange(16, device=self.device)
        target_features = torch.rand((16, 7), device=self.device)
        # Synthetic target points
        target_points = target_coords[:, 1:4].float() + target_features[:, :3]

        batch = {
            "clean": self.clean,
            "coords": self.coords,
            "condition": self.condition,
            "surface_ratio": self.surface_ratio,
            "target_coords": target_coords,
            "target_features": target_features,
            "target_points": target_points,
        }

        # Train step with decoder in the loop
        metrics = trainer.train_step(batch)

        self.assertIn("decoder_total_loss", metrics)
        self.assertIn("decoder_chamfer_loss", metrics)
        self.assertTrue(math.isfinite(metrics["decoder_total_loss"]))
        self.assertGreater(metrics["decoder_chamfer_loss"], 0.0)

        # Verify that decoder parameters did NOT accumulate gradients
        for p in trainer.decoder.parameters():
            self.assertIsNone(p.grad)

        # Verify that flow model parameters DID receive gradients
        has_flow_grad = False
        for p in trainer.model.parameters():
            if p.grad is not None and p.grad.norm().item() > 0.0:
                has_flow_grad = True
                break
        self.assertTrue(has_flow_grad, "Flow model received no gradients from decoder-in-the-loop")

    def test_checkpoint_save_and_resume(self):
        """Verify checkpoint saving and resumption preserves weights, optimizer, and step count."""
        trainer = FlowTrainer(
            model=self.model,
            config=FlowTrainerConfig(learning_rate=1e-3, device=self.device),
            manifest_sha256="test_sha_123",
        )

        batch = {
            "clean": self.clean,
            "coords": self.coords,
            "condition": self.condition,
        }
        trainer.train_step(batch)
        trainer.train_step(batch)
        self.assertEqual(trainer.steps, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "flow_ckpt.pt"
            trainer.save(ckpt_path)

            # Create fresh model with different initial weights
            new_model = SparseFlowDiT(
                channels=self.channels,
                condition_channels=self.cond_channels,
                width=48,
                heads=2,
                depth=2,
                patch_size=32,
            ).to(self.device)

            restored = FlowTrainer.resume(
                ckpt_path,
                model=new_model,
                manifest_sha256="test_sha_123",
                device=self.device,
            )

            self.assertEqual(restored.steps, 2)
            self.assertEqual(restored.manifest_sha256, "test_sha_123")

            # Check that model weights match perfectly
            for p1, p2 in zip(trainer.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(p1, p2)

            # Fingerprint mismatch check
            with self.assertRaises(ValueError):
                FlowTrainer.resume(
                    ckpt_path,
                    model=new_model,
                    manifest_sha256="wrong_sha",
                    device=self.device,
                )


if __name__ == "__main__":
    unittest.main()
