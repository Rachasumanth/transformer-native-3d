"""Unit test suite for Milestone 5B: End-to-End Inference Orchestrator (T3DPipeline).

Validates:
1. Pipeline initialization with configurable submodules, devices, dtypes, and dimensions.
2. End-to-end generate() from synthetic text tokens.
3. End-to-end generate() from precomputed c_raw conditioning.
4. End-to-end generate() from synthetic image tokens and multimodal combinations.
5. End-to-end unconditional generate().
6. Deterministic sampling with seed.
7. Watertight 2-manifold mesh generation (0 non-manifold edges, 1500 <= faces <= 5000).
8. glTF 2.0 Binary (.glb) export with embedded PBR materials and skinning buffers.
9. glTF 2.0 validation via trimesh.load and GLBAccessors (JOINTS_0, WEIGHTS_0).
10. Automated production acceptance gate evaluation via scripts.eval_acceptance.evaluate_asset.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import trimesh

from t3d.pipeline import T3DPipeline, T3DOutput
from t3d.models.frontend_adapter import MultimodalFrontEndAdapter
from t3d.models.unified_dit import UnifiedDiT
from t3d.models.rigging import SkeletonFlowModel, SurfacePointToBoneSkinning
from t3d.models.pbr_flow import SurfaceGroundedPBRFlow
from t3d.gltf_accessors import GLBAccessors
from scripts.eval_acceptance import (
    evaluate_asset,
    check_topology_and_manifold,
    evaluate_coarse_occupancy,
    evaluate_skeleton_rigging,
)


class TestT3DPipeline(unittest.TestCase):
    """Test suite for T3DPipeline and T3DOutput."""

    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.dim = 96
        cls.depth = 2
        cls.heads = 4
        cls.coarse_res = 4
        cls.latent_channels = 16
        cls.num_steps = 2
        cls.texture_res = 64

        # Shared lightweight pipeline for rapid unit testing
        cls.pipeline = T3DPipeline(
            frontend=MultimodalFrontEndAdapter(
                text_dim=512,
                image_dim=768,
                embed_dim=128,
                latent_channels=cls.latent_channels,
                coarse_res=cls.coarse_res,
                num_layers=2,
                num_heads=2,
            ),
            dit=UnifiedDiT(
                dim=cls.dim,
                depth=cls.depth,
                heads=cls.heads,
                latent_channels=cls.latent_channels,
                coarse_res=cls.coarse_res,
            ),
            skeleton_flow=SkeletonFlowModel(
                joint_count=5,
                cond_dim=cls.latent_channels + 1,
                hidden_dim=cls.dim,
                num_layers=2,
            ),
            skinning_module=SurfacePointToBoneSkinning(
                query_dim=cls.dim,
                key_dim=cls.dim,
                hidden_dim=cls.dim,
                num_heads=2,
            ),
            pbr_flow=SurfaceGroundedPBRFlow(
                channels=(16, 32, 64),
                latent_channels=cls.latent_channels,
                geom_channels=cls.latent_channels + 1,
                time_dim=cls.dim,
            ),
            device=cls.device,
            default_num_steps=cls.num_steps,
        )

    def test_pipeline_initialization_defaults(self):
        """Verify pipeline can be instantiated with automatic default components."""
        p = T3DPipeline(device="cpu", default_num_steps=2)
        self.assertIsInstance(p.dit, UnifiedDiT)
        self.assertIsInstance(p.frontend, MultimodalFrontEndAdapter)
        self.assertIsInstance(p.skeleton_flow, SkeletonFlowModel)
        self.assertIsInstance(p.skinning_module, SurfacePointToBoneSkinning)
        self.assertIsInstance(p.pbr_flow, SurfaceGroundedPBRFlow)
        self.assertEqual(p.default_num_steps, 2)

    def test_generate_from_text_tokens(self):
        """Verify generation from synthetic text sequence tokens."""
        B, L_text, text_dim = 1, 8, 512
        text_tokens = torch.randn(B, L_text, text_dim, device=self.device)

        output = self.pipeline.generate(
            text_tokens=text_tokens,
            num_steps=self.num_steps,
            seed=42,
            texture_res=self.texture_res,
        )

        self.assertIsInstance(output, T3DOutput)
        self.assertIsInstance(output.mesh, trimesh.Trimesh)
        self.assertTrue(output.mesh.is_watertight)
        self.assertGreaterEqual(len(output.mesh.faces), 1500)
        self.assertLessEqual(len(output.mesh.faces), 5000)

        # Check joints and skinning weights
        self.assertEqual(output.joints.shape, (5, 3))
        self.assertEqual(output.skin_weights.shape, (len(output.mesh.vertices), 5))
        np.testing.assert_allclose(output.skin_weights.sum(axis=1), 1.0, atol=1e-4)

        # Check PBR texture maps
        self.assertIn("basecolor", output.pbr_textures)
        self.assertIn("metallic_roughness", output.pbr_textures)
        self.assertIn("normal", output.pbr_textures)
        self.assertEqual(output.pbr_textures["basecolor"].shape, (self.texture_res, self.texture_res, 3))
        self.assertEqual(output.pbr_textures["metallic_roughness"].shape, (self.texture_res, self.texture_res, 2))
        self.assertEqual(output.pbr_textures["normal"].shape, (self.texture_res, self.texture_res, 3))

        # Check bounds: Basecolor linear RGB in [0, 1], luminance debaked to [0.04, 0.96]
        bc = output.pbr_textures["basecolor"]
        self.assertGreaterEqual(bc.min(), 0.0)
        self.assertLessEqual(bc.max(), 1.0)
        lum = np.sum(bc * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=-1)
        self.assertGreaterEqual(lum.min(), 0.039)
        self.assertLessEqual(lum.max(), 0.961)

    def test_generate_from_c_raw(self):
        """Verify generation directly from precomputed c_raw conditioning tensor."""
        N = self.coarse_res ** 3
        c_raw = torch.randn(1, N, self.latent_channels + 1, device=self.device)

        output = self.pipeline.generate(
            c_raw=c_raw,
            num_steps=self.num_steps,
            seed=101,
            texture_res=self.texture_res,
        )

        self.assertIsInstance(output, T3DOutput)
        self.assertTrue(output.mesh.is_watertight)
        self.assertGreaterEqual(len(output.mesh.faces), 1500)
        self.assertLessEqual(len(output.mesh.faces), 5000)
        self.assertEqual(output.joints.shape, (5, 3))

    def test_generate_from_image_tokens_and_multimodal(self):
        """Verify generation from image tokens and joint text+image tokens."""
        text_tokens = torch.randn(1, 6, 512, device=self.device)
        image_tokens = torch.randn(1, 10, 768, device=self.device)

        # Image only
        out_img = self.pipeline.generate(
            image_tokens=image_tokens,
            num_steps=self.num_steps,
            seed=202,
            texture_res=self.texture_res,
        )
        self.assertTrue(out_img.mesh.is_watertight)

        # Joint text + image
        out_joint = self.pipeline.generate(
            text_tokens=text_tokens,
            image_tokens=image_tokens,
            num_steps=self.num_steps,
            seed=303,
            texture_res=self.texture_res,
        )
        self.assertTrue(out_joint.mesh.is_watertight)
        self.assertEqual(out_joint.joints.shape, (5, 3))

    def test_deterministic_sampling_with_seed(self):
        """Verify identical seed yields identical mesh and joint predictions."""
        text_tokens = torch.randn(1, 4, 512, device=self.device)

        out_a = self.pipeline.generate(text_tokens=text_tokens, seed=777, num_steps=2, texture_res=32)
        out_b = self.pipeline.generate(text_tokens=text_tokens, seed=777, num_steps=2, texture_res=32)

        np.testing.assert_allclose(out_a.mesh.vertices, out_b.mesh.vertices, atol=1e-5)
        np.testing.assert_allclose(out_a.joints, out_b.joints, atol=1e-5)
        np.testing.assert_allclose(out_a.skin_weights, out_b.skin_weights, atol=1e-5)

    def test_export_glb_and_structure_validation(self):
        """Verify export_glb produces a valid glTF 2.0 binary with required buffers."""
        text_tokens = torch.randn(1, 4, 512, device=self.device)
        output = self.pipeline.generate(
            text_tokens=text_tokens,
            num_steps=self.num_steps,
            seed=42,
            texture_res=self.texture_res,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = self.pipeline.export_glb(output, tmp_dir, asset_name="pipeline_test")
            self.assertTrue(out_file.is_file())
            self.assertGreater(out_file.stat().st_size, 1000)

            # 1. Validate with GLBAccessors for binary buffer structure
            accessors = GLBAccessors(str(out_file))
            doc = accessors.document

            self.assertIn("meshes", doc)
            self.assertEqual(len(doc["meshes"]), 1)
            primitive = doc["meshes"][0]["primitives"][0]
            attributes = primitive["attributes"]

            # Required geometry and skinning accessors
            self.assertIn("POSITION", attributes)
            self.assertIn("NORMAL", attributes)
            self.assertIn("TEXCOORD_0", attributes)
            self.assertIn("JOINTS_0", attributes)
            self.assertIn("WEIGHTS_0", attributes)

            # Verify skin hierarchy
            self.assertIn("skins", doc)
            self.assertEqual(len(doc["skins"]), 1)
            self.assertEqual(len(doc["skins"][0]["joints"]), 5)

            # 2. Validate with trimesh.load
            loaded = trimesh.load(str(out_file), file_type="glb", force="mesh", process=False)
            self.assertIsInstance(loaded, trimesh.Trimesh)
            self.assertTrue(loaded.is_watertight)
            self.assertGreaterEqual(len(loaded.faces), 1500)
            self.assertLessEqual(len(loaded.faces), 5000)

            # Validate embedded PBR visual material attributes
            mat = loaded.visual.material
            self.assertIsInstance(mat, trimesh.visual.material.PBRMaterial)
            self.assertIsNotNone(mat.baseColorTexture)
            self.assertIsNotNone(mat.metallicRoughnessTexture)
            self.assertIsNotNone(getattr(mat, "normalTexture", None))

    def test_unconditional_generate(self):
        """Verify unconditional generation without conditioning tokens."""
        out = self.pipeline.generate(num_steps=self.num_steps, seed=12, texture_res=32)
        self.assertIsInstance(out, T3DOutput)
        self.assertTrue(out.mesh.is_watertight)
        self.assertGreaterEqual(len(out.mesh.faces), 1500)
        self.assertLessEqual(len(out.mesh.faces), 5000)

    def test_output_container_to_dict(self):
        """Verify T3DOutput container to_dict summary."""
        out = self.pipeline.generate(num_steps=1, seed=1, texture_res=32)
        summary = out.to_dict()
        self.assertTrue(summary["is_watertight"])
        self.assertEqual(summary["joint_count"], 5)
        self.assertIn("basecolor", summary["texture_keys"])

    def test_from_pretrained_checkpoint(self):
        """Verify pipeline checkpoint saving and from_pretrained loading."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            self.pipeline.save_pretrained(f.name)
            loaded_pipeline = T3DPipeline.from_pretrained(f.name, device=self.device)
            self.assertIsInstance(loaded_pipeline, T3DPipeline)
            self.assertEqual(len(loaded_pipeline.dit.blocks), len(self.pipeline.dit.blocks))
            self.assertEqual(loaded_pipeline.frontend.embed_dim, self.pipeline.frontend.embed_dim)

    def test_acceptance_gates_on_pipeline_output(self):
        """Run automated production acceptance gate harness on generated output."""
        text_tokens = torch.randn(1, 4, 512, device=self.device)
        output = self.pipeline.generate(
            text_tokens=text_tokens,
            num_steps=self.num_steps,
            seed=42,
            texture_res=self.texture_res,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = self.pipeline.export_glb(output, tmp_dir, asset_name="gate_eval_test")
            loaded = trimesh.load(str(out_file), file_type="glb", force="mesh", process=False)

            # Evaluate asset against ground truth reference
            gate_results = evaluate_asset(
                pred_mesh=loaded,
                gt_mesh=output.mesh,
                pred_joints=output.joints,
                gt_joints=output.joints,
                pred_weights=output.skin_weights,
            )

            # Gate 1: Topology & Watertightness (0 non-manifold edges, 0 self-intersections, 1500-5000 faces)
            self.assertTrue(gate_results.watertight, "Mesh is not watertight")
            self.assertEqual(gate_results.non_manifold_edges, 0, "Found non-manifold edges")
            self.assertEqual(gate_results.self_intersections, 0, "Found self-intersections")
            self.assertTrue(gate_results.triangle_count_valid, f"Face count {gate_results.triangle_count} not in [1500, 5000]")
            self.assertTrue(gate_results.topology_passed, "Topology acceptance gate failed")

            # Gate 2: Coarse Occupancy Recall (>= 0.98)
            self.assertTrue(gate_results.occupancy_passed, f"Occupancy recall {gate_results.occupancy_recall} < 0.98")

            # Gate 3: Surface Distance & F-Score (MSD <= 0.035, F-Score >= 0.70)
            self.assertTrue(gate_results.msd_passed, f"MSD {gate_results.msd} > 0.035")
            self.assertTrue(gate_results.f_score_passed, f"F-Score {gate_results.f_score} < 0.70")

            # Gate 4: Rigging & Skinning (Joint error <= 0.05, 0 containment violations, valid skinning)
            self.assertTrue(gate_results.joint_error_passed, f"Joint error {gate_results.joint_error} > 0.05")
            self.assertEqual(gate_results.containment_violations, 0, "Containment violations found")
            self.assertTrue(gate_results.skinning_buffers_valid, "Skinning buffers invalid")
            self.assertTrue(gate_results.rigging_passed, "Rigging acceptance gate failed")

            # Gate 5: All Production Acceptance Gates Passed
            self.assertTrue(gate_results.all_gates_passed, "Not all production acceptance gates passed")


if __name__ == "__main__":
    unittest.main()
