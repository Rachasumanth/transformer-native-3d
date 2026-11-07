"""Unit tests for single-atlas UV unwrapping and PBR texture baking pipeline."""

import unittest
import numpy as np
from PIL import Image
import trimesh

from t3d.data.refinement.canonicalize_pbr import (
    unwrap_single_atlas,
    debake_luminance,
    pad_texture,
    compute_tangent_frames,
    rasterize_uv_atlas,
    bake_single_atlas_pbr,
    LUMINANCE_WEIGHTS,
)
from t3d.import_glb import ImportedMesh
from t3d.materials import PBRMaterial


class TestCanonicalizePBR(unittest.TestCase):
    """Test suite for xatlas unwrapping, BVH raycast/closest-point baking, and luminance debaking."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_unwrap_single_atlas_bounds_and_validity(self):
        """Test that xatlas generates guaranteed non-overlapping UVs bounded in [0, 1]^2."""
        box = trimesh.creation.box()
        res = unwrap_single_atlas(box, texture_res=512, padding=4)

        # Check required fields
        self.assertIn("uv", res)
        self.assertIn("indices", res)
        self.assertIn("vmapping", res)
        self.assertIn("vertices", res)
        self.assertIn("faces", res)
        self.assertIn("corner_uv", res)
        self.assertIn("chart_count", res)
        self.assertIn("mesh", res)

        uv = res["uv"]
        # UV bounds must strictly be in [0, 1]^2
        self.assertGreaterEqual(float(uv.min()), 0.0)
        self.assertLessEqual(float(uv.max()), 1.0)
        self.assertTrue((uv >= 0.0).all())
        self.assertTrue((uv <= 1.0).all())

        # Face count must match original mesh face count
        self.assertEqual(len(res["faces"]), len(box.faces))
        self.assertGreater(res["chart_count"], 0)

        # Reconstructed 3D geometry from unwrapped faces must match original geometry
        vmapping = res["vmapping"]
        indices = res["indices"]
        reconstructed_triangles = box.vertices[vmapping[indices]]
        original_triangles = box.vertices[box.faces]
        np.testing.assert_allclose(reconstructed_triangles, original_triangles, atol=1e-6)

        # Also test on curved geometry (icosphere)
        sphere = trimesh.creation.icosphere(subdivisions=2)
        sphere_res = unwrap_single_atlas(sphere, texture_res=512, padding=4)
        sphere_uv = sphere_res["uv"]
        self.assertGreaterEqual(float(sphere_uv.min()), 0.0)
        self.assertLessEqual(float(sphere_uv.max()), 1.0)
        self.assertEqual(len(sphere_res["faces"]), len(sphere.faces))

    def test_unwrap_single_atlas_errors(self):
        """Test error handling for invalid mesh representations and parameters."""
        # Empty mesh
        with self.assertRaises(ValueError):
            unwrap_single_atlas(trimesh.Trimesh(vertices=[], faces=[]))

        # Invalid resolution
        box = trimesh.creation.box()
        with self.assertRaises(ValueError):
            unwrap_single_atlas(box, texture_res=8)
        with self.assertRaises(ValueError):
            unwrap_single_atlas(box, texture_res="invalid")

        # Invalid padding
        with self.assertRaises(ValueError):
            unwrap_single_atlas(box, padding=-1)

        # Non-finite vertices
        bad_v = box.vertices.copy()
        bad_v[0, 0] = np.nan
        with self.assertRaises(ValueError):
            unwrap_single_atlas((bad_v, box.faces))

    def test_debake_luminance(self):
        """Validate luminance clamping to [0.04, 0.96] to strip shadows and specular blowouts."""
        # Test extreme shadows and pitch black
        black = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        debaked_black = debake_luminance(black, min_lum=0.04, max_lum=0.96)
        lum_black = float(np.sum(debaked_black * LUMINANCE_WEIGHTS))
        self.assertAlmostEqual(lum_black, 0.04, places=4)

        deep_shadow = np.array([[0.01, 0.01, 0.01]], dtype=np.float32)
        debaked_shadow = debake_luminance(deep_shadow, min_lum=0.04, max_lum=0.96)
        lum_shadow = float(np.sum(debaked_shadow * LUMINANCE_WEIGHTS))
        self.assertAlmostEqual(lum_shadow, 0.04, places=4)

        # Test extreme specular blowouts and pure white
        white = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        debaked_white = debake_luminance(white, min_lum=0.04, max_lum=0.96)
        lum_white = float(np.sum(debaked_white * LUMINANCE_WEIGHTS))
        self.assertAlmostEqual(lum_white, 0.96, places=4)

        specular = np.array([[0.99, 0.98, 0.97]], dtype=np.float32)
        debaked_specular = debake_luminance(specular, min_lum=0.04, max_lum=0.96)
        lum_specular = float(np.sum(debaked_specular * LUMINANCE_WEIGHTS))
        self.assertAlmostEqual(lum_specular, 0.96, places=4)

        # Test neutral mid-range color (unchanged)
        mid = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        debaked_mid = debake_luminance(mid, min_lum=0.04, max_lum=0.96)
        np.testing.assert_allclose(debaked_mid, mid, atol=1e-5)

        # Test chromaticity preservation
        colored = np.array([[0.0, 0.02, 0.0]], dtype=np.float32)
        debaked_col = debake_luminance(colored, min_lum=0.04, max_lum=0.96)
        lum_col = float(np.sum(debaked_col * LUMINANCE_WEIGHTS))
        self.assertAlmostEqual(lum_col, 0.04, places=4)
        # R and B channels should remain 0
        self.assertEqual(debaked_col[0, 0], 0.0)
        self.assertEqual(debaked_col[0, 2], 0.0)
        self.assertGreater(debaked_col[0, 1], 0.0)

        # Batch validation across 1,000 random colors
        random_colors = self.rng.uniform(0.0, 1.0, (1000, 3)).astype(np.float32)
        debaked_batch = debake_luminance(random_colors, min_lum=0.04, max_lum=0.96)
        batch_lums = np.sum(debaked_batch * LUMINANCE_WEIGHTS, axis=-1)
        self.assertTrue((batch_lums >= 0.04 - 1e-5).all())
        self.assertTrue((batch_lums <= 0.96 + 1e-5).all())

    def test_pad_texture(self):
        """Test nearest-neighbor atlas dilation preserving covered texels."""
        H, W = 32, 32
        img = np.zeros((H, W, 3), dtype=np.float32)
        mask = np.zeros((H, W), dtype=bool)

        # Fill central square with known color
        img[10:22, 10:22] = [0.8, 0.4, 0.2]
        mask[10:22, 10:22] = True

        padded = pad_texture(img, mask)

        # Covered texels must be identical
        np.testing.assert_allclose(padded[mask], img[mask])

        # Previously uncovered corners must now be filled with the border color
        self.assertTrue((padded[0, 0] > 0).all())
        np.testing.assert_allclose(padded[0, 0], [0.8, 0.4, 0.2])

    def test_compute_tangent_frames(self):
        """Test orthonormal tangent frame construction on a quad."""
        # Simple XY plane quad
        v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        uvs = np.array([
            [[0, 0], [1, 0], [1, 1]],
            [[0, 0], [1, 1], [0, 1]],
        ], dtype=np.float32)

        T, B, N = compute_tangent_frames(v, f, uvs)

        # Both triangles face +Z
        expected_N = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)
        np.testing.assert_allclose(N, expected_N, atol=1e-5)

        # U aligns with +X, V aligns with +Y
        expected_T = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)
        expected_B = np.array([[0, 1, 0], [0, 1, 0]], dtype=np.float32)
        np.testing.assert_allclose(T, expected_T, atol=1e-5)
        np.testing.assert_allclose(B, expected_B, atol=1e-5)

    def test_bake_single_atlas_pbr_vertex_colors(self):
        """Test baking pipeline on a mesh with multi-colored vertices."""
        raw = trimesh.creation.box()
        vc = np.array([
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255, 255, 0, 255],
            [255, 0, 255, 255],
            [0, 255, 255, 255],
            [128, 128, 128, 255],
            [200, 200, 200, 255],
        ], dtype=np.uint8)
        raw.visual = trimesh.visual.ColorVisuals(vertex_colors=vc)

        clean = raw.copy()

        result = bake_single_atlas_pbr(clean, raw, texture_res=128, padding=4)

        # Validate returned keys
        self.assertIn("uv", result)
        self.assertIn("indices", result)
        self.assertIn("vmapping", result)
        self.assertIn("basecolor", result)
        self.assertIn("base_color", result)
        self.assertIn("metallic_roughness", result)
        self.assertIn("metallic_roughness_3ch", result)
        self.assertIn("normal", result)
        self.assertIn("basecolor_image", result)
        self.assertIn("metallic_roughness_image", result)
        self.assertIn("normal_image", result)
        self.assertIn("valid_mask", result)
        self.assertIn("mesh", result)
        self.assertIn("atlas_info", result)

        # Validate map dimensions and types
        self.assertEqual(result["basecolor"].shape, (128, 128, 3))
        self.assertEqual(result["metallic_roughness"].shape, (128, 128, 2))
        self.assertEqual(result["metallic_roughness_3ch"].shape, (128, 128, 3))
        self.assertEqual(result["normal"].shape, (128, 128, 3))

        # Validate PIL images
        self.assertIsInstance(result["basecolor_image"], Image.Image)
        self.assertEqual(result["basecolor_image"].size, (128, 128))
        self.assertEqual(result["basecolor_image"].mode, "RGB")

        self.assertIsInstance(result["metallic_roughness_image"], Image.Image)
        self.assertEqual(result["metallic_roughness_image"].size, (128, 128))

        self.assertIsInstance(result["normal_image"], Image.Image)
        self.assertEqual(result["normal_image"].size, (128, 128))

        # Validate valid mask coverage
        valid_mask = result["valid_mask"]
        self.assertGreater(valid_mask.sum(), 0)

        # Validate luminance clamping on covered texels
        covered_bc = result["basecolor"][valid_mask]
        lums = np.sum(covered_bc * LUMINANCE_WEIGHTS, axis=-1)
        self.assertTrue((lums >= 0.04 - 1e-4).all())
        self.assertTrue((lums <= 0.96 + 1e-4).all())

        # Validate tangent-space normal map is near neutral [0.5, 0.5, 1.0] for planar faces
        covered_normals = result["normal"][valid_mask]
        # In tangent space with identical geometry, Z component should dominate (near 1.0)
        self.assertGreater(covered_normals[:, 2].mean(), 0.9)

    def test_bake_single_atlas_pbr_textured_material(self):
        """Test baking from a textured PBRMaterial source mesh."""
        tex_img = Image.new("RGB", (64, 64), (180, 90, 40))
        mat = trimesh.visual.material.PBRMaterial(
            baseColorTexture=tex_img,
            metallicFactor=0.35,
            roughnessFactor=0.65,
        )

        raw = trimesh.creation.box()
        uvs = np.zeros((len(raw.vertices), 2), dtype=np.float32)
        uvs[:, 0] = raw.vertices[:, 0] + 0.5
        uvs[:, 1] = raw.vertices[:, 1] + 0.5
        raw.visual = trimesh.visual.TextureVisuals(uv=uvs, material=mat)

        clean = raw.copy()

        result = bake_single_atlas_pbr(clean, raw, texture_res=128, padding=4)

        valid_mask = result["valid_mask"]
        mr = result["metallic_roughness"][valid_mask]

        # Channel 0 is roughness, Channel 1 is metallic
        np.testing.assert_allclose(mr[:, 0], 0.65, atol=1e-2)
        np.testing.assert_allclose(mr[:, 1], 0.35, atol=1e-2)

        # Base color linear RGB values reflect linear conversion of sRGB texture
        bc = result["basecolor"][valid_mask]
        srgb_r = 180.0 / 255.0
        srgb_g = 90.0 / 255.0
        srgb_b = 40.0 / 255.0
        expected_r_lin = ((srgb_r + 0.055) / 1.055) ** 2.4
        expected_g_lin = ((srgb_g + 0.055) / 1.055) ** 2.4
        expected_b_lin = ((srgb_b + 0.055) / 1.055) ** 2.4

        self.assertAlmostEqual(bc[:, 0].mean(), expected_r_lin, delta=0.05)
        self.assertAlmostEqual(bc[:, 1].mean(), expected_g_lin, delta=0.05)
        self.assertAlmostEqual(bc[:, 2].mean(), expected_b_lin, delta=0.05)

        # Base color image (sRGB) reflects original texture byte values
        bc_img_arr = np.array(result["basecolor_image"])[valid_mask]
        self.assertAlmostEqual(bc_img_arr[:, 0].mean(), 180.0, delta=10.0)
        self.assertAlmostEqual(bc_img_arr[:, 1].mean(), 90.0, delta=10.0)
        self.assertAlmostEqual(bc_img_arr[:, 2].mean(), 40.0, delta=10.0)

    def test_bake_single_atlas_pbr_object_space_normals(self):
        """Test baking normal map in object-space mode."""
        raw = trimesh.creation.box()
        clean = raw.copy()

        result = bake_single_atlas_pbr(
            clean,
            raw,
            texture_res=128,
            padding=4,
            normal_space="object",
        )

        valid_mask = result["valid_mask"]
        normals = result["normal"][valid_mask]

        # In object space, normals are encoded as (N + 1) / 2
        # Cube face normals are ±X, ±Y, ±Z, so encoded values are 0.0, 0.5, or 1.0
        for val in normals.ravel():
            self.assertTrue(
                np.isclose(val, 0.0, atol=0.05)
                or np.isclose(val, 0.5, atol=0.05)
                or np.isclose(val, 1.0, atol=0.05)
            )

    def test_bake_single_atlas_pbr_imported_mesh(self):
        """Test baking multi-material ImportedMesh object."""
        box = trimesh.creation.box()
        mat1 = PBRMaterial(base_color=(0.8, 0.2, 0.2, 1.0), metallic=0.1, roughness=0.9)
        mat2 = PBRMaterial(base_color=(0.2, 0.8, 0.2, 1.0), metallic=0.8, roughness=0.2)

        face_mats = np.zeros(len(box.faces), dtype=int)
        face_mats[len(box.faces) // 2:] = 1

        corner_uv = np.zeros((len(box.faces), 3, 2), dtype=np.float32)

        imported = ImportedMesh(
            vertices=box.vertices,
            faces=box.faces,
            corner_uv=corner_uv,
            materials=[mat1, mat2],
            face_materials=face_mats,
            source_sha256="test_sha",
        )

        result = bake_single_atlas_pbr(box, imported, texture_res=128, padding=4)

        valid_mask = result["valid_mask"]
        self.assertGreater(valid_mask.sum(), 0)

        # Check that both materials' roughness values are represented
        roughness_vals = result["metallic_roughness"][valid_mask, 0]
        self.assertTrue((roughness_vals > 0.85).any())  # mat1 roughness = 0.9
        self.assertTrue((roughness_vals < 0.25).any())  # mat2 roughness = 0.2

    def test_bake_single_atlas_pbr_1024_resolution(self):
        """Test baking at full 1024x1024 production resolution with 4px chart padding."""
        box = trimesh.creation.box()
        box.visual = trimesh.visual.ColorVisuals(
            vertex_colors=self.rng.integers(50, 200, (len(box.vertices), 4), dtype=np.uint8)
        )

        result = bake_single_atlas_pbr(box, box, texture_res=1024, padding=4)

        self.assertEqual(result["basecolor"].shape, (1024, 1024, 3))
        self.assertEqual(result["metallic_roughness"].shape, (1024, 1024, 2))
        self.assertEqual(result["normal"].shape, (1024, 1024, 3))
        self.assertEqual(result["basecolor_image"].size, (1024, 1024))

        # Check atlas metadata
        info = result["atlas_info"]
        self.assertEqual(info["total_texels"], 1024 * 1024)
        self.assertGreater(info["covered_texels"], 0)
        self.assertGreater(info["coverage_ratio"], 0.3)


if __name__ == "__main__":
    unittest.main()
