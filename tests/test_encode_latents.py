"""Unit tests for SC-VAE latent & Hermite target pre-encoder."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import trimesh

from t3d.sc_vae import SparseCompressionVAE
from t3d.data.refinement.encode_latents import (
    encode_paired_asset,
    encode_micro_batch,
    extract_voxel_hermite_normals,
    synthesize_corrupted_mesh,
    load_mesh_or_asset,
)


class TestEncodeLatents(unittest.TestCase):
    def setUp(self):
        # Create a canonical test box mesh
        box = trimesh.creation.box(extents=(0.6, 0.6, 0.6))
        self.clean_mesh = (np.asarray(box.vertices, dtype=np.float64), np.asarray(box.faces, dtype=np.int64))
        raw_v, raw_f = synthesize_corrupted_mesh(self.clean_mesh[0], self.clean_mesh[1], noise_std=0.04)
        self.raw_mesh = (raw_v, raw_f)

    def test_shapes_and_tensor_types_torch(self):
        """Verify return dictionary keys, torch tensor types, and exact shapes."""
        res = encode_paired_asset(
            self.clean_mesh,
            self.raw_mesh,
            coarse_res=4,
            fine_res=32,
            return_torch=True,
        )

        expected_keys = {'x_0_occ', 'x_0_lat', 'c_raw', 'hermite_normals', 'surface_ratio'}
        self.assertEqual(set(res.keys()), expected_keys)

        for k, v in res.items():
            self.assertIsInstance(v, torch.Tensor, f"{k} is not a torch.Tensor")
            self.assertEqual(v.dtype, torch.float32, f"{k} dtype is not torch.float32")

        # N = 4^3 = 64 coarse cells, default C = 16 latent channels
        self.assertEqual(res['x_0_occ'].shape, torch.Size([64]))
        self.assertEqual(res['x_0_lat'].shape, torch.Size([64, 16]))
        self.assertEqual(res['c_raw'].shape, torch.Size([64, 17]))
        self.assertEqual(res['hermite_normals'].shape, torch.Size([64, 3]))
        self.assertEqual(res['surface_ratio'].shape, torch.Size([64]))

    def test_shapes_and_types_numpy(self):
        """Verify numpy array return mode and shapes."""
        res = encode_paired_asset(
            self.clean_mesh,
            self.raw_mesh,
            coarse_res=4,
            fine_res=32,
            return_torch=False,
        )

        for k, v in res.items():
            self.assertIsInstance(v, np.ndarray, f"{k} is not a np.ndarray")
            self.assertEqual(v.dtype, np.float32, f"{k} dtype is not np.float32")

        self.assertEqual(res['x_0_occ'].shape, (64,))
        self.assertEqual(res['x_0_lat'].shape, (64, 16))
        self.assertEqual(res['c_raw'].shape, (64, 17))
        self.assertEqual(res['hermite_normals'].shape, (64, 3))
        self.assertEqual(res['surface_ratio'].shape, (64,))

    def test_surface_ratio_range(self):
        """Verify surface_ratio is strictly bounded in [0, 1] and positive on active cells."""
        res = encode_paired_asset(
            self.clean_mesh,
            self.raw_mesh,
            coarse_res=4,
            fine_res=32,
            return_torch=True,
        )
        ratio = res['surface_ratio']
        occ = res['x_0_occ']

        self.assertTrue((ratio >= 0.0).all())
        self.assertTrue((ratio <= 1.0).all())

        # Active cells must have positive surface ratio
        active_mask = (occ > 0.0)
        self.assertTrue(active_mask.any(), "Expected at least one active cell")
        self.assertTrue((ratio[active_mask] > 0.0).all())

        # Void cells must have 0.0 surface ratio
        void_mask = (occ < 0.0)
        self.assertTrue(void_mask.any(), "Expected at least one void cell")
        self.assertTrue((ratio[void_mask] == 0.0).all())

    def test_zero_latent_masking_on_void_cells(self):
        """Verify zero-latent masking on void cells for both x_0_lat and hermite_normals."""
        res = encode_paired_asset(
            self.clean_mesh,
            self.raw_mesh,
            coarse_res=4,
            fine_res=32,
            return_torch=True,
        )
        occ = res['x_0_occ']
        lat = res['x_0_lat']
        normals = res['hermite_normals']

        void_indices = torch.where(occ < 0.0)[0]
        active_indices = torch.where(occ > 0.0)[0]

        self.assertGreater(len(void_indices), 0, "No void cells found")
        self.assertGreater(len(active_indices), 0, "No active cells found")

        # Void cells must have exactly zero latents and zero normals
        self.assertTrue(
            torch.all(lat[void_indices] == 0.0),
            "Non-zero latents detected on void cells"
        )
        self.assertTrue(
            torch.all(normals[void_indices] == 0.0),
            "Non-zero normals detected on void cells"
        )

        # Active cells must have non-zero latents and unit normals
        lat_norms = torch.norm(lat[active_indices], dim=-1)
        self.assertTrue((lat_norms > 0.0).all(), "Zero latents found on active cells")

        normal_norms = torch.norm(normals[active_indices], dim=-1)
        torch.testing.assert_close(
            normal_norms,
            torch.ones_like(normal_norms),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_c_raw_conditioning_structure(self):
        """Verify corrupted conditioning latents c_raw channel layout [occ, latents]."""
        res = encode_paired_asset(
            self.clean_mesh,
            self.raw_mesh,
            coarse_res=4,
            fine_res=32,
            return_torch=True,
        )
        c_raw = res['c_raw']

        # Channel 0 is coarse occupancy in {-1.0, 1.0}
        raw_occ = c_raw[:, 0]
        unique_occ = torch.unique(raw_occ)
        for val in unique_occ:
            self.assertIn(val.item(), {-1.0, 1.0})

        # Remaining channels are continuous latents with zero on void cells
        raw_lat = c_raw[:, 1:]
        raw_void = torch.where(raw_occ < 0.0)[0]
        if len(raw_void) > 0:
            self.assertTrue(torch.all(raw_lat[raw_void] == 0.0))

    def test_custom_vae_latent_dimension(self):
        """Verify custom SC-VAE model with 4 latent channels."""
        custom_vae = SparseCompressionVAE(
            mode='shape',
            channels=(2, 4, 8, 16, 32),
            latent_channels=4,
            convolution_backend='reference',
        ).eval()

        res = encode_paired_asset(
            self.clean_mesh,
            self.raw_mesh,
            sc_vae_model=custom_vae,
            coarse_res=4,
            fine_res=32,
            return_torch=True,
        )
        self.assertEqual(res['x_0_lat'].shape, torch.Size([64, 4]))
        self.assertEqual(res['c_raw'].shape, torch.Size([64, 5]))

    def test_coarse_resolution_scaling(self):
        """Verify encoding with a different coarse resolution (e.g. coarse_res=2)."""
        res = encode_paired_asset(
            self.clean_mesh,
            self.raw_mesh,
            coarse_res=2,
            fine_res=32,
            return_torch=True,
        )
        N = 2 ** 3  # 8
        self.assertEqual(res['x_0_occ'].shape, torch.Size([N]))
        self.assertEqual(res['x_0_lat'].shape, torch.Size([N, 16]))
        self.assertEqual(res['c_raw'].shape, torch.Size([N, 17]))
        self.assertEqual(res['hermite_normals'].shape, torch.Size([N, 3]))
        self.assertEqual(res['surface_ratio'].shape, torch.Size([N]))

    def test_single_asset_npz_output(self):
        """Verify output_npz_path parameter saves a valid archive."""
        with tempfile.NamedTemporaryFile(suffix='.npz') as tmp:
            res = encode_paired_asset(
                self.clean_mesh,
                self.raw_mesh,
                coarse_res=4,
                fine_res=32,
                output_npz_path=tmp.name,
                return_torch=False,
            )
            loaded = np.load(tmp.name)
            for k in ('x_0_occ', 'x_0_lat', 'c_raw', 'hermite_normals', 'surface_ratio'):
                self.assertIn(k, loaded.files)
                np.testing.assert_allclose(loaded[k], res[k])

    def test_encode_micro_batch(self):
        """Verify micro-batch encoding utility stacks assets along batch dimension."""
        box2 = trimesh.creation.box(extents=(0.4, 0.5, 0.4))
        mesh2 = (np.asarray(box2.vertices, dtype=np.float64), np.asarray(box2.faces, dtype=np.int64))

        pairs = [
            (self.clean_mesh, self.raw_mesh),
            (mesh2, None),  # synthesized raw
        ]

        with tempfile.NamedTemporaryFile(suffix='.npz') as tmp:
            batch = encode_micro_batch(
                pairs,
                output_npz_path=tmp.name,
                coarse_res=4,
                fine_res=32,
                return_torch=True,
            )

            self.assertEqual(batch['x_0_occ'].shape, torch.Size([2, 64]))
            self.assertEqual(batch['x_0_lat'].shape, torch.Size([2, 64, 16]))
            self.assertEqual(batch['c_raw'].shape, torch.Size([2, 64, 17]))
            self.assertEqual(batch['hermite_normals'].shape, torch.Size([2, 64, 3]))
            self.assertEqual(batch['surface_ratio'].shape, torch.Size([2, 64]))

            # Verify saved npz file
            loaded = np.load(tmp.name)
            self.assertEqual(loaded['x_0_occ'].shape, (2, 64))
            self.assertEqual(loaded['x_0_lat'].shape, (2, 64, 16))
            self.assertEqual(loaded['c_raw'].shape, (2, 64, 17))
            self.assertEqual(loaded['hermite_normals'].shape, (2, 64, 3))
            self.assertEqual(loaded['surface_ratio'].shape, (2, 64))


if __name__ == '__main__':
    unittest.main()
