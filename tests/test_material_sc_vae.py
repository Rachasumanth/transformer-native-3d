"""Unit tests verifying Multi-Asset Material SC-VAE checkpoint integrity and inference."""
import hashlib
from pathlib import Path
import unittest
import torch
from t3d.dataset import AssetDataset
from t3d.sc_vae import SparseCompressionVAE


class MaterialSCVAETests(unittest.TestCase):
    def setUp(self):
        self.ckpt_path = Path('artifacts/kaggle-material-multiasset-32-v1/material-multiasset-300.pt')
        self.manifest_path = Path('artifacts/material-multiasset-32/manifest.json')

    def test_checkpoint_sha256_integrity(self):
        if not self.ckpt_path.exists():
            self.skipTest(f"Checkpoint {self.ckpt_path} not found")
        digest = hashlib.sha256(self.ckpt_path.read_bytes()).hexdigest()
        self.assertEqual(digest, 'f830a9ebc58ccb0977ca20ae468d906e000caf72c9e7c954f50462240b76b6a6')

    def test_material_vae_inference_and_pbr_bounds(self):
        if not self.ckpt_path.exists() or not self.manifest_path.exists():
            self.skipTest("Missing checkpoint or material dataset")

        ckpt = torch.load(self.ckpt_path, map_location='cpu', weights_only=False)
        cfg = dict(
            mode='material',
            channels=ckpt['config']['channels'],
            latent_channels=ckpt['config']['latent_channels'],
            convolution_backend='reference'
        )
        model = SparseCompressionVAE(**cfg).eval()
        model.load_state_dict(ckpt['model'])

        val_ds = AssetDataset(self.manifest_path, split='validation', require_pbr=True)
        self.assertGreater(len(val_ds), 0)
        asset = val_ds[0]

        from t3d.dataset import batch_arrays
        coords, features = batch_arrays([asset], 'material')
        c_t = torch.from_numpy(coords)
        f_t = torch.from_numpy(features)

        with torch.no_grad():
            posterior = model.encode(c_t, f_t)
            self.assertEqual(posterior.coords.shape[1], 4)  # (B, X, Y, Z)
            self.assertEqual(posterior.mean.shape[1], 16)   # 16-channel material latent

            out_coords, raw, _ = model.decode(posterior.coords, posterior.mean, teacher_occupancy=posterior.occupancy)
            sorted_coords, inverse = torch.unique(c_t, dim=0, sorted=True, return_inverse=True)
            self.assertTrue(torch.equal(out_coords, sorted_coords))

            pred_pbr = raw.sigmoid()
            self.assertEqual(pred_pbr.shape, f_t.shape)
            self.assertTrue((pred_pbr >= 0.0).all())
            self.assertTrue((pred_pbr <= 1.0).all())

            # Evaluate MAE on this validation asset
            target = f_t[torch.argsort(inverse)]
            mae = (pred_pbr - target).abs().mean().item()
            self.assertLess(mae, 0.25)


if __name__ == '__main__':
    unittest.main()
