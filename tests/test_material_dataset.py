import json
from pathlib import Path
import unittest
import numpy as np
from t3d.dataset import AssetDataset


class MaterialDatasetTests(unittest.TestCase):
    def test_multiasset_material_dataset(self):
        manifest_path = Path('artifacts/material-multiasset-32/manifest.json')
        if not manifest_path.exists():
            self.skipTest('Material multiasset dataset not built')

        train_ds = AssetDataset(manifest_path, split='train')
        val_ds = AssetDataset(manifest_path, split='validation')

        self.assertGreater(len(train_ds), 0)
        self.assertGreater(len(val_ds), 0)

        # Source pack/group separation check: strictly disjoint
        train_groups = {r.source_group for r in train_ds.records}
        val_groups = {r.source_group for r in val_ds.records}
        self.assertTrue(train_groups.isdisjoint(val_groups), f"Overlap between splits: {train_groups & val_groups}")

        # Check unique asset IDs and SHA256 hashes across entire manifest
        all_records = list(train_ds.records) + list(val_ds.records)
        ids = [r.asset_id for r in all_records]
        hashes = [r.sha256 for r in all_records]
        self.assertEqual(len(ids), len(set(ids)), 'Duplicate asset IDs found')
        self.assertEqual(len(hashes), len(set(hashes)), 'Duplicate content SHA256 found')

        # Test first train asset loading and PBR integrity
        first_asset = train_ds[0]
        self.assertIsNotNone(first_asset.pbr)
        self.assertEqual(first_asset.pbr.shape, (len(first_asset.coords), 6))
        # PBR attributes: RGB [0, 1], Metallic [0, 1], Roughness [0, 1], Alpha [0, 1]
        self.assertTrue(np.all(first_asset.pbr >= -1e-6))
        self.assertTrue(np.all(first_asset.pbr <= 1.0 + 1e-6))

        # Test report
        rep_path = manifest_path.parent / 'conversion-report.json'
        if rep_path.exists():
            rep = json.loads(rep_path.read_text())
            self.assertIn('duplicates_removed', rep)
            self.assertGreaterEqual(rep['duplicates_removed'], 2)
