import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from t3d.checkpoints import CheckpointError, release_report, resolve_checkpoint


class CheckpointRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = self.root / 'manifest.json'
        payload = b'weights'
        digest = hashlib.sha256(payload).hexdigest()
        self.manifest.write_text(json.dumps({'checkpoints': [
            {'name': 'occupancy-dit-64-4000.pt', 'sha256': digest, 'stage': 'test'},
        ]}))
        directory = self.root / 'kaggle-kenney-64-occupancy-dit-v4'
        directory.mkdir()
        (directory / 'occupancy-dit-64-4000.pt').write_bytes(payload)

    def tearDown(self):
        self.tmp.cleanup()

    def test_resolves_role_and_verifies_manifest_hash(self):
        found = resolve_checkpoint(
            'occupancy', resolution=64, root=self.root, manifest_path=self.manifest
        )
        self.assertEqual(found.name, 'occupancy-dit-64-4000.pt')

    def test_explicit_path_rejects_hash_mismatch(self):
        bad = self.root / 'occupancy-dit-64-4000.pt'
        bad.write_bytes(b'tampered')
        with self.assertRaises(CheckpointError):
            resolve_checkpoint(
                'occupancy', path=bad, manifest_path=self.manifest
            )

    def test_explicit_unlisted_path_is_accepted(self):
        custom = self.root / 'custom.pt'
        custom.write_bytes(b'custom')
        found = resolve_checkpoint('geometry', path=custom, manifest_path=self.manifest)
        self.assertEqual(found, custom)

    def test_missing_role_raises_actionable_error(self):
        with self.assertRaises(CheckpointError) as ctx:
            resolve_checkpoint('rig_flow', root=self.root, manifest_path=self.manifest)
        self.assertIn('Set T3D_ARTIFACTS', str(ctx.exception))

    def test_release_report_counts_available_roles(self):
        report = release_report(root=self.root, manifest_path=self.manifest)
        self.assertTrue(report['roles']['occupancy_64']['present'])
        self.assertFalse(report['roles']['geometry_64']['present'])
        self.assertEqual(report['manifest_entries'], 1)


if __name__ == '__main__':
    unittest.main()
