import subprocess
import sys
import unittest
from pathlib import Path


class CLITests(unittest.TestCase):
    def test_cli_help(self):
        res = subprocess.run([sys.executable, '-m', 't3d.cli', '--help'], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn('Transformer-Native 3D (.t3d) Model Format CLI', res.stdout)
        self.assertIn('generate', res.stdout)
        self.assertIn('export', res.stdout)
        self.assertIn('migrate', res.stdout)
        self.assertIn('info', res.stdout)

    def test_cli_info(self):
        sample_path = Path('artifacts/material-multiasset-32/avocado-rot0.t3d')
        if not sample_path.exists():
            self.skipTest('Sample asset missing')
        res = subprocess.run([sys.executable, '-m', 't3d.cli', 'info', str(sample_path)], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn('PBR Material: True', res.stdout)
        self.assertIn('Active Voxels: 1991', res.stdout)
