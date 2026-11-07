import unittest
from dataclasses import replace
import numpy as np
from t3d.schema import VoxelAsset
from t3d.export import export_raw


class ExportContractTests(unittest.TestCase):
    def setUp(self):
        self.asset=VoxelAsset([[0,0,0]],[[.5]*3],np.zeros((1,3),bool),
                              (2,)*3,[[0]*3,[1]*3],pbr=[[.2,.3,.4,0,.8,1]])

    def test_missing_material_and_incomplete_rig_rejected_before_gpu_allocation(self):
        with self.assertRaisesRegex(ValueError,'PBR'):
            export_raw(replace(self.asset,pbr=None))
        with self.assertRaisesRegex(ValueError,'skeleton_field'):
            export_raw(replace(self.asset,skin_embedding=np.zeros((1,4))))
        with self.assertRaisesRegex(ValueError,'SkinAE checkpoint'):
            export_raw(self.asset,skinae_checkpoint='/tmp/fake.pt')
        # Assets with skeleton_field proceed to GPU check without requiring skinae_checkpoint
        rigged=replace(self.asset,skeleton_field=np.zeros((1,6)))
        with self.assertRaisesRegex(ValueError,'CUDA export device required'):
            export_raw(rigged)

    def test_invalid_options_rejected_before_gpu_allocation(self):
        for options in (dict(source_up='x'),dict(alpha_mode='unknown'),dict(target_faces=0),
                        dict(texture_size=0),dict(alpha_cutoff=float('nan')),dict(remesh=1)):
            with self.assertRaises(ValueError):export_raw(self.asset,**options)
