import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from t3d.container import dumps
from t3d.schema import VoxelAsset
from t3d.dataset import AssetDataset, batch_arrays


def fixture():
    return VoxelAsset([[0,0,0]], [[.5]*3], np.ones((1,3),bool), (2,)*3, [[0]*3,[1]*3], pbr=[[.3]*6])


class DatasetTests(unittest.TestCase):
    def test_manifest_hash_and_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=dumps(fixture()); (root/'a.t3d').write_bytes(data)
            record=dict(asset_id='a', path='a.t3d', sha256=hashlib.sha256(data).hexdigest(), split='train', source='synthetic test fixture', license='test fixture')
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps(dict(version=1,assets=[record])))
            dataset=AssetDataset(manifest,split='train',require_pbr=True)
            c,f=batch_arrays([dataset[0],dataset[0]],'material')
            np.testing.assert_array_equal(c[:,0],[0,1]); self.assertEqual(f.shape,(2,6))
            (root/'a.t3d').write_bytes(data+b'changed')
            with self.assertRaises(ValueError): dataset[0]
            for records in ([record,dict(record,asset_id='b',split='validation')], [dict(record,path='../outside')]):
                manifest.write_text(json.dumps(dict(version=1,assets=records)))
                with self.assertRaises(ValueError): AssetDataset(manifest,split='train')

    def test_source_identity_does_not_cross_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=Path(tmp)/'manifest.json'
            first=dict(asset_id='a',path='a.t3d',sha256='a'*64,split='train',
                       source='source',license='test',source_sha256='c'*64,source_group='family')
            second=dict(first,asset_id='b',path='b.t3d',sha256='b'*64)
            def check(records):
                manifest.write_text(json.dumps(dict(version=1,assets=records)))
                return AssetDataset(manifest,split='train')
            self.assertEqual(len(check([first,second])),2)
            for changed in (dict(second,split='validation'),
                            dict(second,split='validation',source_sha256='d'*64),
                            dict(second,split='validation',source_group='other')):
                with self.assertRaisesRegex(ValueError,'crosses dataset splits'):
                    check([first,changed])
            with self.assertRaisesRegex(ValueError,'Invalid source content hash'):
                check([dict(first,source_sha256='bad')])
            with self.assertRaisesRegex(ValueError,'Invalid source group'):
                check([dict(first,source_group=' ')])

    def test_in_distribution_holdout_allows_group_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / 'manifest.json'
            first = dict(asset_id='a', path='a.t3d', sha256='a' * 64, split='train',
                         source='source', license='test', source_sha256='c' * 64, source_group='family')
            second = dict(first, asset_id='b', path='b.t3d', sha256='b' * 64, split='validation')
            manifest.write_text(json.dumps(dict(version=1, split_policy='in_distribution_holdout', assets=[first, second])))
            train = AssetDataset(manifest, split='train')
            validation = AssetDataset(manifest, split='validation')
            self.assertEqual((len(train), len(validation), train.split_policy), (1, 1, 'in_distribution_holdout'))
            manifest.write_text(json.dumps(dict(version=1, split_policy='unknown', assets=[first])))
            with self.assertRaisesRegex(ValueError, 'Unsupported split policy'):
                AssetDataset(manifest, split='train')
