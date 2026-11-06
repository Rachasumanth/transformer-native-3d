import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from tools.build_cuda_dependency import compatible_wheel


class WheelCacheTests(unittest.TestCase):
    def fixture(self,root):
        folder=Path(root)/'CuMesh';folder.mkdir()
        wheel=folder/'cumesh-cp313.whl';wheel.write_bytes(b'nonexecuted test payload')
        runtime=dict(python='3.13.1',torch='2.11.0+cu128',cuda='12.8',cxx11_abi=True,architectures='7.5')
        receipt=dict(name='CuMesh',url='upstream',commit='revision',wheel=wheel.name,
                     sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),**runtime)
        (folder/'receipt.json').write_text(json.dumps(receipt))
        return wheel,runtime,receipt

    def test_runtime_source_and_hash_gates(self):
        with tempfile.TemporaryDirectory() as root:
            wheel,runtime,receipt=self.fixture(root)
            query=lambda **kw:compatible_wheel(root,'CuMesh','upstream','revision',dict(runtime,**kw))
            self.assertEqual(query()[0],wheel)
            self.assertEqual(query(python='3.13.2')[0],wheel)
            flat=Path(root)/wheel.name
            flat.write_bytes(wheel.read_bytes())
            (Path(root)/'CuMesh.receipt.json').write_text(json.dumps(receipt))
            (wheel.parent/'receipt.json').unlink()
            self.assertEqual(query()[0],flat)
            (wheel.parent/'receipt.json').write_text(json.dumps(receipt))
            for changes in (dict(python='3.14.0'),dict(torch='different'),dict(cuda='13.0'),
                            dict(cxx11_abi=False),dict(architectures='8.0')):
                self.assertIsNone(query(**changes))
            self.assertIsNone(compatible_wheel(root,'CuMesh','upstream','other-revision',runtime))
            wheel.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'checksum'):query()

    def test_missing_cache_and_invalid_filename(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(compatible_wheel(root,'CuMesh','upstream','revision',{}))
            wheel,runtime,receipt=self.fixture(root)
            receipt['wheel']='../escape.whl'
            (wheel.parent/'receipt.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'filename'):
                compatible_wheel(root,'CuMesh','upstream','revision',runtime)
