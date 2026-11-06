import unittest
import struct
import tempfile
from pathlib import Path
import numpy as np
from t3d.latent_container import LatentAsset,dumps_latents,loads_latents,save_latents,load_latents
from t3d.articulation import Skeleton


class LatentContainerTests(unittest.TestCase):
    def asset(self,**kw):
        values=dict(coords=[[1,0,0],[0,0,0]],shape=[[.1,.2],[.3,.4]],material=[[1],[2]],rig=[[3]*4,[4]*4],
                    grid_size=(32,)*3,aabb=[[-.5]*3,[.5]*3],shape_model_sha256='a'*64,
                    material_model_sha256='b'*64,rig_model_sha256='c'*64,
                    skeleton=Skeleton([[0,0,0],[0,.5,0]],[-1,0]))
        values.update(kw);return LatentAsset(**values)

    def test_all_streams_joints_curves_and_tokens(self):
        for curve in ('z','tz','h','th'):
            data=dumps_latents(self.asset(),curve=curve,bpt_tokens=[0,5119])
            restored=loads_latents(data,expected_models={'shape':'a'*64})
            self.assertEqual(restored.curve,curve)
            np.testing.assert_allclose(restored.latents.shape,[[.3,.4],[.1,.2]],atol=.0002)
            np.testing.assert_array_equal(restored.latents.skeleton.parents,[-1,0])
            np.testing.assert_array_equal(restored.bpt_tokens,[0,5119])
            self.assertEqual(dumps_latents(restored.latents,curve=curve,bpt_tokens=restored.bpt_tokens),data)
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            loads_latents(data,expected_models={'shape':'d'*64})

    def test_shape_only_and_empty(self):
        a=self.asset(material=None,material_model_sha256=None,rig=None,rig_model_sha256=None,skeleton=None)
        self.assertIsNone(loads_latents(dumps_latents(a)).latents.material)
        a=LatentAsset(np.empty((0,3),int),np.empty((0,8)),(16,)*3,[[-.5]*3,[.5]*3],'a'*64)
        self.assertEqual(loads_latents(dumps_latents(a)).latents.shape.shape,(0,8))

    def test_atomic_save_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'latents.t3d'
            save_latents(path,self.asset())
            restored=load_latents(path)
            self.assertEqual(len(restored.latents.coords),2)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),['latents.t3d'])

    def test_corruption_and_validation(self):
        data=dumps_latents(self.asset())
        self.assertEqual(struct.unpack_from('<H',data,110)[0],4)
        changed_payload=bytearray(data);changed_payload[128]^=1
        with self.assertRaisesRegex(ValueError,'integrity'):
            loads_latents(changed_payload)
        # Profile 2 predates the checksum trailer and remains readable.
        legacy=bytearray(data[:-32]);struct.pack_into('<H',legacy,110,2)
        self.assertEqual(len(loads_latents(legacy).latents.coords),2)
        for payload in (data[:127],data[:-1],data+b'junk'):
            with self.assertRaises(ValueError):loads_latents(payload)
        for offset,value in ((56,1),(72,8),(110,99),(116,1)):
            bad=bytearray(data);struct.pack_into('<H',bad,offset,value)
            with self.assertRaises(ValueError):loads_latents(bad)
        for change in (dict(coords=[[2,0,0],[0,0,0]]),dict(shape_model_sha256=''),dict(shape=[[1e10]*2]*2)):
            with self.assertRaises(ValueError):dumps_latents(self.asset(**change))
