import hashlib
import json
import struct
from pathlib import Path
import tempfile
import unittest
import numpy as np
import trimesh
from PIL import Image
from t3d.import_glb import load_glb, load_glb_geometry, prepare_glb, _document
from t3d.container import load


class GlbImportTests(unittest.TestCase):
    def textured_triangle(self):
        mesh=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]],process=False)
        material=trimesh.visual.material.PBRMaterial(baseColorFactor=[1.,1.,1.,1.],
                   baseColorTexture=Image.fromarray(np.array([[[255,0,0],[0,255,0]],[[0,0,255],[255,255,255]]],dtype=np.uint8)),
                   metallicFactor=.2,roughnessFactor=.7)
        mesh.visual=trimesh.visual.texture.TextureVisuals(uv=[[0,0],[1,0],[0,1]],material=material)
        return mesh

    def test_instances_transforms_and_uv_convention(self):
        mesh=self.textured_triangle(); scene=trimesh.Scene()
        scene.add_geometry(mesh,node_name='first')
        matrix=np.eye(4);matrix[0,0]=-1;matrix[0,3]=3
        scene.add_geometry(mesh,node_name='second',transform=matrix)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'test.glb';p.write_bytes(scene.export(file_type='glb'))
            result=load_glb(p)
            self.assertEqual(len(result.faces),2)
            self.assertAlmostEqual(result.vertices[:,0].max(),3)
            self.assertEqual(len(result.materials),2)
            # Original trimesh UV V is bottom-up; importer restores glTF top-down.
            np.testing.assert_allclose(result.corner_uv[0],[[0,1],[1,1],[0,0]])
            normal=np.cross(result.vertices[result.faces[:,1]]-result.vertices[result.faces[:,0]],
                            result.vertices[result.faces[:,2]]-result.vertices[result.faces[:,0]])
            self.assertTrue(np.all(normal[:,2]>0))

    def test_end_to_end_textured_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.glb';output=Path(tmp)/'prepared.t3d'
            source.write_bytes(self.textured_triangle().export(file_type='glb'))
            report=prepare_glb(source,output,grid_size=4)
            asset=load(output).asset
            self.assertEqual(report['faces'],1)
            self.assertIsNotNone(asset.pbr)
            self.assertGreater(np.ptp(asset.pbr[:,:3]),.1)
            np.testing.assert_allclose(asset.pbr[:,3],.2,atol=.001)
            np.testing.assert_allclose(asset.pbr[:,4],.7,atol=.001)

    def test_corrupt_header(self):
        with self.assertRaises(ValueError): _document(b'invalid')

    def test_ambiguous_crossings_require_explicit_diagnostic_import(self):
        v=[];f=[]
        for x in (.31,.4):
            offset=len(v)
            v.extend([[x,0,0],[x,1,0],[x,1,1],[x,0,1]])
            f.extend([[offset,offset+1,offset+2],[offset,offset+2,offset+3]])
        mesh=trimesh.Trimesh(vertices=v,faces=f,process=False)
        mesh.visual=trimesh.visual.texture.TextureVisuals(uv=np.zeros((8,2)),
                         material=trimesh.visual.material.PBRMaterial())
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'thin.glb';out=Path(tmp)/'thin.t3d'
            p.write_bytes(mesh.export(file_type='glb'))
            with self.assertRaisesRegex(ValueError,'multiple distinct crossings'):
                prepare_glb(p,out,grid_size=3)
            self.assertFalse(out.exists())
            report=prepare_glb(p,out,grid_size=3,allow_multiple_crossings=True)
            self.assertGreater(report['crossing_diagnostics']['multiple_crossing_edges'],0)
            self.assertTrue(report['multiple_crossings_allowed'])

    def test_geometry_only_allows_material_uv_transform_but_rejects_geometry_extensions(self):
        original=self.textured_triangle().export(file_type='glb')
        length,kind=struct.unpack_from('<II',original,12);doc=json.loads(original[20:20+length])
        material=doc['materials'][0]['pbrMetallicRoughness']['baseColorTexture']
        material['extensions']={'KHR_texture_transform':{'offset':[0.2,0.3]}}
        doc['extensionsRequired']=['KHR_texture_transform']
        def pack(value):
            raw=json.dumps(value,separators=(',',':')).encode();raw+=b' '*((-len(raw))%4)
            tail=original[20+length:];return struct.pack('<4sII',b'glTF',2,20+len(raw)+len(tail))+struct.pack('<II',len(raw),kind)+raw+tail
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'transformed.glb';path.write_bytes(pack(doc))
            with self.assertRaises(ValueError):load_glb(path)
            v,f,digest=load_glb_geometry(path)
            self.assertEqual(v.shape,(3,3));self.assertEqual(f.shape,(1,3))
            self.assertEqual(digest,hashlib.sha256(path.read_bytes()).hexdigest())
            doc['extensionsRequired']=['KHR_draco_mesh_compression']
            path.write_bytes(pack(doc))
            with self.assertRaisesRegex(ValueError,'Required glTF extensions'):load_glb_geometry(path)
