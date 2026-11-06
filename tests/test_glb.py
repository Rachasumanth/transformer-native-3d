import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
import trimesh
from t3d.articulation import Skeleton
from t3d.glb import pack_glb,limit_skin_influences,rewrite_skin_weights
from t3d.import_rig import load_rig_glb
from t3d.gltf_accessors import GLBAccessors


class GLBTests(unittest.TestCase):
    def test_skin_influence_limit_keeps_top_weights_and_normalizes(self):
        weights=np.tile(np.arange(1,21,dtype=float),(2,1))
        weights[0]=1
        limited,audit=limit_skin_influences(weights,max_influences=16)
        np.testing.assert_array_equal(np.flatnonzero(limited[0]),np.arange(16))
        np.testing.assert_array_equal(np.flatnonzero(limited[1]),np.arange(4,20))
        np.testing.assert_allclose(limited.sum(1),1,atol=1e-12)
        self.assertEqual(audit['max_influences_before'],20)
        self.assertEqual(audit['max_influences_after'],16)
        self.assertAlmostEqual(audit['mean_discarded_mass'],(.2+10/210)/2)
        with self.assertRaisesRegex(ValueError,'positive finite mass'):
            limit_skin_influences(np.zeros((1,2)))

    def test_rewrite_skin_weights_limits_in_place_and_roundtrips(self):
        args=self.fixture();weights=np.array([[1,0,0,0,0],[.1,.2,.3,.15,.25],[0,0,0,0,1.]])
        skeleton=Skeleton([[0,0,0],[0,.5,0],[.2,.7,0],[.4,.8,0],[.6,.9,0]],[-1,0,1,1,3])
        data=pack_glb(**args,source_up='z',skin_weights=weights,skeleton=skeleton)
        sparse,report=rewrite_skin_weights(data,weights,max_influences=1)
        self.assertEqual(len(sparse),len(data))
        self.assertEqual(report['max_influences_after'],1)
        self.assertEqual(report['skin_attribute_sets_after'],1)
        self.assertGreater(report['truncated_rows'],0)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'sparse.glb';path.write_bytes(sparse)
            rig=load_rig_glb(path)
        expected=np.zeros((3,5));expected[np.arange(3),weights.argmax(axis=1)]=1
        np.testing.assert_allclose(rig.weights,expected,atol=1e-7)
        np.testing.assert_array_equal(rig.faces,np.asarray(args['faces']))
        with self.assertRaisesRegex(ValueError,'match POSITION'):
            rewrite_skin_weights(data,weights[:2],max_influences=1)
        with self.assertRaises(ValueError):
            rewrite_skin_weights(b'not a glb',weights,max_influences=1)
        with self.assertRaisesRegex(ValueError,'one skin'):
            rewrite_skin_weights(pack_glb(**args),weights,max_influences=1)
        concentrated=np.zeros_like(weights);concentrated[:,0]=1
        concentrated_data=pack_glb(**args,source_up='z',skin_weights=concentrated,
                                   skeleton=skeleton)
        with self.assertRaisesRegex(ValueError,'more skin attribute sets'):
            rewrite_skin_weights(concentrated_data,weights,max_influences=16)

    def fixture(self):
        return dict(vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]],
            uv=[[0,0],[1,0],[0,1]],normals=[[0,0,1]]*3,
            pbr=np.tile([.25,.0,1.,.25,.75,.5],(2,2,1)))

    def test_embedded_pbr_and_coordinate_roundtrip(self):
        args=self.fixture()
        data=pack_glb(**args,alpha_mode='BLEND')
        magic,version,length=struct.unpack_from('<4sII',data)
        self.assertEqual((magic,version,length),(b'glTF',2,len(data)))
        json_length,kind=struct.unpack_from('<I4s',data,12)
        self.assertEqual(kind,b'JSON')
        document=json.loads(data[20:20+json_length])
        self.assertIn('NORMAL',document['meshes'][0]['primitives'][0]['attributes'])
        self.assertTrue(all('uri' not in x for x in document['images']))
        self.assertTrue(all('uri' not in x for x in document['buffers']))
        mesh=trimesh.load(io.BytesIO(data),file_type='glb',force='mesh',process=False)
        np.testing.assert_allclose(mesh.vertices,[[0,0,0],[1,0,0],[0,0,-1]])
        np.testing.assert_allclose(mesh.visual.uv,[[0,1],[1,1],[0,0]])
        np.testing.assert_allclose(mesh.vertex_normals,[[0,1,0]]*3)
        mat=mesh.visual.material
        self.assertEqual(mat.alphaMode,'BLEND')
        np.testing.assert_array_equal(np.asarray(mat.baseColorTexture)[0,0],[137,0,255,128])
        np.testing.assert_array_equal(np.asarray(mat.metallicRoughnessTexture)[0,0],[0,191,64])
        self.assertEqual(mat.metallicFactor,1)
        self.assertEqual(mat.roughnessFactor,1)

    def test_reject_invalid_payload_and_conventions(self):
        for changes in (dict(source_up='x'),dict(alpha_mode='invalid'),dict(faces=[[0,1,3]]),
                        dict(normals=[[0,0,0]]*3),dict(pbr=np.ones((2,2,6))*2)):
            args=self.fixture();args.update(changes)
            with self.assertRaises(ValueError):pack_glb(**args)

    def test_skinned_glb_roundtrip_preserves_geometry_skeleton_and_weights(self):
        args=self.fixture();weights=np.array([[1,0,0,0,0],[.1,.2,.3,.15,.25],[0,0,0,0,1.]])
        skeleton=Skeleton([[0,0,0],[0,.5,0],[.2,.7,0],[.4,.8,0],[.6,.9,0]],[-1,0,1,1,3])
        data=pack_glb(**args,source_up='z',skin_weights=weights,skeleton=skeleton)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rigged.glb';path.write_bytes(data)
            doc=GLBAccessors(path).document
            primitive=doc['meshes'][0]['primitives'][0]
            self.assertIn('JOINTS_0',primitive['attributes'])
            self.assertIn('JOINTS_1',primitive['attributes'])
            for accessor_id in primitive['attributes'].values():
                view_id=doc['accessors'][accessor_id]['bufferView']
                self.assertEqual(doc['bufferViews'][view_id]['target'],34962)
            index_view=doc['accessors'][primitive['indices']]['bufferView']
            self.assertEqual(doc['bufferViews'][index_view]['target'],34963)
            self.assertEqual(len(doc['skins'][0]['joints']),5)
            rig=load_rig_glb(path)
        expected_vertices=np.asarray(args['vertices'])[:,[0,2,1]]*[1,1,-1]
        expected_joints=skeleton.joints[:,[0,2,1]]*[1,1,-1]
        np.testing.assert_allclose(rig.vertices,expected_vertices,atol=1e-6)
        np.testing.assert_allclose(rig.skeleton.joints,expected_joints,atol=1e-6)
        np.testing.assert_array_equal(rig.skeleton.parents,skeleton.parents)
        np.testing.assert_allclose(rig.weights,weights,atol=1e-7)

    def test_skinned_glb_requires_matching_skin_inputs(self):
        args=self.fixture();weights=np.ones((3,1))
        with self.assertRaisesRegex(ValueError,'supplied together'):
            pack_glb(**args,skin_weights=weights)
        with self.assertRaisesRegex(ValueError,'normalized finite'):
            pack_glb(**args,skin_weights=np.full((3,1),.5),skeleton=Skeleton([[0,0,0]],[-1]))

    def test_pack_static_glb_roundtrip_and_rejection(self):
        from t3d.glb import pack_static_glb
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        glb = pack_static_glb(verts, faces, source_up='z')
        self.assertEqual(glb[:4], b'glTF')
        loaded = trimesh.load(trimesh.util.wrap_as_stream(glb), file_type='glb')
        mesh = loaded.to_geometry() if hasattr(loaded, 'to_geometry') else loaded
        if isinstance(mesh, list):
            mesh = mesh[0]
        self.assertEqual(len(mesh.vertices), 3)
        self.assertEqual(len(mesh.faces), 1)

        with self.assertRaises(ValueError):
            pack_static_glb(np.zeros((0, 3)), faces)
        with self.assertRaises(ValueError):
            pack_static_glb(verts, np.array([[0, 1, 5]]))
        with self.assertRaises(ValueError):
            pack_static_glb(verts, faces, source_up='invalid')

    def test_pack_static_glb_preserves_six_pbr_channels_in_materials_and_textures(self):
        from t3d.glb import pack_static_glb, validate_glb_semantics
        from t3d.import_glb import load_glb
        verts = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0, 0.5, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        # 1. Uniform PBR preservation across all six channels
        pbr_uniform = np.array([[0.2, 0.3, 0.4, 0.75, 0.25, 0.9]] * 3, dtype=np.float32)
        glb_u = pack_static_glb(verts, faces, pbr=pbr_uniform, source_up='z')
        audit_u = validate_glb_semantics(glb_u)
        self.assertTrue(audit_u['valid'])
        self.assertTrue(audit_u['six_channels_preserved'])
        self.assertFalse(audit_u['has_color_0'])
        self.assertTrue(audit_u['pbr_semantics']['has_base_color_texture'])
        self.assertTrue(audit_u['pbr_semantics']['has_metallic_roughness_texture'])

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'uniform.glb'
            p.write_bytes(glb_u)
            imported = load_glb(p)
            self.assertEqual(len(imported.materials), 1)
            mat = imported.materials[0]
            self.assertIsNotNone(mat.base_texture)
            self.assertIsNotNone(mat.metallic_roughness_texture)
            # Roughness is G (channel 1), Metallic is B (channel 2)
            np.testing.assert_allclose(mat.metallic_roughness_texture[..., 1].mean(), 0.25, atol=0.05)
            np.testing.assert_allclose(mat.metallic_roughness_texture[..., 2].mean(), 0.75, atol=0.05)
            np.testing.assert_allclose(mat.base_texture[..., 3].mean(), 0.9, atol=0.05)

        # 2. Spatially varying PBR preservation via per-triangle UV atlas
        pbr_varying = np.array([
            [0.9, 0.1, 0.2, 0.1, 0.9, 1.0],
            [0.1, 0.9, 0.2, 0.5, 0.5, 1.0],
            [0.2, 0.1, 0.9, 0.8, 0.2, 0.6]
        ], dtype=np.float32)
        glb_v = pack_static_glb(verts, faces, pbr=pbr_varying, source_up='z')
        audit_v = validate_glb_semantics(glb_v)
        self.assertTrue(audit_v['valid'])
        self.assertTrue(audit_v['six_channels_preserved'])
        self.assertFalse(audit_v['has_color_0'])
        self.assertTrue(audit_v['has_uv'])
        self.assertTrue(audit_v['pbr_semantics']['has_base_color_texture'])
        self.assertTrue(audit_v['pbr_semantics']['has_metallic_roughness_texture'])

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'varying.glb'
            p.write_bytes(glb_v)
            imported_v = load_glb(p)
            self.assertEqual(len(imported_v.materials), 1)
            mat_v = imported_v.materials[0]
            self.assertIsNotNone(mat_v.base_texture)
            self.assertIsNotNone(mat_v.metallic_roughness_texture)
            # Verify that metallic ranges across variation
            mr_tex = mat_v.metallic_roughness_texture
            self.assertGreater(mr_tex[..., 2].max(), 0.6)
            self.assertLess(mr_tex[..., 2].min(), 0.3)
            # Verify that roughness ranges across variation
            self.assertGreater(mr_tex[..., 1].max(), 0.7)
            self.assertLess(mr_tex[..., 1].min(), 0.4)

    def test_pack_static_glb_with_rigging_and_attach_skin(self):
        from t3d.glb import pack_static_glb, validate_glb_semantics, attach_skin
        from t3d.import_rig import load_rig_glb
        verts = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0, 0.5, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        pbr_varying = np.array([
            [0.8, 0.1, 0.1, 0.2, 0.8, 1.0],
            [0.1, 0.8, 0.1, 0.5, 0.5, 1.0],
            [0.1, 0.1, 0.8, 0.9, 0.1, 1.0],
        ], dtype=np.float32)
        skel = Skeleton(np.array([[0, 0, 0], [0, 0.5, 0]], dtype=np.float32), np.array([-1, 0], dtype=np.int64))
        weights = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)

        # 1. Packaging with skin directly
        glb_rigged = pack_static_glb(verts, faces, pbr=pbr_varying, skin_weights=weights, skeleton=skel, source_up='z')
        audit = validate_glb_semantics(glb_rigged)
        self.assertTrue(audit['valid'])
        self.assertTrue(audit['skinning']['is_skinned'])
        self.assertEqual(audit['skinning']['joint_count'], 2)
        self.assertTrue(audit['six_channels_preserved'])

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'rigged.glb'
            p.write_bytes(glb_rigged)
            rig = load_rig_glb(p, include_pbr=True)
            self.assertEqual(len(rig.skeleton.joints), 2)
            self.assertEqual(len(rig.pbr_materials), 1)

        # 2. Packaging static first, then attaching skin via attach_skin
        glb_static = pack_static_glb(verts, faces, pbr=pbr_varying, source_up='z')
        glb_attached = attach_skin(glb_static, weights, skel)
        audit_att = validate_glb_semantics(glb_attached)
        self.assertTrue(audit_att['valid'])
        self.assertTrue(audit_att['skinning']['is_skinned'])

    def test_validate_glb_semantics_comprehensive(self):
        from t3d.glb import pack_static_glb, validate_glb_semantics
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        glb = pack_static_glb(verts, faces, source_up='z')
        report = validate_glb_semantics(glb)
        self.assertTrue(report['valid'])
        self.assertEqual(report['vertex_count'], 3)
        self.assertEqual(report['face_count'], 1)
        self.assertFalse(report['skinning']['is_skinned'])

        # Corrupt header rejection
        with self.assertRaisesRegex(ValueError, 'complete binary glTF|Invalid GLB header'):
            validate_glb_semantics(b'glTFcorrupt')

        # Invalid PBR inputs rejection in pack_static_glb
        with self.assertRaisesRegex(ValueError, 'finite'):
            pack_static_glb(verts, faces, pbr=np.array([[np.nan]*6]*3))
        with self.assertRaisesRegex(ValueError, 'finite'):
            pack_static_glb(verts, faces, pbr=np.array([[-0.1]*6]*3))
        with self.assertRaisesRegex(ValueError, 'shape'):
            pack_static_glb(verts, faces, pbr=np.zeros((3, 5)))
        with self.assertRaisesRegex(ValueError, 'shape'):
            pack_static_glb(verts, faces, pbr=np.zeros((4, 6)))


