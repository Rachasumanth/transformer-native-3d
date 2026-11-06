import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from t3d.gltf_accessors import GLBAccessors


def fixture(folder, binary, views, accessors):
    doc=dict(asset={'version':'2.0'},buffers=[{'byteLength':len(binary)}],bufferViews=views,accessors=accessors)
    raw=json.dumps(doc).encode();raw+=b' '*((-len(raw))%4)
    padded=binary+b'\0'*((-len(binary))%4)
    data=struct.pack('<4sII',b'glTF',2,28+len(raw)+len(padded))
    data+=struct.pack('<II',len(raw),0x4e4f534a)+raw+struct.pack('<II',len(padded),0x004e4942)+padded
    path=Path(folder)/'fixture.glb';path.write_bytes(data)
    return GLBAccessors(path)


class AccessorTests(unittest.TestCase):
    def test_interleaved_normalized_values_and_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            glb=fixture(folder,bytes([0,255,17,17,128,64,17,17]),
                        [{'buffer':0,'byteLength':8,'byteStride':4}],
                        [{'bufferView':0,'componentType':5121,'count':2,'type':'VEC2','normalized':True}])
            np.testing.assert_allclose(glb.read(0),[[0,1],[128/255,64/255]])
            glb.document['accessors'][0]['count']=3
            with self.assertRaises(ValueError):glb.read(0)

    def test_sparse_zero_base_and_bad_index_order(self):
        with tempfile.TemporaryDirectory() as folder:
            glb=fixture(folder,bytes([0,2,0,0])+struct.pack('<ff',4.,7.),
                        [{'buffer':0,'byteOffset':0,'byteLength':2},{'buffer':0,'byteOffset':4,'byteLength':8}],
                        [{'componentType':5126,'count':3,'type':'SCALAR','sparse':{
                            'count':2,'indices':{'bufferView':0,'componentType':5121},'values':{'bufferView':1}}}])
            np.testing.assert_array_equal(glb.read(0)[:,0],[4,0,7])
            glb.binary=bytes([2,0])+glb.binary[2:]
            with self.assertRaises(ValueError):glb.read(0)

    def test_signed_normalization_and_matrix_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            binary=bytes([128,127,0,0])+np.arange(16,dtype='<f4').tobytes()
            glb=fixture(folder,binary,[{'buffer':0,'byteLength':2},{'buffer':0,'byteOffset':4,'byteLength':64}],
                        [{'bufferView':0,'componentType':5120,'normalized':True,'count':2,'type':'SCALAR'},
                         {'bufferView':1,'componentType':5126,'count':1,'type':'MAT4'}])
            np.testing.assert_array_equal(glb.read(0)[:,0],[-1,1])
            np.testing.assert_array_equal(glb.read(1)[0],np.arange(16))
