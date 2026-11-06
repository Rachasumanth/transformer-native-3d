from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from t3d.import_rig import load_rig_glb,node_world_matrices,sample_animation_nodes


class RigImportTests(unittest.TestCase):
    def test_world_skinning_intermediate_parent_and_mesh_transform(self):
        nodes=[{'translation':[10,0,0],'children':[1]}, {'translation':[2,0,0],'children':[2]},
               {'translation':[3,0,0]}, {'translation':[999,0,0],'skin':0,'mesh':0}]
        inverse=np.repeat(np.eye(4)[None],2,axis=0);inverse[:,0,3]=[-10,-13]
        v=np.array([[0.,0,0],[0,1,0],[1,0,0]]);weights=np.array([[.25,.75],[1,0],[0,1.]])
        arrays={0:v,1:inverse.transpose(0,2,1).reshape(2,16),2:np.array([[0],[1],[2]])}
        doc=dict(nodes=nodes,scenes=[{'nodes':[0,3]}],skins=[{'joints':[0,2],'inverseBindMatrices':1}],
                 meshes=[{'primitives':[{'attributes':{'POSITION':0},'indices':2}]}],
                 accessors=[{'type':'VEC3','componentType':5126},{'type':'MAT4','componentType':5126},
                            {'type':'SCALAR','componentType':5123}])
        fake=SimpleNamespace(document=doc,read=lambda i:arrays[i],skin_weights=lambda p,j:(weights,0.))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'rig.glb';path.write_bytes(b'fixture')
            with patch('t3d.import_rig.GLBAccessors',return_value=fake):rig=load_rig_glb(path)
        np.testing.assert_allclose(rig.vertices,[[1.5,0,0],[0,1,0],[3,0,0]])
        np.testing.assert_allclose(rig.skeleton.joints,[[10,0,0],[15,0,0]])
        np.testing.assert_array_equal(rig.skeleton.parents,[-1,0])
        normalized,transform=rig.normalized()
        np.testing.assert_allclose(transform.inverse(normalized.skeleton.joints),rig.skeleton.joints)

    def test_quaternion_matrix_and_invalid_hierarchy(self):
        q=np.sqrt(.5)
        world,parents=node_world_matrices([{'rotation':[0,0,q,q],'children':[1]}, {'translation':[2,0,0]}])
        np.testing.assert_allclose(world[1,:3,3],[0,2,0],atol=1e-12)
        matrix=np.eye(4);matrix[:3,3]=[3,4,5]
        result,_=node_world_matrices([{'matrix':matrix.flatten(order='F').tolist()}])
        np.testing.assert_array_equal(result[0],matrix)
        for nodes in ([{'children':[1]},{'children':[0]}],
                      [{'children':[2]},{'children':[2]},{}],
                      [{'matrix':matrix.flatten().tolist(),'translation':[0,0,0]}]):
            with self.assertRaises(ValueError):node_world_matrices(nodes)

    def test_animation_step_linear_and_cubic_sampling(self):
        times=np.array([[0.],[1.]])
        linear=np.array([[0.,0,0],[2.,0,0]])
        step=np.array([[0.,0,0],[2.,0,0]])
        cubic=np.array([[0.,0,0],[0,0,0],[1,0,0],
                        [-1.,0,0],[2,0,0],[0,0,0]])
        arrays={0:times,1:linear,2:step,3:cubic}
        def apply(output,interpolation):
            animation={'samplers':[{'input':0,'output':output,'interpolation':interpolation}],
                       'channels':[{'sampler':0,'target':{'node':0,'path':'translation'}}]}
            return sample_animation_nodes([{}],animation,lambda i:arrays[i],.5)[0]['translation']
        np.testing.assert_allclose(apply(1,'LINEAR'),[1,0,0])
        np.testing.assert_allclose(apply(2,'STEP'),[0,0,0])
        np.testing.assert_allclose(apply(3,'CUBICSPLINE'),[1.25,0,0])

    def test_animation_quaternion_slerp_and_invalid_channels(self):
        q0=np.array([0.,0,0,1]);q1=np.array([0.,0,1,0])
        arrays={0:np.array([[0.],[1.]]),1:np.stack([q0,q1])}
        animation={'samplers':[{'input':0,'output':1}],
                   'channels':[{'sampler':0,'target':{'node':0,'path':'rotation'}}]}
        nodes=sample_animation_nodes([{}],animation,lambda i:arrays[i],.5)
        np.testing.assert_allclose(nodes[0]['rotation'],[0,0,np.sqrt(.5),np.sqrt(.5)],atol=1e-12)
        bad={'samplers':[{'input':0,'output':1}],
             'channels':[{'sampler':0,'target':{'node':0,'path':'weights'}}]}
        with self.assertRaises(ValueError):sample_animation_nodes([{}],bad,lambda i:arrays[i],.5)

    def test_animated_pose_updates_skeleton_and_skinned_vertices(self):
        vertices=np.array([[0.,0,0],[1,0,0],[0,0,1]])
        inverse=np.eye(4).reshape(1,16)
        arrays={0:vertices,1:inverse,2:np.array([[0],[1],[2]]),
                3:np.array([[0.],[1.]]),4:np.array([[0.,0,0],[0,2.,0]])}
        weights=np.ones((3,1))
        doc={'nodes':[{}, {'skin':0,'mesh':0}], 'scenes':[{'nodes':[0,1]}], 'scene':0,
             'skins':[{'joints':[0],'inverseBindMatrices':1}],
             'meshes':[{'primitives':[{'attributes':{'POSITION':0},'indices':2}]}],
             'accessors':[{'type':'VEC3','componentType':5126},
                          {'type':'MAT4','componentType':5126},
                          {'type':'SCALAR','componentType':5123},
                          {'type':'SCALAR','componentType':5126},
                          {'type':'VEC3','componentType':5126}],
             'animations':[{'samplers':[{'input':3,'output':4}],
                           'channels':[{'sampler':0,'target':{'node':0,'path':'translation'}}]}]}
        fake=SimpleNamespace(document=doc,read=lambda i:arrays[i],
                             skin_weights=lambda p,j:(weights,0.))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'rig.glb';path.write_bytes(b'animated-fixture')
            with patch('t3d.import_rig.GLBAccessors',return_value=fake):
                rig=load_rig_glb(path,animation_index=0,time=.5)
        np.testing.assert_allclose(rig.skeleton.joints,[[0,1,0]])
        np.testing.assert_allclose(rig.vertices,vertices+[[0,1,0]])
