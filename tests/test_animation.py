import struct
import tempfile
from pathlib import Path
import unittest
import numpy as np

from t3d.animation import append_trs_animation, retarget_animation, validate_animation
from t3d.articulation import Skeleton
from t3d.glb import pack_glb
from t3d.import_rig import load_rig_glb, rig_joint_world


def _base_glb():
    weights = np.array([[1., 0.], [.5, .5], [0., 1.]])
    skeleton = Skeleton([[0., 0., 0.], [0., 1., 0.]], [-1, 0])
    return pack_glb(
        [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], [[0, 1, 2]],
        [[0., 0.], [1., 0.], [0., 1.]], [[0., 0., 1.]] * 3,
        np.tile([.25, .5, .75, .25, .75, .5], (2, 2, 1)),
        source_up='y', skin_weights=weights, skeleton=skeleton)


def _joint_nodes(data):
    import json
    json_length = struct.unpack_from('<I', data, 12)[0]
    document = json.loads(data[20:20 + json_length])
    return document['skins'][0]['joints']


class AnimationTests(unittest.TestCase):
    def test_append_trs_animation_roundtrips_translation(self):
        data = _base_glb()
        root_node, child_node = _joint_nodes(data)
        tracks = [dict(node=child_node, path='translation', times=[0., 1.],
                       values=[[0., 1., 0.], [0., 2., 0.]])]
        animated = append_trs_animation(data, tracks, name='lift')
        validation = validate_animation(animated)
        self.assertEqual((validation['animations'], validation['samplers'],
                          validation['channels']), (1, 1, 1))
        with self.assertRaisesRegex(ValueError, 'no animations'):
            validate_animation(data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'animated.glb'
            path.write_bytes(animated)
            rig = load_rig_glb(path, animation_index=0, time=.5)
            nodes, world = rig_joint_world(path, animation_index=0, time=.5)
        self.assertEqual(rig.skeleton.joints[1][1], 1.5)
        np.testing.assert_allclose(world[1][:3, 3], [0., 1.5, 0.])
        np.testing.assert_allclose(rig.vertices[1], [1., .25, 0.], atol=1e-6)
        np.testing.assert_allclose(rig.vertices[2], [0., 1.5, 0.], atol=1e-6)
        self.assertEqual(len(nodes), 2)

    def test_retarget_identical_rigs_reproduces_source_motion(self):
        data = _base_glb()
        root_node, child_node = _joint_nodes(data)
        theta = np.pi / 4
        tracks = [
            dict(node=root_node, path='rotation', times=[0., 1.],
                 values=[[0., 0., 0., 1.], [0., 0., np.sin(theta / 2), np.cos(theta / 2)]]),
            dict(node=child_node, path='translation', times=[0., 1.],
                 values=[[0., 1., 0.], [0., 2., 0.]]),
        ]
        source = append_trs_animation(data, tracks, name='source')
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / 'source.glb'
            output_path = Path(directory) / 'output.glb'
            source_path.write_bytes(source)
            output_path.write_bytes(data)
            animated, audit = retarget_animation(output_path, source_path)
            target = Path(directory) / 'target.glb'
            target.write_bytes(animated)
            self.assertEqual(audit['joints'], 2)
            self.assertEqual(audit['times'], 2)
            self.assertLess(audit['max_trs_reconstruction_error'], 1e-6)
            for time in (0., .25, .5, .75, 1.):
                expected = load_rig_glb(source_path, animation_index=0, time=time)
                actual = load_rig_glb(target, animation_index=0, time=time)
                np.testing.assert_allclose(actual.skeleton.joints, expected.skeleton.joints,
                                           atol=1e-5)
                np.testing.assert_allclose(actual.vertices, expected.vertices, atol=1e-5)

    def test_reject_invalid_tracks(self):
        data = _base_glb()
        node = _joint_nodes(data)[1]
        for tracks in (
                [],
                [dict(node=node, path='translation', times=[1., 0.], values=[[0, 0, 0]] * 2)],
                [dict(node=node, path='scale', times=[0.], values=[[0., 1., 1.]])],
                [dict(node=999, path='translation', times=[0.], values=[[0., 0., 0.]])],
                [dict(node=node, path='weights', times=[0.], values=[[0., 0., 0.]])]):
            with self.assertRaises(ValueError):
                append_trs_animation(data, tracks)
