"""Retarget core glTF TRS clips onto a packed skinned GLB rig.

This module samples source bind-relative joint motion, composes it onto the
output rig's bind world matrices, decomposes the resulting local transforms,
and appends LINEAR translation/rotation/scale tracks to the output GLB. It does
not infer a skeleton, decode skin weights, or create morph animation.
"""
import json
import struct
from pathlib import Path
import numpy as np

from .articulation import joint_deformation_matrices
from .glb import _glb_chunks
from .gltf_accessors import GLBAccessors
from .import_rig import load_rig_glb, rig_joint_world


def canonical_matrix(transform):
    """Return the homogeneous matrix for a ``CanonicalTransform``."""
    scale = float(transform.scale)
    matrix = np.eye(4)
    matrix[:3, :3] *= scale
    matrix[:3, 3] = -scale * np.asarray(transform.center, dtype=float)
    return matrix


def nearest_joint_mapping(source_joints, output_joints):
    """Return a one-to-one source-to-output mapping ordered by joint distance."""
    distances = np.linalg.norm(np.asarray(source_joints)[:, None]
                               - np.asarray(output_joints)[None], axis=2)
    mapping = {}
    used = set()
    for source in range(len(distances)):
        for output in np.argsort(distances[source], kind='stable'):
            if int(output) not in used:
                mapping[source] = int(output)
                used.add(int(output))
                break
    if len(mapping) != len(source_joints):
        raise ValueError('Joint mapping must be one-to-one')
    matched = np.array([distances[source, output] for source, output in mapping.items()])
    return mapping, matched


def _decompose(matrix):
    """Closest TRS decomposition of an affine matrix, with reconstruction error."""
    translation = np.array(matrix[:3, 3], dtype=float, copy=True)
    linear = np.array(matrix[:3, :3], dtype=float, copy=True)
    u, singular, vt = np.linalg.svd(linear)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        # Keep scales positive; the reconstruction error records any reflection
        # that TRS channels cannot represent.
        u[:, -1] *= -1
        rotation = u @ vt
    reconstructed = np.eye(4)
    reconstructed[:3, :3] = rotation @ np.diag(singular)
    reconstructed[:3, 3] = translation
    error = float(np.max(np.abs(reconstructed - matrix)))
    return translation, rotation, singular, error


def _matrix_to_quaternion(rotation):
    """Return a normalized (x, y, z, w) quaternion for a proper rotation."""
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    value = np.array([x, y, z, w], dtype=float)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Rotation produced an invalid quaternion')
    return value / norm


def _check_track(node, path, times, values):
    times = np.asarray(times, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    width = 4 if path == 'rotation' else 3
    if (not isinstance(node, (int, np.integer)) or node < 0
            or path not in ('translation', 'rotation', 'scale')
            or times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or times[0] < 0
            or values.shape != (len(times), width) or not np.isfinite(values).all()):
        raise ValueError('Animation tracks need ordered times and matching finite values')
    if path == 'scale' and np.any(values <= 0):
        raise ValueError('Animation scale values must be positive')
    return times, values


def append_trs_animation(data, tracks, *, name=None):
    """Append LINEAR T/R/S tracks to a self-contained GLB and return new bytes.

    ``tracks`` is an iterable of dicts with ``node``, ``path``, ``times`` and
    ``values``. Existing accessors, skins and animations are preserved.
    """
    if type(name) is not str and name is not None:
        raise ValueError('Animation name must be a string or None')
    chunks = _glb_chunks(data)
    if len(chunks) != 2 or chunks[1][0] != b'BIN\x00':
        raise ValueError('Animation attachment requires one JSON and one BIN chunk')
    document = json.loads(chunks[0][1].decode('utf-8'))
    buffers = document.get('buffers', [])
    if len(buffers) != 1 or 'uri' in buffers[0]:
        raise ValueError('Animation attachment requires one embedded buffer')
    payload = chunks[1][1][:buffers[0]['byteLength']]
    binary = bytearray(payload)
    buffer_views = document.setdefault('bufferViews', [])
    accessors = document.setdefault('accessors', [])
    nodes = document.get('nodes', [])

    def add_accessor(array, component_type, kind, *, minimum=None, maximum=None):
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        array = np.ascontiguousarray(array)
        payload = array.tobytes()
        binary.extend(payload)
        view_id = len(buffer_views)
        buffer_views.append(dict(buffer=0, byteOffset=offset, byteLength=len(payload)))
        accessor = dict(bufferView=view_id, byteOffset=0, componentType=component_type,
                        count=len(array), type=kind)
        if minimum is not None:
            accessor['min'] = [float(v) for v in minimum]
            accessor['max'] = [float(v) for v in maximum]
        accessors.append(accessor)
        return len(accessors) - 1

    samplers = []
    channels = []
    for track in tracks:
        if not isinstance(track, dict):
            raise ValueError('Each animation track must be a dict')
        node = track.get('node')
        path = track.get('path')
        if not isinstance(node, (int, np.integer)) or not 0 <= int(node) < len(nodes):
            raise ValueError('Animation track references an invalid node')
        times, values = _check_track(node, path, track.get('times'), track.get('values'))
        input_id = add_accessor(times, 5126, 'SCALAR',
                                minimum=[float(times.min())], maximum=[float(times.max())])
        output_id = add_accessor(values, 5126,
                                 'VEC4' if path == 'rotation' else 'VEC3')
        sampler = len(samplers)
        samplers.append(dict(input=input_id, output=output_id, interpolation='LINEAR'))
        channels.append(dict(sampler=sampler, target=dict(node=int(node), path=path)))
    if not channels:
        raise ValueError('At least one animation track is required')
    animation = dict(channels=channels, samplers=samplers)
    if name is not None:
        animation['name'] = name
    document.setdefault('animations', []).append(animation)
    buffers[0]['byteLength'] = len(binary)
    json_chunk = json.dumps(document, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    json_chunk += b' ' * ((-len(json_chunk)) % 4)
    binary = bytes(binary) + bytes((-len(binary)) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    return (struct.pack('<4sII', b'glTF', 2, total)
            + struct.pack('<I4s', len(json_chunk), b'JSON') + json_chunk
            + struct.pack('<I4s', len(binary), b'BIN\x00') + binary)


def _source_times(path, animation_index):
    glb = GLBAccessors(path)
    animations = glb.document.get('animations', [])
    if not 0 <= animation_index < len(animations):
        raise ValueError('Invalid animation index')
    animation = animations[animation_index]
    keyed = [glb.read(sampler['input'])[:, 0] for sampler in animation.get('samplers', [])]
    if not keyed:
        raise ValueError('Source animation has no samplers')
    return animation, np.unique(np.concatenate(keyed)).astype(float)


def retarget_animation(output, source, *, animation_index=0, name=None):
    """Return (animated GLB bytes, audit) retargeting one source clip onto output.

    Both inputs must be paths to core single-skin GLBs. The output's joints are
    matched to the source by normalized bind position and the parent hierarchy
    is checked before any track is written. Existing output animations are kept.
    """
    output, source = Path(output), Path(source)
    source_bind, source_transform = load_rig_glb(source).normalized()
    output_bind, output_transform = load_rig_glb(output).normalized()
    mapping, matched = nearest_joint_mapping(source_bind.skeleton.joints,
                                             output_bind.skeleton.joints)
    source_to_output = np.array([mapping[i] for i in range(len(mapping))], dtype=np.int64)
    output_to_source = np.empty_like(source_to_output)
    output_to_source[source_to_output] = np.arange(len(source_to_output))
    expected_parents = np.where(source_bind.skeleton.parents < 0, -1,
                               source_to_output[np.maximum(source_bind.skeleton.parents, 0)])
    mismatches = output_bind.skeleton.parents[source_to_output] != expected_parents
    if np.any(mismatches):
        raise ValueError('Mapped joint hierarchy does not match the source')

    source_bind_nodes, source_bind_world = rig_joint_world(source)
    output_bind_nodes, output_bind_world = rig_joint_world(output)
    del source_bind_nodes
    animation, times = _source_times(source, animation_index)
    source_matrix = canonical_matrix(source_transform)
    source_to_output_space = np.linalg.inv(canonical_matrix(output_transform)) @ source_matrix

    translations = np.zeros((len(times), len(output_to_source), 3))
    rotations = np.zeros((len(times), len(output_to_source), 4))
    scales = np.zeros((len(times), len(output_to_source), 3))
    worst_error = 0.
    parents = output_bind.skeleton.parents
    for row, time in enumerate(times):
        _, pose_world = rig_joint_world(source, animation_index=animation_index, time=float(time))
        deformation = joint_deformation_matrices(source_bind_world, pose_world)
        deformation = source_to_output_space @ deformation @ np.linalg.inv(source_to_output_space)
        pose_matrices = np.empty_like(output_bind_world)
        for index in range(len(output_bind_world)):
            pose_matrices[index] = deformation[output_to_source[index]] @ output_bind_world[index]
        previous = rotations[row - 1] if row else None
        for index in range(len(pose_matrices)):
            parent = int(parents[index])
            local = pose_matrices[index] if parent < 0 else (
                np.linalg.inv(pose_matrices[parent]) @ pose_matrices[index])
            translation, rotation, scale, error = _decompose(local)
            worst_error = max(worst_error, error)
            quaternion = _matrix_to_quaternion(rotation)
            if row and float(np.dot(quaternion, previous[index])) < 0:
                quaternion = -quaternion
            translations[row, index] = translation
            rotations[row, index] = quaternion
            scales[row, index] = scale

    tracks = []
    for index, node in enumerate(output_bind_nodes):
        tracks.append(dict(node=int(node), path='translation',
                           times=times, values=translations[:, index]))
        tracks.append(dict(node=int(node), path='rotation',
                           times=times, values=rotations[:, index]))
        tracks.append(dict(node=int(node), path='scale',
                           times=times, values=scales[:, index]))
    existing_animations = len(GLBAccessors(output).document.get('animations', []))
    data = append_trs_animation(Path(output).read_bytes(), tracks, name=name)
    audit = dict(animation_index=animation_index, animation_name=animation.get('name'),
                 output_animation_index=existing_animations,
                 times=len(times), time_range=[float(times[0]), float(times[-1])],
                 joints=len(output_to_source), channels=len(tracks),
                 source_to_output=source_to_output.tolist(),
                 output_to_source=output_to_source.tolist(),
                 matched_position_distance_mean=float(matched.mean()),
                 matched_position_distance_max=float(matched.max()),
                 max_trs_reconstruction_error=worst_error,
                 structural_validation=validate_animation(data))
    return data, audit


_WIDTHS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def _read_float_accessor(document, binary, index):
    accessors = document.get('accessors', [])
    if not isinstance(index, int) or not 0 <= index < len(accessors):
        raise ValueError('Invalid animation accessor index')
    accessor = accessors[index]
    if (accessor.get('extensions') or accessor.get('componentType') != 5126
            or accessor.get('type') not in _WIDTHS):
        raise ValueError('Animation accessors must be float32 without extensions')
    views = document.get('bufferViews', [])
    view_id = accessor.get('bufferView')
    if not isinstance(view_id, int) or not 0 <= view_id < len(views):
        raise ValueError('Animation accessor needs a valid bufferView')
    view = views[view_id]
    count = accessor.get('count')
    width = _WIDTHS[accessor['type']]
    start = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
    stride = view.get('byteStride', width * 4)
    if (not isinstance(count, int) or count <= 0 or stride < width * 4
            or stride % 4 or start % 4 or start < 0
            or start + (count - 1) * stride + width * 4 > len(binary)):
        raise ValueError('Animation accessor exceeds the binary buffer')
    return np.ndarray((count, width), dtype='<f4', buffer=binary,
                      offset=start, strides=(stride, 4)).copy()


def validate_animation(data):
    """Structurally validate core glTF animation tracks in a self-contained GLB.

    Checks sampler/channel references, float32 accessor spans, ordered finite
    times, monotonic scales and unit rotations. It is not the Khronos validator
    and does not evaluate skinning.
    """
    chunks = _glb_chunks(data)
    if len(chunks) != 2 or chunks[1][0] != b'BIN\x00':
        raise ValueError('Animation validation requires one JSON and one BIN chunk')
    document = json.loads(chunks[0][1].decode('utf-8'))
    binary = chunks[1][1][:document['buffers'][0]['byteLength']]
    animations = document.get('animations', [])
    if not animations:
        raise ValueError('GLB contains no animations')
    nodes = len(document.get('nodes', []))
    channel_count = 0
    for animation in animations:
        samplers = animation.get('samplers', [])
        channels = animation.get('channels', [])
        if not samplers or not channels:
            raise ValueError('Animation needs at least one sampler and channel')
        for sampler in samplers:
            interpolation = sampler.get('interpolation', 'LINEAR')
            if interpolation not in ('STEP', 'LINEAR', 'CUBICSPLINE'):
                raise ValueError('Unsupported animation interpolation')
            times = _read_float_accessor(document, binary, sampler.get('input'))
            if (times.shape[1] != 1 or not np.isfinite(times).all()
                    or np.any(times[:, 0] < 0) or np.any(np.diff(times[:, 0]) <= 0)):
                raise ValueError('Animation times must be finite, ordered and nonnegative')
        for channel in channels:
            sampler_id = channel.get('sampler')
            target = channel.get('target', {})
            node, path = target.get('node'), target.get('path')
            if (not isinstance(sampler_id, int) or not 0 <= sampler_id < len(samplers)
                    or not isinstance(node, int) or not 0 <= node < nodes
                    or path not in ('translation', 'rotation', 'scale')):
                raise ValueError('Animation channel has an invalid target')
            sampler = samplers[sampler_id]
            times = _read_float_accessor(document, binary, sampler.get('input'))
            values = _read_float_accessor(document, binary, sampler.get('output'))
            expected = (3 * len(times) if sampler.get('interpolation') == 'CUBICSPLINE'
                        else len(times))
            if values.shape != (expected, 4 if path == 'rotation' else 3):
                raise ValueError('Animation output shape does not match its channel')
            if not np.isfinite(values).all():
                raise ValueError('Animation values must be finite')
            if path == 'scale' and np.any(values <= 0):
                raise ValueError('Animation scales must be positive')
            if path == 'rotation':
                norms = np.linalg.norm(values, axis=1)
                if not np.allclose(norms, 1, rtol=0, atol=1e-4):
                    raise ValueError('Animation rotations must be unit quaternions')
            channel_count += 1
    return dict(animations=len(animations), samplers=sum(
        len(a.get('samplers', [])) for a in animations), channels=channel_count,
        accessors=len(document.get('accessors', [])),
        buffer_views=len(document.get('bufferViews', [])))
