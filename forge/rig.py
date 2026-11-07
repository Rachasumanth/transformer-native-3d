"""Minimal GLB skin/skeleton reader for rig supervision.

Reads the glTF JSON and BIN chunks directly: joint world positions from the node
hierarchy, joint parents, and per-vertex JOINTS_0/WEIGHTS_0 from mesh primitives.
Assets with more than MAX_JOINTS joints are rejected for the v1 template rig.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import numpy as np
from scipy.spatial import cKDTree

MAX_JOINTS = 24
_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
              5125: np.uint32, 5126: np.float32}
_WIDTH = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


@dataclass
class RigData:
    joints: np.ndarray
    parents: np.ndarray
    vertices: np.ndarray
    joint_ids: np.ndarray
    weights: np.ndarray

    @property
    def joint_count(self):
        return len(self.joints)


def _chunks(data):
    if len(data) < 20 or struct.unpack_from('<4sI', data, 0) != (b'glTF', 2):
        raise ValueError('not a GLB')
    length, kind = struct.unpack_from('<II', data, 12)
    if kind != 0x4e4f534a:
        raise ValueError('missing JSON chunk')
    document = json.loads(data[20:20 + length])
    binary = b''
    offset = 20 + length
    if offset < len(data):
        bin_length, bin_kind = struct.unpack_from('<II', data, offset)
        if bin_kind == 0x004e4942:
            binary = data[offset + 8:offset + 8 + bin_length]
    return document, binary


def _accessor(document, binary, index):
    accessor = document['accessors'][index]
    view = document['bufferViews'][accessor['bufferView']]
    dtype = np.dtype(_COMPONENT[accessor['componentType']])
    width = _WIDTH[accessor['type']]
    count = accessor['count']
    start = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
    stride = view.get('byteStride') or dtype.itemsize * width
    if stride == dtype.itemsize * width:
        values = np.frombuffer(binary, dtype=dtype, count=count * width, offset=start)
        return values.reshape(count, width)
    rows = []
    for i in range(count):
        rows.append(np.frombuffer(binary, dtype=dtype, count=width,
                                  offset=start + i * stride))
    return np.stack(rows)


def _local_matrix(node):
    if 'matrix' in node:
        return np.asarray(node['matrix'], dtype=np.float64).reshape(4, 4).T
    matrix = np.eye(4)
    translation = node.get('translation', [0.0, 0.0, 0.0])
    rotation = node.get('rotation', [0.0, 0.0, 0.0, 1.0])
    scale = node.get('scale', [1.0, 1.0, 1.0])
    x, y, z, w = rotation
    rotation_matrix = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    matrix[:3, :3] = rotation_matrix @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def _world_matrices(document):
    nodes = document.get('nodes', [])
    parent = {}
    for index, node in enumerate(nodes):
        for child in node.get('children', []):
            parent[child] = index

    def world(index):
        chain = []
        cursor = index
        seen = set()
        while cursor is not None and cursor not in seen:
            seen.add(cursor)
            chain.append(cursor)
            cursor = parent.get(cursor)
        matrix = np.eye(4)
        for node_index in reversed(chain):
            matrix = matrix @ _local_matrix(nodes[node_index])
        return matrix

    return world, parent


def load_rig(path):
    """Return RigData or None if the asset has no usable skin."""
    suffix = Path(path).suffix.lower()
    if suffix != '.glb':
        return None
    try:
        document, binary = _chunks(Path(path).read_bytes())
    except Exception:
        return None
    skins = document.get('skins') or []
    if not skins:
        return None
    skin = skins[0]
    joints = skin.get('joints') or []
    if not (2 <= len(joints) <= MAX_JOINTS):
        return None
    world, parent = _world_matrices(document)
    joint_positions = np.stack([world(int(j))[:3, 3] for j in joints])
    joint_set = {int(j): i for i, j in enumerate(joints)}
    joint_parents = np.full(len(joints), -1, dtype=np.int64)
    for slot, node_index in enumerate(joints):
        cursor = parent.get(int(node_index))
        while cursor is not None and cursor not in joint_set:
            cursor = parent.get(cursor)
        joint_parents[slot] = joint_set.get(cursor, -1)
    mesh_nodes = {}
    for index, node in enumerate(document.get('nodes', [])):
        if 'mesh' in node:
            mesh_nodes.setdefault(node['mesh'], []).append(index)
    vertices, joints_all, weights_all = [], [], []
    for mesh_index, mesh in enumerate(document.get('meshes', [])):
        nodes = mesh_nodes.get(mesh_index, [])
        if not nodes:
            continue
        mesh_world = world(nodes[0])
        for primitive in mesh.get('primitives', []):
            attributes = primitive.get('attributes', {})
            if 'POSITION' not in attributes or 'JOINTS_0' not in attributes \
                    or 'WEIGHTS_0' not in attributes:
                continue
            position = _accessor(document, binary, attributes['POSITION'])[:, :3]
            position = (mesh_world[:3, :3] @ position.T).T + mesh_world[:3, 3]
            joint = _accessor(document, binary, attributes['JOINTS_0'])
            weight = _accessor(document, binary, attributes['WEIGHTS_0']).astype(np.float64)
            if len(joint) != len(position) or len(weight) != len(position):
                continue
            vertices.append(position)
            joints_all.append(joint.astype(np.int64))
            weights_all.append(weight)
    if not vertices:
        return None
    vertices = np.concatenate(vertices)
    joint_ids = np.concatenate(joints_all)
    weights = np.concatenate(weights_all)
    total = weights.sum(axis=1, keepdims=True)
    keep = total[:, 0] > 0
    vertices, joint_ids, weights = vertices[keep], joint_ids[keep], weights[keep]
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
    if not len(vertices):
        return None
    return RigData(joint_positions.astype(np.float64), joint_parents, vertices,
                   joint_ids, weights)


def sample_rig(rig, points, *, top=4):
    """Nearest-vertex skin weights at arbitrary points (normalized coordinates)."""
    if rig is None or not len(rig.vertices):
        return None
    tree = cKDTree(rig.vertices)
    _, nearest = tree.query(np.asarray(points, dtype=np.float64))
    weights = np.zeros((len(points), rig.joint_count), dtype=np.float32)
    for row, vertex in enumerate(nearest):
        ids = rig.joint_ids[vertex]
        values = rig.weights[vertex]
        order = np.argsort(-values, kind='stable')[:top]
        weights[row, ids[order]] = values[order]
    total = weights.sum(axis=1, keepdims=True)
    return weights / np.maximum(total, 1e-9)
