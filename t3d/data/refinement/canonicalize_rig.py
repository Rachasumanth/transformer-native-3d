"""Canonicalize skeleton rigs and skinning weights for T3D refinement.

Includes:
1. Dead-Bone & IK Pruning (< 0.02 max weight) with parent relinking and weight transfer.
2. Deterministic Spatial DFS Ordering breaking child ties lexicographically by (y, x, z).
3. Barycentric BVH Weight Transfer from raw source to remeshed watertight vertices with
   1 step of manifold-edge Laplacian smoothing.
4. GLB Top-4 Truncation packing into standard GLTF JOINTS_0 (uint16[4]) and WEIGHTS_0 (float32[4]).
5. PartNet-Mobility URDF Converter for revolute and prismatic joints into merged skinned GLBs.
"""
from dataclasses import dataclass
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh

from t3d.articulation import Skeleton
from t3d.proximity import TriangleBVH
from t3d.glb import attach_skin


class RigValidationError(ValueError):
    """Raised when skeleton or skinning weights violate canonicalization constraints."""
    pass


def prune_dead_bones(weights, skeleton, threshold=0.02):
    """Prune any bone j where max_i W[i, j] < threshold (default 0.02).

    Transfers residual weights of pruned bones to their nearest surviving ancestor,
    and relinks children to that ancestor.
    Rejects rigs where remaining joint count J < 2 or J > 64.

    Parameters
    ----------
    weights : array-like of shape (V, J)
        Per-vertex skinning weights. Each row must sum to 1.
    skeleton : Skeleton
        Skeleton instance containing joints (J, 3) and parents (J,).
    threshold : float, optional
        Minimum maximum-influence threshold for a bone to remain alive (default 0.02).

    Returns
    -------
    pruned_weights : np.ndarray of shape (V, J_alive)
        Normalized skinning weights for surviving bones.
    pruned_skeleton : Skeleton
        Surviving skeleton with updated joints and relinked parent indices.
    pruned_indices : list[int]
        Original indices of pruned dead bones.
    """
    if not isinstance(skeleton, Skeleton):
        raise RigValidationError("skeleton must be a valid t3d.articulation.Skeleton instance")
    W = np.asarray(weights, dtype=np.float64)
    J = len(skeleton.joints)
    if W.ndim != 2 or W.shape[1] != J or len(W) == 0:
        raise RigValidationError(f"Weights shape {W.shape} does not match skeleton joints {J}")
    if J < 2 or J > 64:
        raise RigValidationError(f"Asset joint count J={J} outside valid range [2, 64]")

    max_influences = np.max(W, axis=0)
    is_dead = max_influences < threshold

    if not np.any(is_dead):
        return W.copy(), skeleton, []

    parents = np.array(skeleton.parents, copy=True)

    # For each dead bone, find its target alive ancestor.
    # If no alive ancestor exists (e.g. root is dead), find its nearest alive descendant.
    target_alive = np.full(J, -1, dtype=np.int64)
    for j in range(J):
        if is_dead[j]:
            curr = parents[j]
            while curr >= 0 and is_dead[curr]:
                curr = parents[curr]
            target_alive[j] = curr

    # Resolve any dead bones with no alive ancestors
    for j in range(J):
        if is_dead[j] and target_alive[j] == -1:
            stack = [j]
            descendants = []
            while stack:
                node = stack.pop()
                for child in range(J):
                    if parents[child] == node:
                        if not is_dead[child]:
                            descendants.append(child)
                        else:
                            stack.append(child)
            if descendants:
                target_alive[j] = max(descendants, key=lambda d: max_influences[d])
            else:
                alive_nodes = np.where(~is_dead)[0]
                if len(alive_nodes) == 0:
                    raise RigValidationError("All skeleton bones are dead; asset rejected")
                target_alive[j] = alive_nodes[0]

    # For each alive bone, determine its new parent by skipping dead ancestors
    new_parents_old_indices = np.full(J, -1, dtype=np.int64)
    for j in range(J):
        if not is_dead[j]:
            curr = parents[j]
            while curr >= 0 and is_dead[curr]:
                curr = parents[curr]
            new_parents_old_indices[j] = curr

    alive_indices = np.where(~is_dead)[0]
    J_alive = len(alive_indices)
    if J_alive < 2 or J_alive > 64:
        raise RigValidationError(f"Post-pruning joint count J={J_alive} outside valid range [2, 64]")

    old_to_alive = {old: new for new, old in enumerate(alive_indices)}

    alive_joints = skeleton.joints[alive_indices]
    alive_parents = np.array([
        -1 if new_parents_old_indices[old] == -1 else old_to_alive[new_parents_old_indices[old]]
        for old in alive_indices
    ], dtype=np.int64)

    # Accumulate weights: alive bones keep their weights + residual weights from dead bones
    new_weights = np.zeros((len(W), J_alive), dtype=np.float64)
    for j in range(J):
        if not is_dead[j]:
            new_weights[:, old_to_alive[j]] += W[:, j]
        else:
            target = target_alive[j]
            new_weights[:, old_to_alive[target]] += W[:, j]

    # Re-normalize to ensure exact partition of unity
    row_sums = new_weights.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise RigValidationError("Zero-sum row encountered after pruning dead bones")
    new_weights /= row_sums

    pruned_skeleton = Skeleton(alive_joints, alive_parents)
    pruned_indices = list(np.where(is_dead)[0])
    return new_weights, pruned_skeleton, pruned_indices


def dfs_order_skeleton(skeleton, weights=None):
    """Sort all skeleton joints in Depth-First Search (DFS) order from the root.

    Child branch ties are broken lexicographically by (y, x, z) coordinates.
    Updates joint order, parent references, and skinning weight columns.

    Parameters
    ----------
    skeleton : Skeleton
        Skeleton instance.
    weights : array-like of shape (V, J), optional
        Per-vertex skinning weights to permute alongside joints.

    Returns
    -------
    new_skeleton : Skeleton
        Skeleton with joints and parents reordered according to DFS order.
    new_weights : np.ndarray, optional
        Permuted weights (only returned if `weights` is not None).
    dfs_order : np.ndarray
        Array of original joint indices in their new DFS order.
    """
    if not isinstance(skeleton, Skeleton):
        raise RigValidationError("skeleton must be a valid t3d.articulation.Skeleton instance")
    J = len(skeleton.joints)
    if J < 2 or J > 64:
        raise RigValidationError(f"Skeleton joint count J={J} outside valid range [2, 64]")

    roots = [i for i in range(J) if skeleton.parents[i] == -1]
    if not roots:
        raise RigValidationError("Skeleton has no root joint (parent == -1)")

    children = [[] for _ in range(J)]
    for i, p in enumerate(skeleton.parents):
        if p >= 0:
            children[p].append(i)

    # Tie-breaking key: lexicographically by (y, x, z) coordinates
    def child_key(idx):
        pos = skeleton.joints[idx]
        return (float(pos[1]), float(pos[0]), float(pos[2]), idx)

    dfs_order = []

    def traverse(u):
        dfs_order.append(u)
        for c in sorted(children[u], key=child_key):
            traverse(c)

    for r in sorted(roots, key=child_key):
        traverse(r)

    if len(dfs_order) != J:
        raise RigValidationError(f"DFS traversal visited {len(dfs_order)}/{J} joints")

    dfs_order = np.array(dfs_order, dtype=np.int64)
    old_to_new = np.empty(J, dtype=np.int64)
    old_to_new[dfs_order] = np.arange(J)

    new_joints = skeleton.joints[dfs_order]
    new_parents = np.where(skeleton.parents[dfs_order] == -1, -1, old_to_new[skeleton.parents[dfs_order]])
    new_skeleton = Skeleton(new_joints, new_parents)

    if weights is not None:
        W = np.asarray(weights, dtype=np.float64)
        if W.shape[1] != J:
            raise RigValidationError(f"Weights column count {W.shape[1]} != skeleton joint count {J}")
        new_weights = W[:, dfs_order]
        return new_skeleton, new_weights, dfs_order

    return new_skeleton, dfs_order


def barycentric_weight_transfer(raw_vertices, raw_faces, raw_weights, clean_vertices):
    """Transfer skinning weights from raw surface to clean vertices via TriangleBVH.

    Finds the closest surface triangle on (raw_vertices, raw_faces) for each clean vertex,
    and performs exact barycentric interpolation of the vertex weights.

    Parameters
    ----------
    raw_vertices : array-like of shape (V_raw, 3)
    raw_faces : array-like of shape (F_raw, 3)
    raw_weights : array-like of shape (V_raw, J)
    clean_vertices : array-like of shape (V_clean, 3)

    Returns
    -------
    clean_weights : np.ndarray of shape (V_clean, J)
        Interpolated skinning weights on the clean vertices.
    """
    raw_v = np.asarray(raw_vertices, dtype=np.float64)
    raw_f = np.asarray(raw_faces, dtype=np.int64)
    raw_w = np.asarray(raw_weights, dtype=np.float64)
    clean_v = np.asarray(clean_vertices, dtype=np.float64)

    if raw_w.ndim != 2 or len(raw_w) != len(raw_v):
        raise RigValidationError("raw_weights must have shape (V_raw, J) matching raw_vertices")

    bvh = TriangleBVH(raw_v, raw_f)
    face_ids, bary = bvh.query(clean_v)

    triangles = raw_f[face_ids]  # (V_clean, 3)
    # Barycentric interpolation: sum_k bary[:, k] * raw_w[triangles[:, k]]
    clean_weights = np.einsum('ni,nij->nj', bary, raw_w[triangles])

    # Re-normalize to prevent numerical drift
    row_sums = clean_weights.sum(axis=1, keepdims=True)
    clean_weights = np.where(row_sums > 0, clean_weights / row_sums, clean_weights)
    return clean_weights


def laplacian_smooth_weights(weights, faces, *, steps=1, lambda_param=0.5):
    """Apply manifold-edge Laplacian smoothing steps over skinning weights.

    Only edges shared by 1 or 2 faces (manifold edges) are considered.

    Parameters
    ----------
    weights : array-like of shape (V, J)
    faces : array-like of shape (F, 3)
    steps : int, optional
        Number of smoothing steps (default 1).
    lambda_param : float, optional
        Smoothing damping factor in [0, 1] (default 0.5).

    Returns
    -------
    smoothed_weights : np.ndarray of shape (V, J)
        Smoothed and renormalized skinning weights.
    """
    W = np.array(weights, dtype=np.float64, copy=True)
    V = len(W)
    f = np.asarray(faces, dtype=np.int64)

    if V == 0 or len(f) == 0 or steps <= 0 or lambda_param <= 0:
        return W

    # Extract all directed half-edges and sort pairs to identify undirected edges
    e01 = np.sort(f[:, [0, 1]], axis=1)
    e12 = np.sort(f[:, [1, 2]], axis=1)
    e20 = np.sort(f[:, [2, 0]], axis=1)
    all_edges = np.vstack([e01, e12, e20])

    unique_edges, counts = np.unique(all_edges, axis=0, return_counts=True)
    # Manifold edges appear in 1 or 2 faces
    manifold_mask = (counts >= 1) & (counts <= 2)
    manifold_edges = unique_edges[manifold_mask]

    if len(manifold_edges) == 0:
        return W

    u = manifold_edges[:, 0]
    v = manifold_edges[:, 1]
    src = np.concatenate([u, v])
    dst = np.concatenate([v, u])

    deg = np.zeros(V, dtype=np.float64)
    np.add.at(deg, src, 1.0)
    has_neighbors = deg > 0

    for _ in range(steps):
        neighbor_sum = np.zeros_like(W)
        np.add.at(neighbor_sum, src, W[dst])

        W_new = W.copy()
        W_new[has_neighbors] = (
            (1.0 - lambda_param) * W[has_neighbors]
            + lambda_param * (neighbor_sum[has_neighbors] / deg[has_neighbors, None])
        )
        row_sums = W_new.sum(axis=1, keepdims=True)
        W = np.where(row_sums > 0, W_new / row_sums, W_new)

    return W


def truncate_top4_weights(weights):
    """Restrict each vertex to its top K=4 bone weights and re-normalize.

    Packs directly into standard GLTF JOINTS_0 (uint16[4]) and WEIGHTS_0 (float32[4]).

    Parameters
    ----------
    weights : array-like of shape (V, J)

    Returns
    -------
    dict
        'JOINTS_0': np.ndarray of shape (V, 4) with dtype uint16
        'WEIGHTS_0': np.ndarray of shape (V, 4) with dtype float32
        'truncated_weights': np.ndarray of shape (V, J) with dtype float64
        'discarded_mass': np.ndarray of shape (V,)
    """
    W = np.asarray(weights, dtype=np.float64)
    V, J = W.shape
    if V == 0 or J == 0:
        raise RigValidationError("Cannot truncate empty weights matrix")

    # Stable descending sort preserves lowest joint index on weight ties
    order = np.argsort(-W, axis=1, kind='stable')
    k = min(4, J)
    selected_indices = order[:, :k]
    selected_weights = np.take_along_axis(W, selected_indices, axis=1)

    retained_mass = selected_weights.sum(axis=1, keepdims=True)
    if np.any(retained_mass <= 0):
        raise RigValidationError("Encountered vertex with zero retained weight mass")

    # Renormalize top weights
    renorm_top_weights = selected_weights / retained_mass

    joints_0 = np.zeros((V, 4), dtype=np.uint16)
    weights_0 = np.zeros((V, 4), dtype=np.float32)

    joints_0[:, :k] = selected_indices.astype(np.uint16)
    weights_0[:, :k] = renorm_top_weights.astype(np.float32)

    # In GLTF, components with zero weight are conventioned to joint index 0
    joints_0[weights_0 == 0] = 0

    # Ensure float32 rows strictly sum to 1.0
    f32_sums = weights_0.sum(axis=1, keepdims=True)
    weights_0 /= f32_sums

    # Full (V, J) truncated representation
    truncated_dense = np.zeros_like(W)
    np.put_along_axis(truncated_dense, selected_indices, renorm_top_weights, axis=1)

    discarded_mass = np.maximum(0.0, 1.0 - retained_mass.squeeze(axis=1))

    return {
        'JOINTS_0': joints_0,
        'WEIGHTS_0': weights_0,
        'truncated_weights': truncated_dense,
        'discarded_mass': discarded_mass,
    }


def pack_gltf_skin_accessors(joints_0, weights_0):
    """Pack JOINTS_0 and WEIGHTS_0 arrays into standard GLTF accessor specifications.

    Parameters
    ----------
    joints_0 : np.ndarray of shape (V, 4) and dtype uint16
    weights_0 : np.ndarray of shape (V, 4) and dtype float32

    Returns
    -------
    dict
        Dictionary containing binary buffers and accessor metadata for JOINTS_0 and WEIGHTS_0.
    """
    j_arr = np.ascontiguousarray(joints_0, dtype='<u2')
    w_arr = np.ascontiguousarray(weights_0, dtype='<f4')
    j_bytes = j_arr.tobytes()
    w_bytes = w_arr.tobytes()

    return {
        'JOINTS_0': {
            'array': j_arr,
            'bytes': j_bytes,
            'componentType': 5123,  # UNSIGNED_SHORT
            'type': 'VEC4',
            'count': len(joints_0),
            'byteLength': len(j_bytes),
        },
        'WEIGHTS_0': {
            'array': w_arr,
            'bytes': w_bytes,
            'componentType': 5126,  # FLOAT
            'type': 'VEC4',
            'count': len(weights_0),
            'byteLength': len(w_bytes),
        }
    }


def _rpy_to_matrix(r, p, y):
    """Compute 3x3 rotation matrix from URDF roll-pitch-yaw Euler angles."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    # URDF uses fixed-axis / extrinsic XYZ: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _parse_origin(element):
    """Parse origin xyz and rpy into a 4x4 affine transform."""
    t = np.zeros(3, dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    if element is not None:
        xyz_str = element.attrib.get('xyz')
        if xyz_str:
            t = np.fromstring(xyz_str, sep=' ', dtype=np.float64)
        rpy_str = element.attrib.get('rpy')
        if rpy_str:
            rpy = np.fromstring(rpy_str, sep=' ', dtype=np.float64)
            if len(rpy) == 3:
                R = _rpy_to_matrix(rpy[0], rpy[1], rpy[2])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _load_geometry(geom_elem, base_dir):
    """Load or construct mesh for a visual geometry element."""
    if geom_elem is None:
        return None
    mesh_elem = geom_elem.find('mesh')
    if mesh_elem is not None:
        filename = mesh_elem.attrib.get('filename', '')
        if filename.startswith('package://'):
            filename = filename[len('package://'):]
        mesh_path = (base_dir / filename) if base_dir else Path(filename)
        if mesh_path.exists():
            loaded = trimesh.load(str(mesh_path), force='mesh', process=False)
            if isinstance(loaded, trimesh.Scene):
                loaded = loaded.dump(concatenate=True)
            return loaded

    box_elem = geom_elem.find('box')
    if box_elem is not None:
        size = np.fromstring(box_elem.attrib.get('size', '1 1 1'), sep=' ', dtype=np.float64)
        return trimesh.creation.box(extents=size)

    cyl_elem = geom_elem.find('cylinder')
    if cyl_elem is not None:
        radius = float(cyl_elem.attrib.get('radius', '0.5'))
        length = float(cyl_elem.attrib.get('length', '1.0'))
        return trimesh.creation.cylinder(radius=radius, height=length)

    sphere_elem = geom_elem.find('sphere')
    if sphere_elem is not None:
        radius = float(sphere_elem.attrib.get('radius', '0.5'))
        return trimesh.creation.icosphere(radius=radius, subdivisions=2)

    return None


def parse_partnet_urdf(urdf_path, *, base_dir=None):
    """Parse a PartNet-Mobility URDF file into a merged skinned GLB asset.

    Extracts revolute, continuous, and prismatic joints into a kinematic hierarchy,
    calculates exact joint axes and pivot locations in world coordinates,
    and assigns binary rigid skinning weights (W[i, j] = 1.0) to each moving link.

    Parameters
    ----------
    urdf_path : str or Path
        Path to `mobility.urdf` or URDF XML string.
    base_dir : str or Path, optional
        Base directory for resolving relative mesh filenames. If None, uses parent of urdf_path.

    Returns
    -------
    dict
        'vertices': (V, 3) merged vertices in world coordinates
        'faces': (F, 3) merged faces
        'weights': (V, J) binary rigid skinning weights
        'skeleton': Skeleton instance
        'joint_names': list of J joint names
        'joint_types': list of J joint types ('root', 'revolute', 'prismatic', etc.)
        'joint_axes': (J, 3) normalized joint axes in world coordinates
        'joint_origins': (J, 3) joint origin positions in world coordinates
        'glb_bytes': bytes of the self-contained binary skinned GLB
        'links': dict mapping link names to link metadata
    """
    if isinstance(urdf_path, (str, Path)) and os.path.exists(str(urdf_path)):
        tree = ET.parse(str(urdf_path))
        root_elem = tree.getroot()
        if base_dir is None:
            base_dir = Path(urdf_path).parent
    elif isinstance(urdf_path, str) and urdf_path.strip().startswith('<'):
        root_elem = ET.fromstring(urdf_path)
        base_dir = Path(base_dir) if base_dir else Path.cwd()
    else:
        raise FileNotFoundError(f"Cannot find URDF file at {urdf_path}")

    links = {}
    for link_elem in root_elem.findall('link'):
        name = link_elem.attrib['name']
        visuals = []
        for vis in link_elem.findall('visual'):
            origin_elem = vis.find('origin')
            T_vis = _parse_origin(origin_elem)
            geom_elem = vis.find('geometry')
            mesh = _load_geometry(geom_elem, base_dir)
            visuals.append({'origin': T_vis, 'mesh': mesh})
        links[name] = {'name': name, 'visuals': visuals}

    # Extract joints
    raw_joints = []
    child_to_joint = {}
    for j_elem in root_elem.findall('joint'):
        j_name = j_elem.attrib['name']
        j_type = j_elem.attrib.get('type', 'fixed')
        parent_elem = j_elem.find('parent')
        child_elem = j_elem.find('child')
        if parent_elem is None or child_elem is None:
            continue
        p_name = parent_elem.attrib['link']
        c_name = child_elem.attrib['link']
        T_joint = _parse_origin(j_elem.find('origin'))

        axis_elem = j_elem.find('axis')
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if axis_elem is not None:
            axis_str = axis_elem.attrib.get('xyz')
            if axis_str:
                axis = np.fromstring(axis_str, sep=' ', dtype=np.float64)
                norm = np.linalg.norm(axis)
                if norm > 1e-8:
                    axis /= norm

        j_info = {
            'name': j_name,
            'type': j_type,
            'parent': p_name,
            'child': c_name,
            'origin': T_joint,
            'axis': axis,
        }
        raw_joints.append(j_info)
        child_to_joint[c_name] = j_info

    # Find root link (a link that is not a child of any joint)
    root_links = [l for l in links if l not in child_to_joint]
    if not root_links:
        # Fallback to 'base' if present, otherwise first link
        root_link = 'base' if 'base' in links else next(iter(links.keys()))
    else:
        root_link = 'base' if 'base' in root_links else root_links[0]

    # Compute world transforms for each link via BFS from root
    world_transforms = {root_link: np.eye(4, dtype=np.float64)}
    queue = [root_link]
    while queue:
        curr = queue.pop(0)
        T_curr = world_transforms[curr]
        # Find children
        for j_info in raw_joints:
            if j_info['parent'] == curr and j_info['child'] not in world_transforms:
                child = j_info['child']
                world_transforms[child] = T_curr @ j_info['origin']
                queue.append(child)

    # Any disconnected links get identity
    for l in links:
        if l not in world_transforms:
            world_transforms[l] = np.eye(4, dtype=np.float64)

    # Map links to skeleton bones:
    # Bone 0: base link (root)
    # Moving links with revolute/prismatic/continuous joints each get a unique bone
    # Fixed joint children collapse to their parent's bone
    bone_names = [root_link]
    bone_types = ['root']
    bone_axes = [np.array([0.0, 0.0, 1.0], dtype=np.float64)]
    bone_origins = [world_transforms[root_link][:3, 3].copy()]
    bone_parents = [-1]

    link_to_bone = {root_link: 0}

    # Queue links in topological order from joints
    processed_links = {root_link}
    added = True
    while added:
        added = False
        for j_info in raw_joints:
            c = j_info['child']
            p = j_info['parent']
            if p in processed_links and c not in processed_links:
                processed_links.add(c)
                added = True
                p_bone = link_to_bone[p]
                j_type = j_info['type']
                if j_type in ('revolute', 'continuous', 'prismatic'):
                    # New moving bone
                    new_bone_idx = len(bone_names)
                    bone_names.append(j_info['name'])
                    bone_types.append(j_type)
                    # Pivot is origin of child frame in world
                    pivot = world_transforms[c][:3, 3].copy()
                    bone_origins.append(pivot)
                    # World axis is child rotation @ local axis
                    R_child = world_transforms[c][:3, :3]
                    w_axis = R_child @ j_info['axis']
                    norm = np.linalg.norm(w_axis)
                    if norm > 1e-8:
                        w_axis /= norm
                    bone_axes.append(w_axis)
                    bone_parents.append(p_bone)
                    link_to_bone[c] = new_bone_idx
                else:
                    # Fixed joint: link moves rigidly with parent link
                    link_to_bone[c] = p_bone

    # For any links still unmapped, map to root
    for l in links:
        if l not in link_to_bone:
            link_to_bone[l] = 0

    J = len(bone_names)
    # If J < 2, ensure at least 2 joints by adding a virtual child joint so rig is valid [2, 64]
    if J == 1:
        bone_names.append('virtual_joint')
        bone_types.append('fixed')
        bone_axes.append(np.array([0.0, 0.0, 1.0], dtype=np.float64))
        bone_origins.append(bone_origins[0] + np.array([0.0, 0.0, 0.1]))
        bone_parents.append(0)
        J = 2

    skeleton = Skeleton(np.array(bone_origins, dtype=np.float64), np.array(bone_parents, dtype=np.int64))

    # Merge geometries and assign binary rigid skinning weights
    all_v = []
    all_f = []
    all_w = []
    vertex_offset = 0

    for link_name, link_data in links.items():
        bone_idx = link_to_bone.get(link_name, 0)
        T_link = world_transforms.get(link_name, np.eye(4, dtype=np.float64))
        for vis in link_data['visuals']:
            mesh = vis['mesh']
            if mesh is None or len(mesh.vertices) == 0:
                continue
            T_vis_world = T_link @ vis['origin']
            # Transform vertices to world coordinates
            v_local = mesh.vertices
            v_world = (T_vis_world[:3, :3] @ v_local.T).T + T_vis_world[:3, 3]
            f_local = mesh.faces

            n_verts = len(v_world)
            w_rigid = np.zeros((n_verts, J), dtype=np.float64)
            w_rigid[:, bone_idx] = 1.0

            all_v.append(v_world)
            all_f.append(f_local + vertex_offset)
            all_w.append(w_rigid)
            vertex_offset += n_verts

    if not all_v:
        # Synthetic unit box fallback if URDF has no visual geometry
        box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        all_v.append(box.vertices)
        all_f.append(box.faces)
        w_rigid = np.zeros((len(box.vertices), J), dtype=np.float64)
        w_rigid[:, 0] = 1.0
        all_w.append(w_rigid)

    merged_v = np.vstack(all_v)
    merged_f = np.vstack(all_f)
    merged_w = np.vstack(all_w)

    # Export merged geometry as GLB and attach skin
    mesh = trimesh.Trimesh(vertices=merged_v, faces=merged_f, process=False)
    base_glb = mesh.export(file_type='glb')
    glb_bytes = attach_skin(base_glb, merged_w, skeleton)

    return {
        'vertices': merged_v,
        'faces': merged_f,
        'weights': merged_w,
        'skeleton': skeleton,
        'joint_names': bone_names,
        'joint_types': bone_types,
        'joint_axes': np.array(bone_axes, dtype=np.float64),
        'joint_origins': np.array(bone_origins, dtype=np.float64),
        'glb_bytes': glb_bytes,
        'links': links,
    }


def canonicalize_rig(raw_vertices, raw_weights, skeleton, clean_vertices, clean_faces,
                     raw_faces=None, *, prune_threshold=0.02, laplacian_steps=1,
                     laplacian_lambda=0.5):
    """Canonicalize a skeletal rig and skinning weights through the Milestone 1 refinement pipeline.

    1. Prunes dead/IK bones with max weight < prune_threshold, transferring weights to parent.
    2. Sorts joints deterministically in spatial DFS order breaking child ties by (y, x, z).
    3. Transfers weights from raw mesh to clean watertight mesh via TriangleBVH barycentric lookup.
    4. Smooths transferred weights with 1 step of manifold-edge Laplacian smoothing.
    5. Truncates weights to top-4 influences, renormalizes, and packs into GLTF accessors.
    6. Packs into a standard skinned GLB.

    Parameters
    ----------
    raw_vertices : array-like of shape (V_raw, 3)
    raw_weights : array-like of shape (V_raw, J)
    skeleton : Skeleton
    clean_vertices : array-like of shape (V_clean, 3)
    clean_faces : array-like of shape (F_clean, 3)
    raw_faces : array-like of shape (F_raw, 3), optional
    prune_threshold : float, optional (default 0.02)
    laplacian_steps : int, optional (default 1)
    laplacian_lambda : float, optional (default 0.5)

    Returns
    -------
    dict
        'skeleton': canonicalized Skeleton
        'clean_vertices': (V_clean, 3) np.ndarray
        'clean_faces': (F_clean, 3) np.ndarray
        'weights': (V_clean, J_final) smoothed continuous skin weights
        'truncated_weights': (V_clean, J_final) top-4 truncated dense skin weights
        'JOINTS_0': (V_clean, 4) np.ndarray of dtype uint16
        'WEIGHTS_0': (V_clean, 4) np.ndarray of dtype float32
        'pruned_joints': list of pruned joint indices
        'dfs_order': array of joint indices in DFS traversal order
        'accessors': dict of packed GLTF accessor specifications
        'glb_bytes': bytes of the self-contained binary skinned GLB
        'audit': summary dictionary of canonicalization statistics
    """
    raw_v = np.asarray(raw_vertices, dtype=np.float64)
    raw_w = np.asarray(raw_weights, dtype=np.float64)
    clean_v = np.asarray(clean_vertices, dtype=np.float64)
    clean_f = np.asarray(clean_faces, dtype=np.int64)

    # 1. Dead-Bone & IK Pruning
    pruned_w, pruned_skel, pruned_indices = prune_dead_bones(raw_w, skeleton, threshold=prune_threshold)

    # 2. Deterministic Spatial DFS Ordering
    dfs_skel, dfs_w, dfs_order = dfs_order_skeleton(pruned_skel, pruned_w)

    # 3. Barycentric BVH Weight Transfer
    if raw_faces is not None and len(raw_faces) > 0:
        transferred_w = barycentric_weight_transfer(raw_v, raw_faces, dfs_w, clean_v)
    else:
        # Fallback when raw_faces is omitted: if vertices match directly or query nearest neighbors
        if len(clean_v) == len(raw_v) and np.allclose(clean_v, raw_v, atol=1e-5):
            transferred_w = dfs_w.copy()
        else:
            # Nearest-vertex fallback
            from scipy.spatial import cKDTree
            tree = cKDTree(raw_v)
            _, nearest_ids = tree.query(clean_v)
            transferred_w = dfs_w[nearest_ids].copy()

    # 1 step of manifold-edge Laplacian smoothing
    smoothed_w = laplacian_smooth_weights(
        transferred_w, clean_f, steps=laplacian_steps, lambda_param=laplacian_lambda
    )

    # 4. GLB Top-4 Truncation
    trunc_result = truncate_top4_weights(smoothed_w)
    joints_0 = trunc_result['JOINTS_0']
    weights_0 = trunc_result['WEIGHTS_0']
    truncated_dense = trunc_result['truncated_weights']

    # Pack accessors
    accessors = pack_gltf_skin_accessors(joints_0, weights_0)

    # Create self-contained skinned GLB
    mesh = trimesh.Trimesh(vertices=clean_v, faces=clean_f, process=False)
    base_glb = mesh.export(file_type='glb')
    glb_bytes = attach_skin(base_glb, truncated_dense, dfs_skel)

    audit = {
        'initial_joints': len(skeleton.joints),
        'pruned_joints_count': len(pruned_indices),
        'final_joints': len(dfs_skel.joints),
        'mean_discarded_mass': float(np.mean(trunc_result['discarded_mass'])),
        'max_discarded_mass': float(np.max(trunc_result['discarded_mass'])),
        'laplacian_steps': laplacian_steps,
        'laplacian_lambda': laplacian_lambda,
    }

    return {
        'skeleton': dfs_skel,
        'clean_vertices': clean_v,
        'clean_faces': clean_f,
        'weights': smoothed_w,
        'truncated_weights': truncated_dense,
        'JOINTS_0': joints_0,
        'WEIGHTS_0': weights_0,
        'pruned_joints': pruned_indices,
        'dfs_order': dfs_order,
        'accessors': accessors,
        'glb_bytes': glb_bytes,
        'audit': audit,
    }
