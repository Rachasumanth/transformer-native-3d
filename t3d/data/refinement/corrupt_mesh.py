"""Synthetic Raw Mesh Corruption Generator for Transformer-Native 3D Refinement Pipeline.

Procedurally degrades clean canonical meshes M_clean into realistic raw/broken inputs M_raw:
1. Random patch hole deletion (10%–25% of faces around 2–4 seed vertices).
2. Aggressive unconstrained decimation (300–500 faces).
3. Isotropic vertex jitter N(0, 0.008^2) and 32^3 coordinate quantization.
4. 3–8 floating internal/external distractor triangles.
"""
from typing import Union, Tuple, Optional
import numpy as np
import trimesh
import fast_simplification


class CorruptedMesh(trimesh.Trimesh):
    """Subclass of trimesh.Trimesh that supports unpacking as (vertices, faces)."""
    def __iter__(self):
        return iter((self.vertices, self.faces))


def synthesize_corrupted_mesh(
    clean_mesh: Union[trimesh.Trimesh, Tuple[np.ndarray, np.ndarray], np.ndarray],
    faces: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    noise_std: float = 0.008,
) -> trimesh.Trimesh:
    """Procedurally synthesize a corrupted raw mesh paired with clean geometry.

    Args:
        clean_mesh: trimesh.Trimesh, or vertices array (if faces provided), or (vertices, faces) tuple.
        faces: Optional face index array if clean_mesh is a vertex array.
        seed: Random seed for deterministic reproducibility.
        noise_std: Standard deviation of isotropic vertex jitter (default: 0.008).

    Returns:
        Corrupted trimesh.Trimesh containing open holes, coordinate quantization,
        aggressive decimation, and floating distractor triangles.
    """
    rng = np.random.default_rng(seed)

    # 0. Unpack mesh input
    if faces is not None:
        v = np.asarray(clean_mesh, dtype=np.float64).copy()
        f = np.asarray(faces, dtype=np.int64).copy()
    elif isinstance(clean_mesh, (tuple, list)) and len(clean_mesh) == 2:
        v = np.asarray(clean_mesh[0], dtype=np.float64).copy()
        f = np.asarray(clean_mesh[1], dtype=np.int64).copy()
    elif isinstance(clean_mesh, trimesh.Trimesh):
        v = np.asarray(clean_mesh.vertices, dtype=np.float64).copy()
        f = np.asarray(clean_mesh.faces, dtype=np.int64).copy()
    else:
        raise TypeError(f"Unsupported clean_mesh type: {type(clean_mesh)}")

    if len(v) == 0 or len(f) == 0:
        raise ValueError("Cannot corrupt empty mesh geometry.")

    # 1. Random Patch Hole Deletion (10%–25% of faces around 2–4 seed vertices)
    n_faces = len(f)
    if n_faces >= 20:
        hole_ratio = rng.uniform(0.10, 0.25)
        target_deleted_faces = max(1, int(n_faces * hole_ratio))
        n_seeds = int(rng.integers(2, 5))  # 2, 3, or 4 seed vertices

        # Build vertex to face adjacency map
        vf_map = [[] for _ in range(len(v))]
        for face_idx, face in enumerate(f):
            for vert_idx in face:
                vf_map[vert_idx].append(face_idx)

        # Choose seed vertices
        seed_verts = rng.choice(len(v), size=min(n_seeds, len(v)), replace=False).tolist()
        removed_face_set = set()
        visited_verts = set(seed_verts)
        queue = list(seed_verts)

        # Grow patches via BFS
        while queue and len(removed_face_set) < target_deleted_faces:
            curr_v = queue.pop(0)
            for face_idx in vf_map[curr_v]:
                if face_idx not in removed_face_set:
                    removed_face_set.add(face_idx)
                    for nxt_v in f[face_idx]:
                        if nxt_v not in visited_verts:
                            visited_verts.add(nxt_v)
                            queue.append(nxt_v)
                    if len(removed_face_set) >= target_deleted_faces:
                        break

        keep_face_indices = [i for i in range(n_faces) if i not in removed_face_set]
        if len(keep_face_indices) >= 4:
            f = f[keep_face_indices]

    # 2. Aggressive Unconstrained Decimation (300–500 faces)
    target_faces = int(rng.integers(300, 501))
    if len(f) > target_faces:
        try:
            v_dec, f_dec = fast_simplification.simplify(
                v, f, target_count=target_faces, agg=7.0
            )
            v, f = v_dec, f_dec
        except Exception:
            subset = rng.choice(len(f), size=target_faces, replace=False)
            f = f[subset]

    # 3. Isotropic Vertex Jitter N(0, 0.008^2) and 32^3 Coordinate Quantization
    jitter = rng.normal(0, noise_std, size=v.shape)
    v = v + jitter
    # 32^3 quantization: snap coordinates to 32 discrete bins in [-0.5, 0.5]
    v = np.round(v * 32.0) / 32.0

    # Filter out degenerate faces collapsed by quantization
    if len(f) > 0:
        e1 = v[f[:, 1]] - v[f[:, 0]]
        e2 = v[f[:, 2]] - v[f[:, 0]]
        areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        valid_f = areas > 1e-7
        if np.any(valid_f):
            f = f[valid_f]

    # 4. Floating Internal/External Distractor Triangles (3–8 triangles)
    n_distractors = int(rng.integers(3, 9))  # 3 to 8 inclusive
    current_v_count = len(v)

    v_min = v.min(axis=0) if len(v) else np.array([-0.45, -0.45, -0.45])
    v_max = v.max(axis=0) if len(v) else np.array([0.45, 0.45, 0.45])
    distractor_verts = []
    distractor_faces = []

    for i in range(n_distractors):
        if i % 2 == 0:
            # External floating distractor in empty boundary domain
            center = rng.uniform(-0.48, 0.48, size=3)
        else:
            # Internal floating distractor near object core
            center = rng.uniform(v_min * 0.7, v_max * 0.7, size=3)

        # Ensure non-degenerate triangle with guaranteed minimum extent
        offsets = np.array([
            [-0.02, -0.01, 0.0],
            [0.02, -0.01, 0.0],
            [0.0, 0.02, 0.0],
        ]) + rng.normal(0, 0.005, size=(3, 3))
        tri_pts = center + offsets
        base_idx = current_v_count + i * 3

        distractor_verts.extend(tri_pts)
        distractor_faces.append([base_idx, base_idx + 1, base_idx + 2])

    all_v = np.vstack([v, np.array(distractor_verts, dtype=np.float64)])
    all_f = np.vstack([f, np.array(distractor_faces, dtype=np.int64)])

    corrupt_mesh_out = CorruptedMesh(vertices=all_v, faces=all_f, process=False)
    return corrupt_mesh_out
