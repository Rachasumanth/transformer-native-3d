"""CPU reference dual-grid quad extraction; CUDA and cleanup remain separate."""
import numpy as np
from .schema import VoxelAsset

# Positive transverse neighbors around the marked primal edge.
_NEIGHBORS = np.array([
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
])
_SPLITS = np.array([[[0, 1, 2], [0, 2, 3]], [[0, 1, 3], [3, 1, 2]]])


def _minimum_angle(triangles):
    """Smallest interior angle across a pair of triangles; degenerate = -inf."""
    angles = []
    for tri in triangles:
        cross = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if np.linalg.norm(cross) == 0:
            return -np.inf
        for i in range(3):
            a, b = tri[(i + 1) % 3] - tri[i], tri[(i + 2) % 3] - tri[i]
            angles.append(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b))
    return min(angles)


def resolve_nonmanifold_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Prune spurious fin/sheet faces until every edge has at most 2 incident triangles.

    Returns cleaned faces array (K, 3) retaining original vertex indices.
    """
    if len(faces) == 0:
        return faces
    faces = np.asarray(faces, dtype=np.int64).copy()
    tri_pts = vertices[faces]
    cross = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    while True:
        num_faces = len(faces)
        e0 = np.sort(faces[:, [0, 1]], axis=1)
        e1 = np.sort(faces[:, [1, 2]], axis=1)
        e2 = np.sort(faces[:, [2, 0]], axis=1)
        edges = np.vstack([e0, e1, e2])
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        sorted_edges = edges[order]
        change = np.ones(len(sorted_edges), dtype=bool)
        if len(sorted_edges) > 1:
            change[1:] = np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1)
        starts = np.flatnonzero(change)
        counts = np.diff(np.append(starts, len(sorted_edges)))
        edge_ids_sorted = np.cumsum(change) - 1
        inv = np.empty(len(edges), dtype=np.int64)
        inv[order] = edge_ids_sorted
        bad_mask = counts > 2
        if not np.any(bad_mask):
            break
        face_bad_counts = np.zeros(num_faces, dtype=np.int32)
        for offset in (0, num_faces, 2 * num_faces):
            edge_inv = inv[offset:offset + num_faces]
            face_bad_counts += bad_mask[edge_inv]
        candidates = np.flatnonzero(face_bad_counts > 0)
        worst_idx = candidates[np.lexsort((areas[candidates], -face_bad_counts[candidates]))[0]]
        faces = np.delete(faces, worst_idx, axis=0)
        areas = np.delete(areas, worst_idx, axis=0)
    return faces


def voxels_to_mesh(asset: VoxelAsset, *, resolve_nonmanifold: bool = False):
    """Return world vertices (L,3) and triangle indices (F,3).

    Retains voxel vertex indices so material/rig row correspondence survives.
    Missing-neighbor quads are skipped. With split weights, diagonal 0–2 wins
    when w0*w2 > w1*w3 (ties choose 1–3). Otherwise maximize the minimum triangle
    interior angle, with ties choosing 1–3. This angle rule follows the blueprint;
    it does not reproduce the upstream wrapper's normal-dot heuristic.

    Boolean crossings contain no normal sign: output winding is provisional.
    Orientation repair and removal of unused vertices belong to postprocessing.
    When resolve_nonmanifold is True, spurious faces on nonmanifold edges (> 2 incident
    triangles) are pruned to guarantee a valid 2-manifold surface.
    """
    vertices = asset.world_vertices
    lookup = {tuple(int(x) for x in c): i for i, c in enumerate(asset.coords)}
    triangles = []
    for i, axis in np.argwhere(asset.intersected_flag):
        neighbors = asset.coords[i].astype(np.int64) + _NEIGHBORS[axis]
        quad = [lookup.get(tuple(c)) for c in neighbors]
        if any(index is None for index in quad):
            continue
        quad = np.array(quad, dtype=np.int64)
        if asset.split_weight is not None:
            w = asset.split_weight[quad, 0]
            # Normalize to avoid overflow in finite but very large weights.
            scale = np.max(np.abs(w))
            if scale:
                w = w / scale
            split = 0 if w[0] * w[2] > w[1] * w[3] else 1
        else:
            scores = [_minimum_angle(vertices[quad[s]]) for s in _SPLITS]
            split = 0 if scores[0] > scores[1] else 1
        triangles.extend(quad[_SPLITS[split]])
    faces = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if resolve_nonmanifold:
        faces = resolve_nonmanifold_faces(vertices, faces)
    return vertices, faces
