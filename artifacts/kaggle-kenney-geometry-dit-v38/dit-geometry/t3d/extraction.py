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


def voxels_to_mesh(asset: VoxelAsset):
    """Return world vertices (L,3) and triangle indices (F,3).

    Retains voxel vertex indices so material/rig row correspondence survives.
    Missing-neighbor quads are skipped. With split weights, diagonal 0–2 wins
    when w0*w2 > w1*w3 (ties choose 1–3). Otherwise maximize the minimum triangle
    interior angle, with ties choosing 1–3. This angle rule follows the blueprint;
    it does not reproduce the upstream wrapper's normal-dot heuristic.

    Boolean crossings contain no normal sign: output winding is provisional.
    Orientation repair and removal of unused vertices belong to postprocessing.
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
    return vertices, np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
