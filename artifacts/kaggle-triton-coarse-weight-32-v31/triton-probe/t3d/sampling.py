"""Step 1 surface conditioning, following Dora section 3.2.1.

CPU reference implementation. Boundary and non-manifold edges are excluded from
interior-edge dihedral detection. Face winding must be consistent. Sharp sample
normals use one incident face (lowest index), retaining a valid surface normal
rather than smoothing across creases. No-sharp-edge meshes return an empty
salient set, as specified by Dora equation 7.
"""
import operator
import numpy as np


def _count(n):
    n = operator.index(n)
    if n < 0:
        raise ValueError('Sample count must be nonnegative')
    return n


def _mesh(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError('Vertices must be finite (V, 3)')
    if f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer):
        raise ValueError('Faces must be integer (F, 3)')
    if not len(f) or np.any(f < 0) or np.any(f >= len(v)):
        raise ValueError('Nonempty faces must reference valid vertices')
    triangles = v[f]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError('Degenerate or numerically invalid triangles')
    return v, f, cross / lengths[:, None], lengths * .5


def _edges(faces, normals, threshold):
    if not np.isfinite(threshold) or not 0 <= threshold <= np.pi:
        raise ValueError('Threshold must be radians in [0, pi]')
    incidence = {}
    for i, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            incidence.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
    edges, owners = [], []
    for edge, adjacent in sorted(incidence.items()):
        if len(adjacent) == 2:
            cosine = np.clip(normals[adjacent[0]] @ normals[adjacent[1]], -1, 1)
            if np.arccos(cosine) > threshold:
                edges.append(edge)
                owners.append(adjacent[0])
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2), np.asarray(owners, dtype=np.int64)


def salient_edges(vertices, faces, *, threshold=np.pi / 6):
    """Return undirected index pairs whose interior dihedral exceeds threshold."""
    _, f, normals, _ = _mesh(vertices, faces)
    return _edges(f, normals, threshold)[0]


def farthest_point_indices(points, count):
    """Deterministic O(N*count) FPS with O(N) workspace; first point is index 0."""
    p = np.asarray(points, dtype=np.float64)
    count = _count(count)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or count > len(p):
        raise ValueError('FPS needs finite (N, 3) points and count <= N')
    chosen = np.empty(count, dtype=np.int64)
    distances = np.full(len(p), np.inf)
    available = np.ones(len(p), dtype=bool)
    index = 0
    for i in range(count):
        chosen[i] = index
        available[index] = False
        distances = np.minimum(distances, np.sum((p - p[index]) ** 2, axis=1))
        index = int(np.argmax(np.where(available, distances, -1)))
    return chosen


def sample_conditioning(vertices, faces, *, uniform_count=32768, salient_count=32768,
                        threshold=np.pi / 6, seed=0):
    """Return separate uniform and salient arrays of xyz + unit normal, (N, 6).

    Uniform samples use area-weighted triangles and uniform barycentrics. Salient
    vertices use FPS when abundant; otherwise retain all and interpolate evenly
    along edges. Remainder samples go to the first edges in sorted index order.
    """
    uniform_count, salient_count = _count(uniform_count), _count(salient_count)
    v, f, normals, areas = _mesh(vertices, faces)
    rng = np.random.default_rng(seed)
    owners = rng.choice(len(f), size=uniform_count, p=areas / areas.sum())
    uv = rng.random((uniform_count, 2))
    root = np.sqrt(uv[:, 0])
    bary = np.column_stack((1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]))
    xyz = np.einsum('ni,nij->nj', bary, v[f[owners]])
    uniform = np.column_stack((xyz, normals[owners]))
    edges, edge_faces = _edges(f, normals, threshold)
    if not len(edges) or salient_count == 0:
        return uniform, np.empty((0, 6), dtype=np.float64)
    vertex_ids = np.unique(edges)
    # Each endpoint uses the first sharp incident face for a reproducible normal.
    owner_map = {}
    for edge, face_id in zip(edges, edge_faces):
        for vertex in edge:
            owner_map.setdefault(vertex, face_id)
    sharp = np.column_stack((v[vertex_ids], normals[[owner_map[i] for i in vertex_ids]]))
    if salient_count <= len(sharp):
        return uniform, sharp[farthest_point_indices(sharp[:, :3], salient_count)]
    extra = salient_count - len(sharp)
    quotient, remainder = divmod(extra, len(edges))
    pieces = [sharp]
    for i, (a, b) in enumerate(edges):
        n = quotient + (i < remainder)
        if n:
            t = (np.arange(1, n + 1) / (n + 1))[:, None]
            points = v[a] * (1 - t) + v[b] * t
            pieces.append(np.column_stack((points, np.tile(normals[edge_faces[i]], (n, 1)))))
    return uniform, np.concatenate(pieces)
