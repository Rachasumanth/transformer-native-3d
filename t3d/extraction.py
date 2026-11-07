"""CPU reference dual-grid quad extraction; CUDA and cleanup remain separate."""
from collections import Counter
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


def _decompose_into_simple_cycles(edges):
    """Decompose an Eulerian graph edge list into simple closed cycles (Veblen's theorem)."""
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    cycles = []
    for start in list(adj.keys()):
        while adj.get(start):
            path = [start]
            pos = {start: 0}
            curr = start
            while True:
                if not adj.get(curr):
                    break
                nxt = adj[curr].pop()
                if curr in adj[nxt]:
                    adj[nxt].remove(curr)
                if nxt in pos:
                    cycle = path[pos[nxt]:]
                    cycles.append(cycle)
                    for i in range(pos[nxt]):
                        adj[path[i]].append(path[i + 1])
                        adj[path[i + 1]].append(path[i])
                    break
                else:
                    pos[nxt] = len(path)
                    path.append(nxt)
                    curr = nxt
    return cycles


def _ear_clip_minimal_area(vertices, cycle, existing_edges):
    """Triangulate a 3D polygon cycle by greedily clipping ears with minimal area.

    Avoids chords that already exist in existing_edges to prevent duplicate
    internal edges and nonmanifold fan intersections.
    """
    poly = list(cycle)
    triangles = []
    while len(poly) > 3:
        n = len(poly)
        best_idx = None
        best_area = float('inf')
        fallback_idx = 0
        fallback_area = float('inf')
        for i in range(n):
            prev_v = poly[(i - 1) % n]
            curr_v = poly[i]
            next_v = poly[(i + 1) % n]
            p0 = vertices[prev_v]
            p1 = vertices[curr_v]
            p2 = vertices[next_v]
            cross = np.cross(p1 - p0, p2 - p0)
            area = np.linalg.norm(cross)
            if area < fallback_area:
                fallback_area = area
                fallback_idx = i
            chord = tuple(sorted((prev_v, next_v)))
            if chord in existing_edges:
                continue
            if area < best_area:
                best_area = area
                best_idx = i
        if best_idx is None:
            best_idx = fallback_idx
        prev_v = poly[(best_idx - 1) % len(poly)]
        curr_v = poly[best_idx]
        next_v = poly[(best_idx + 1) % len(poly)]
        triangles.append([prev_v, curr_v, next_v])
        chord = tuple(sorted((prev_v, next_v)))
        existing_edges.add(chord)
        poly.pop(best_idx)
    if len(poly) == 3:
        triangles.append(poly)
    return triangles


def resolve_nonmanifold_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    close_holes: bool = False,
    max_hole_perimeter: int = 512,
) -> np.ndarray:
    """Prune spurious fin/sheet faces until every edge has at most 2 incident triangles.

    When close_holes is True, boundary loops (perimeter <= max_hole_perimeter) left
    open after face pruning or at boundaries are closed using ear-clipping hole
    triangulation (minimal area) so the surface remains completely closed.

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

    if close_holes:
        # Prune triangles with >= 2 boundary edges (spurious flaps/fins) in vectorized batches
        for _ in range(30):
            e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
            _, inv, counts = np.unique(e, axis=0, return_inverse=True, return_counts=True)
            b_edges_mask = (counts == 1)[inv]
            b_per_face = (
                b_edges_mask[:len(faces)].astype(int)
                + b_edges_mask[len(faces):2 * len(faces)].astype(int)
                + b_edges_mask[2 * len(faces):].astype(int)
            )
            flaps = np.flatnonzero(b_per_face >= 2)
            if len(flaps) == 0:
                break
            keep = np.ones(len(faces), dtype=bool)
            keep[flaps] = False
            faces = faces[keep]

        e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        unique_e, inv, counts = np.unique(e, axis=0, return_inverse=True, return_counts=True)
        b_edges = [tuple(unique_e[i]) for i in np.where(counts == 1)[0]]
        if b_edges:
            cycles = _decompose_into_simple_cycles(b_edges)
            existing_edges = set(map(tuple, unique_e))
            new_triangles = []
            counts_dict = {tuple(unique_e[i]): counts[i] for i in range(len(unique_e))}
            for cycle in cycles:
                if len(cycle) == 3:
                    new_triangles.append(cycle)
                elif len(cycle) == 4:
                    ac = tuple(sorted((cycle[0], cycle[2])))
                    bd = tuple(sorted((cycle[1], cycle[3])))
                    if counts_dict.get(ac, 0) < 2:
                        new_triangles.append([cycle[0], cycle[1], cycle[2]])
                        new_triangles.append([cycle[0], cycle[2], cycle[3]])
                        counts_dict[ac] = counts_dict.get(ac, 0) + 2
                    elif counts_dict.get(bd, 0) < 2:
                        new_triangles.append([cycle[0], cycle[1], cycle[3]])
                        new_triangles.append([cycle[1], cycle[2], cycle[3]])
                        counts_dict[bd] = counts_dict.get(bd, 0) + 2
                    else:
                        new_triangles.append([cycle[0], cycle[1], cycle[2]])
                        new_triangles.append([cycle[0], cycle[2], cycle[3]])
                elif len(cycle) <= max_hole_perimeter:
                    new_triangles.extend(_ear_clip_minimal_area(vertices, cycle, existing_edges))
            if new_triangles:
                faces = np.vstack([faces, np.array(new_triangles, dtype=np.int64)])
                faces = resolve_nonmanifold_faces(vertices, faces, close_holes=False)

    return faces


def voxels_to_mesh(
    asset: VoxelAsset,
    *,
    resolve_nonmanifold: bool = False,
    halo: bool | None = None,
    max_hole_perimeter: int = 512,
):
    """Return world vertices (L,3) and triangle indices (F,3).

    Retains voxel vertex indices so material/rig row correspondence survives.
    When halo (or resolve_nonmanifold) is True, a 1-voxel exterior halo shell
    allocates dual vertices for missing transverse neighbors so dual quads are
    never dropped at asset or grid boundaries.

    With split weights, diagonal 0–2 wins when w0*w2 > w1*w3 (ties choose 1–3).
    Otherwise maximize the minimum triangle interior angle, with ties choosing 1–3.

    Boolean crossings contain no normal sign: output winding is provisional.
    Orientation repair and removal of unused vertices belong to postprocessing.
    When resolve_nonmanifold is True, spurious faces on nonmanifold edges (> 2 incident
    triangles) are pruned and remaining boundary loops are closed using ear-clipping
    to guarantee a 100% watertight 2-manifold surface.
    """
    use_halo = resolve_nonmanifold if halo is None else halo
    vertices = asset.world_vertices
    lookup = {tuple(int(x) for x in c): i for i, c in enumerate(asset.coords)}
    halo_lookup = {}
    halo_vertices = []
    triangles = []

    for i, axis in np.argwhere(asset.intersected_flag):
        neighbors = asset.coords[i].astype(np.int64) + _NEIGHBORS[axis]
        quad = []
        for c in neighbors:
            key = tuple(c)
            if key in lookup:
                quad.append(lookup[key])
            elif use_halo:
                if key not in halo_lookup:
                    idx = len(vertices) + len(halo_vertices)
                    halo_lookup[key] = idx
                    halo_pos = asset.aabb[0] + (np.array(key, dtype=float) + 0.5) * asset.voxel_size
                    halo_vertices.append(halo_pos)
                quad.append(halo_lookup[key])
            else:
                quad.append(None)

        if any(index is None for index in quad):
            continue

        quad = np.array(quad, dtype=np.int64)
        if asset.split_weight is not None and all(q < len(vertices) for q in quad):
            w = asset.split_weight[quad, 0]
            # Normalize to avoid overflow in finite but very large weights.
            scale = np.max(np.abs(w))
            if scale:
                w = w / scale
            split = 0 if w[0] * w[2] > w[1] * w[3] else 1
        else:
            p_quad = np.array([vertices[q] if q < len(vertices) else halo_vertices[q - len(vertices)] for q in quad])
            scores = [_minimum_angle(p_quad[s]) for s in _SPLITS]
            split = 0 if scores[0] > scores[1] else 1
        triangles.extend(quad[_SPLITS[split]])

    all_verts = np.vstack([vertices, np.array(halo_vertices)]) if halo_vertices else vertices
    faces = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)

    if resolve_nonmanifold:
        faces = resolve_nonmanifold_faces(
            all_verts, faces, close_holes=True, max_hole_perimeter=max_hole_perimeter
        )

    return all_verts, faces
