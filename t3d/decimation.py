"""QEM Decimation and Tangent Space Normal Map Baker for production 3D assets."""
from __future__ import annotations

from collections import defaultdict, deque
import numpy as np
import fast_simplification
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt


def collapse_degenerate_faces(
    points: np.ndarray,
    faces: np.ndarray,
    min_area: float = 1e-8,
    max_aspect_ratio: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Eliminate zero-area and high-aspect-ratio needle triangles by edge collapse."""
    p = np.array(points, dtype=np.float64, copy=True)
    tri = np.array(faces, dtype=np.int64, copy=True)
    for _ in range(30):
        valid = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 2] != tri[:, 0])
        tri = tri[valid]
        p0 = p[tri[:, 0]]
        p1 = p[tri[:, 1]]
        p2 = p[tri[:, 2]]
        cross = np.cross(p1 - p0, p2 - p0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        e0 = np.linalg.norm(p1 - p0, axis=1)
        e1 = np.linalg.norm(p2 - p1, axis=1)
        e2 = np.linalg.norm(p0 - p2, axis=1)
        l_max = np.maximum(np.maximum(e0, e1), e2)
        h_min = np.where(l_max > 1e-12, 2.0 * areas / np.maximum(l_max, 1e-12), 0.0)
        aspect = np.where(h_min > 1e-12, l_max / h_min, np.inf)

        bad = np.flatnonzero((areas <= min_area) | (aspect > max_aspect_ratio))
        if len(bad) == 0:
            break
        bad = bad[np.argsort(-aspect[bad])]
        for idx in bad:
            t = tri[idx]
            v0, v1, v2 = t[0], t[1], t[2]
            if v0 == v1 or v1 == v2 or v2 == v0:
                continue
            d01 = np.linalg.norm(p[v0] - p[v1])
            d12 = np.linalg.norm(p[v1] - p[v2])
            d20 = np.linalg.norm(p[v2] - p[v0])
            dists = [d01, d12, d20]
            min_e = np.argmin(dists)
            if min_e == 0:
                p[v1] = p[v0]
                tri[tri == v1] = v0
            elif min_e == 1:
                p[v2] = p[v1]
                tri[tri == v2] = v1
            else:
                p[v0] = p[v2]
                tri[tri == v0] = v2
    valid = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 2] != tri[:, 0])
    tri = tri[valid]
    sorted_faces = np.sort(tri, axis=1)
    _, unique_indices = np.unique(sorted_faces, axis=0, return_index=True)
    tri = tri[np.sort(unique_indices)]
    used = np.unique(tri)
    inv = np.full(len(p), -1, dtype=np.int64)
    inv[used] = np.arange(len(used))
    return p[used], inv[tri]


def split_non_manifold_vertices(positions: np.ndarray, faces: np.ndarray, tol: float = 2e-4) -> tuple[np.ndarray, np.ndarray]:
    """Resolve hourglass pinch vertices by decoupling disjoint 1-ring star components."""
    from collections import defaultdict
    p = list(positions)
    f = np.array(faces, dtype=np.int64, copy=True)
    _, unique_indices, inv = np.unique(np.round(positions, decimals=6), axis=0, return_index=True, return_inverse=True)
    wf = inv[f]
    v_stars = defaultdict(list)
    v_faces = defaultdict(list)
    for fi, tri in enumerate(wf):
        if tri[0] == tri[1] or tri[1] == tri[2] or tri[2] == tri[0]:
            continue
        v_stars[int(tri[0])].append((int(tri[1]), int(tri[2])))
        v_stars[int(tri[1])].append((int(tri[2]), int(tri[0])))
        v_stars[int(tri[2])].append((int(tri[0]), int(tri[1])))
        v_faces[int(tri[0])].append((fi, 0))
        v_faces[int(tri[1])].append((fi, 1))
        v_faces[int(tri[2])].append((fi, 2))

    for v_id, opp in v_stars.items():
        adj = defaultdict(set)
        nodes = set()
        for a, b in opp:
            adj[a].add(b)
            adj[b].add(a)
            nodes.add(a)
            nodes.add(b)
        visited = set()
        components = []
        for node in nodes:
            if node not in visited:
                comp = set()
                q = [node]
                visited.add(node)
                comp.add(node)
                while q:
                    curr = q.pop()
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            comp.add(nbr)
                            q.append(nbr)
                components.append(comp)
        if len(components) > 1:
            orig_v = unique_indices[v_id]
            base_pos = np.array(p[orig_v])
            for ci, comp in enumerate(components[1:], start=1):
                new_idx = len(p)
                for fi, corner in v_faces[v_id]:
                    tri = wf[fi]
                    other1 = tri[(corner + 1) % 3]
                    other2 = tri[(corner + 2) % 3]
                    if other1 in comp or other2 in comp:
                        f[fi, corner] = new_idx
                offset = tol * ci * np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
                p.append(base_pos + offset)

    clean_p = np.asarray(p, dtype=np.float64)
    used = np.unique(f)
    if len(used) < len(clean_p):
        inv2 = np.full(len(clean_p), -1, dtype=np.int64)
        inv2[used] = np.arange(len(used))
        clean_p = clean_p[used]
        f = inv2[f]
    return clean_p, f


def decimate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int = 5000,
    aggressive: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """QEM Decimation preserving feature edges (>30 deg dihedral) and watertight manifold topology.

    Parameters
    ----------
    vertices : array-like of shape (V, 3)
        Input mesh vertex positions.
    faces : array-like of shape (F, 3)
        Input mesh triangle face indices.
    target_faces : int, default 5000
        Target face count for simplification.
    aggressive : bool, default False
        If True, use more aggressive decimation parameter (agg=7.0-10.0).
        If False, prioritize geometric preservation and feature edges (agg=3.0-5.0).

    Returns
    -------
    clean_v : np.ndarray of shape (V_out, 3)
        Decimated mesh vertices.
    clean_f : np.ndarray of shape (F_out, 3)
        Decimated mesh triangle faces.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError("Vertices must be finite (V, 3)")
    if f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer):
        raise ValueError("Faces must be integer (F, 3)")
    if len(f) == 0:
        return v, f
    if (f < 0).any() or (f >= len(v)).any():
        raise ValueError("Face indices out of range")
    if target_faces <= 0:
        raise ValueError("target_faces must be > 0")

    if len(f) <= target_faces:
        clean_v, clean_f = collapse_degenerate_faces(v, f, min_area=1e-8, max_aspect_ratio=40.0)
        from .extraction import resolve_nonmanifold_faces
        clean_f = resolve_nonmanifold_faces(clean_v, clean_f, close_holes=True)
        clean_v, clean_f = collapse_degenerate_faces(clean_v, clean_f, min_area=1e-8, max_aspect_ratio=40.0)
        clean_f = resolve_nonmanifold_faces(clean_v, clean_f, close_holes=True)
        clean_v, clean_f = split_non_manifold_vertices(clean_v, clean_f)
        return clean_v, clean_f

    # Determine aggressiveness schedule
    if aggressive:
        aggs = [7.0, 8.0, 10.0]
    else:
        # Conservative QEM preservation of dihedral feature edges (> 30 deg)
        aggs = [3.0, 4.0, 5.0, 7.0]

    out_v, out_f = None, None
    for agg in aggs:
        points, triangles = fast_simplification.simplify(
            v, f, target_count=target_faces, agg=agg, preserve_border=True
        )
        out_v = np.asarray(points, dtype=np.float64)
        out_f = np.asarray(triangles, dtype=np.int64)
        if len(out_f) <= int(target_faces * 1.05):
            break

    # Clean unreferenced vertices
    used_indices = np.unique(out_f)
    index_map = np.full(len(out_v), -1, dtype=np.int64)
    index_map[used_indices] = np.arange(len(used_indices))
    clean_v = out_v[used_indices]
    clean_f = index_map[out_f]

    # Collapse degenerate faces and remove duplicates
    clean_v, clean_f = collapse_degenerate_faces(clean_v, clean_f)

    # Multi-pass hole and boundary closure to guarantee watertight 2-manifold surface
    from .extraction import resolve_nonmanifold_faces
    for _ in range(3):
        clean_f = resolve_nonmanifold_faces(clean_v, clean_f, close_holes=True)

    # Split any non-manifold vertices (pinch points / hourglass apexes)
    clean_v, clean_f = split_non_manifold_vertices(clean_v, clean_f)

    # Final pass to eliminate any needle triangles introduced by ear-clipping hole closure
    clean_v, clean_f = collapse_degenerate_faces(clean_v, clean_f, min_area=1e-8, max_aspect_ratio=40.0)
    clean_f = resolve_nonmanifold_faces(clean_v, clean_f, close_holes=True)

    # Final split to ensure any pinch points introduced by needle collapse are resolved
    clean_v, clean_f = split_non_manifold_vertices(clean_v, clean_f)

    return clean_v, clean_f


def bake_tangent_normal_map(
    high_v: np.ndarray,
    high_f: np.ndarray,
    low_v: np.ndarray,
    low_f: np.ndarray,
    low_uvs: np.ndarray,
    texture_size: int = 1024,
    pad_margin: int = 2,
    as_uint8: bool = False,
) -> np.ndarray:
    """Project normals from high-res mesh onto low-poly decimated mesh UV texture atlas.

    Preserves geometric surface details from dual contouring / sculpting high-poly meshes
    by baking tangent-space normal deviations into a standard RGB normal map.

    Parameters
    ----------
    high_v : array-like of shape (V_high, 3)
        High-res mesh vertex positions.
    high_f : array-like of shape (F_high, 3)
        High-res mesh triangle face indices.
    low_v : array-like of shape (V_low, 3)
        Low-poly mesh vertex positions.
    low_f : array-like of shape (F_low, 3)
        Low-poly mesh triangle face indices.
    low_uvs : array-like of shape (V_low, 2) or (F_low, 3, 2)
        Low-poly mesh UV texture coordinates in [0, 1]^2.
    texture_size : int, default 1024
        Square texture resolution (width and height in pixels).
    pad_margin : int, default 2
        Pixel margin to dilate valid texels into gutter to avoid texture filtering seams.
    as_uint8 : bool, default False
        If True, return uint8 array in [0, 255]. Otherwise float32 in [0, 1].

    Returns
    -------
    normal_map : np.ndarray of shape (texture_size, texture_size, 3)
        Baked tangent-space normal map. Neutral normal (0, 0, 1) encodes to (0.5, 0.5, 1.0).
    """
    hv = np.asarray(high_v, dtype=np.float64)
    hf = np.asarray(high_f, dtype=np.int64)
    lv = np.asarray(low_v, dtype=np.float64)
    lf = np.asarray(low_f, dtype=np.int64)
    uvs = np.asarray(low_uvs, dtype=np.float64)

    if hv.ndim != 2 or hv.shape[1] != 3 or not len(hf):
        raise ValueError("Invalid high-res geometry")
    if lv.ndim != 2 or lv.shape[1] != 3 or not len(lf):
        raise ValueError("Invalid low-poly geometry")
    if uvs.ndim == 2:
        if len(uvs) < len(lv):
            raise ValueError("UV coordinate count mismatch with vertices")
        uv_tris = uvs[lf]
    elif uvs.ndim == 3 and uvs.shape[1:] == (3, 2):
        if len(uvs) != len(lf):
            raise ValueError("Per-face UV count mismatch with faces")
        uv_tris = uvs
    else:
        raise ValueError("low_uvs must have shape (V, 2) or (F, 3, 2)")

    # 1. High-res surface normals and KD-Tree
    e1 = hv[hf[:, 1]] - hv[hf[:, 0]]
    e2 = hv[hf[:, 2]] - hv[hf[:, 0]]
    cross_h = np.cross(e1, e2)
    cross_len = np.linalg.norm(cross_h, axis=1, keepdims=True)
    high_fn = cross_h / np.maximum(cross_len, 1e-12)

    high_vn = np.zeros_like(hv)
    for i in range(3):
        np.add.at(high_vn, hf[:, i], cross_h)
    high_vn /= np.maximum(np.linalg.norm(high_vn, axis=1, keepdims=True), 1e-12)

    # Densify high-res surface points for closest surface projection
    centroids = np.mean(hv[hf], axis=1)
    edge_mids = np.vstack([
        0.5 * (hv[hf[:, 0]] + hv[hf[:, 1]]),
        0.5 * (hv[hf[:, 1]] + hv[hf[:, 2]]),
        0.5 * (hv[hf[:, 2]] + hv[hf[:, 0]]),
    ])
    edge_norms = np.vstack([
        0.5 * (high_vn[hf[:, 0]] + high_vn[hf[:, 1]]),
        0.5 * (high_vn[hf[:, 1]] + high_vn[hf[:, 2]]),
        0.5 * (high_vn[hf[:, 2]] + high_vn[hf[:, 0]]),
    ])
    edge_norms /= np.maximum(np.linalg.norm(edge_norms, axis=1, keepdims=True), 1e-12)

    high_sample_pts = np.vstack([hv, centroids, edge_mids])
    high_sample_norms = np.vstack([high_vn, high_fn, edge_norms])
    high_tree = cKDTree(high_sample_pts)

    # 2. Low-poly TBN frame per triangle
    v_tris = lv[lf]  # (F, 3, 3)
    dp1 = v_tris[:, 1] - v_tris[:, 0]
    dp2 = v_tris[:, 2] - v_tris[:, 0]
    duv1 = uv_tris[:, 1] - uv_tris[:, 0]
    duv2 = uv_tris[:, 2] - uv_tris[:, 0]

    det = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
    valid_det = np.abs(det) > 1e-10
    det_safe = np.where(valid_det, det, 1.0)[:, None]

    T = (dp1 * duv2[:, 1:2] - dp2 * duv1[:, 1:2]) / det_safe
    B = (dp2 * duv1[:, 0:1] - dp1 * duv2[:, 0:1]) / det_safe

    cross_l = np.cross(dp1, dp2)
    cross_l_len = np.linalg.norm(cross_l, axis=1, keepdims=True)
    N = cross_l / np.maximum(cross_l_len, 1e-12)

    # Gram-Schmidt orthogonalize T
    T = T - np.sum(T * N, axis=1, keepdims=True) * N
    T_len = np.linalg.norm(T, axis=1, keepdims=True)
    # Default tangent if degenerate
    default_T = np.where(np.abs(N[:, 0:1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    default_T = default_T - np.sum(default_T * N, axis=1, keepdims=True) * N
    default_T /= np.maximum(np.linalg.norm(default_T, axis=1, keepdims=True), 1e-12)

    T = np.where(T_len > 1e-8, T / np.maximum(T_len, 1e-12), default_T)
    B = np.cross(N, T)

    # 3. Rasterize low-poly UV triangles
    im = Image.new("I", (texture_size, texture_size), 0)
    draw = ImageDraw.Draw(im)
    uv_px = uv_tris * texture_size
    for i in range(len(lf)):
        poly = [tuple(p) for p in uv_px[i]]
        draw.polygon(poly, fill=i + 1)

    face_map = np.array(im)
    mask = face_map > 0

    normal_map = np.full((texture_size, texture_size, 3), [0.5, 0.5, 1.0], dtype=np.float32)
    if not mask.any():
        if as_uint8:
            return (np.clip(normal_map, 0, 1) * 255).astype(np.uint8)
        return normal_map

    ys, xs = np.where(mask)
    faces_idx = face_map[ys, xs] - 1

    # Barycentric coordinates for texel centers
    u_p = (xs + 0.5) / texture_size
    v_p = (ys + 0.5) / texture_size
    P_uv = np.column_stack([u_p, v_p])

    A_uv = uv_tris[faces_idx, 0]
    B_uv = uv_tris[faces_idx, 1]
    C_uv = uv_tris[faces_idx, 2]

    v0 = B_uv - A_uv
    v1 = C_uv - A_uv
    v2 = P_uv - A_uv

    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)

    denom = d00 * d11 - d01 * d01
    denom_safe = np.where(np.abs(denom) > 1e-12, denom, 1.0)
    w1 = np.clip((d11 * d20 - d01 * d21) / denom_safe, 0.0, 1.0)
    w2 = np.clip((d00 * d21 - d01 * d20) / denom_safe, 0.0, 1.0)
    w0 = np.clip(1.0 - w1 - w2, 0.0, 1.0)
    w_sum = np.maximum(w0 + w1 + w2, 1e-12)
    w0 /= w_sum
    w1 /= w_sum
    w2 /= w_sum

    # Interpolate 3D position on low-poly surface
    P_3d = (
        w0[:, None] * v_tris[faces_idx, 0]
        + w1[:, None] * v_tris[faces_idx, 1]
        + w2[:, None] * v_tris[faces_idx, 2]
    )

    # 4. Query closest high-res surface point and its normal
    _, closest_idx = high_tree.query(P_3d, workers=-1)
    N_high = high_sample_norms[closest_idx]

    # 5. Transform into low-poly tangent space
    T_p = T[faces_idx]
    B_p = B[faces_idx]
    N_p = N[faces_idx]

    N_tx = np.sum(N_high * T_p, axis=1)
    N_ty = np.sum(N_high * B_p, axis=1)
    N_tz = np.sum(N_high * N_p, axis=1)

    tangent_normal = np.column_stack([N_tx, N_ty, N_tz])
    tangent_len = np.linalg.norm(tangent_normal, axis=1, keepdims=True)
    tangent_normal /= np.maximum(tangent_len, 1e-12)

    # Encode to RGB [0, 1]
    normal_map[ys, xs] = 0.5 * tangent_normal + 0.5

    # 6. Apply gutter padding around islands if requested
    if pad_margin > 0 and not mask.all():
        dist, (iy, ix) = distance_transform_edt(~mask, return_indices=True)
        pad_mask = (~mask) & (dist <= pad_margin)
        normal_map[pad_mask] = normal_map[iy[pad_mask], ix[pad_mask]]

    if as_uint8:
        return np.rint(np.clip(normal_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    return normal_map.astype(np.float32)


def generate_uv_charts(
    vertices: np.ndarray,
    faces: np.ndarray,
    texture_size: int = 1024,
    margin: int = 8,
    gutter: int = 8,
    normal_cos_thresh: float = 0.5,
    min_chart_faces: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Segment mesh into contiguous developable UV charts, unwrap, and pack with gutters.

    Groups connected triangles into contiguous UV charts via normal flood fill with
    small-chart merging (preventing triangle-soup fragmentation), unfolds each chart
    onto an orthonormal projection plane, and packs charts into [0, 1]^2 with guaranteed
    border margins and inter-island gutters (>= 2 px).

    Returns:
        render_vertices: (V_render, 3) float32 array (duplicated only across chart seams)
        render_faces: (F, 3) int64 array referencing render_vertices
        render_uvs: (V_render, 2) float32 array in [0, 1]^2
        vertex_sources: (V_render,) int64 array mapping render_vertices to original vertices
    """
    v_arr = np.asarray(vertices, dtype=np.float32)
    f_arr = np.asarray(faces, dtype=np.int64)
    V = len(v_arr)
    F = len(f_arr)
    if F == 0 or V == 0:
        return v_arr, f_arr, np.zeros((V, 2), dtype=np.float32), np.arange(V, dtype=np.int64)

    tri = v_arr[f_arr]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(cross, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    fnorms = cross / norm

    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(f_arr):
        for i in range(3):
            e = tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
            edge_to_faces[e].append(fi)

    adj = defaultdict(list)
    for fi_list in edge_to_faces.values():
        if len(fi_list) == 2:
            adj[fi_list[0]].append(fi_list[1])
            adj[fi_list[1]].append(fi_list[0])

    visited = np.zeros(F, dtype=bool)
    face_chart = np.full(F, -1, dtype=int)
    chart_id = 0
    for start in range(F):
        if visited[start]:
            continue
        visited[start] = True
        face_chart[start] = chart_id
        q = deque([start])
        seed_norm = fnorms[start]
        while q:
            curr = q.popleft()
            for nbr in adj[curr]:
                if not visited[nbr]:
                    if np.dot(fnorms[nbr], seed_norm) >= normal_cos_thresh:
                        visited[nbr] = True
                        face_chart[nbr] = chart_id
                        q.append(nbr)
        chart_id += 1

    counts = np.bincount(face_chart, minlength=chart_id)
    small_charts = np.where(counts < min_chart_faces)[0]
    for sc in small_charts:
        sc_faces = np.where(face_chart == sc)[0]
        nbr_charts = []
        for fi in sc_faces:
            for nbr in adj[fi]:
                c_nbr = face_chart[nbr]
                if c_nbr != sc:
                    nbr_charts.append(c_nbr)
        if nbr_charts:
            best_c = max(set(nbr_charts), key=nbr_charts.count)
            face_chart[sc_faces] = best_c

    unique_charts, inv_c = np.unique(face_chart, return_inverse=True)
    face_chart = inv_c
    num_charts = len(unique_charts)

    chart_rects = []
    chart_uv_local = {}
    for cid in range(num_charts):
        c_faces = np.where(face_chart == cid)[0]
        c_norm = fnorms[c_faces].sum(axis=0)
        cn_len = np.linalg.norm(c_norm)
        c_norm = np.array([0, 0, 1.0]) if cn_len < 1e-6 else c_norm / cn_len

        if abs(c_norm[0]) < 0.9:
            u_axis = np.cross(c_norm, [1.0, 0.0, 0.0])
        else:
            u_axis = np.cross(c_norm, [0.0, 1.0, 0.0])
        u_axis /= np.linalg.norm(u_axis)
        v_axis = np.cross(c_norm, u_axis)

        c_v_indices = np.unique(f_arr[c_faces])
        c_pts = v_arr[c_v_indices]
        u_coords = c_pts @ u_axis
        v_coords = c_pts @ v_axis
        u_min, u_max = u_coords.min(), u_coords.max()
        v_min, v_max = v_coords.min(), v_coords.max()
        w = max(u_max - u_min, 1e-5)
        h = max(v_max - v_min, 1e-5)
        for vi, u, v in zip(c_v_indices, u_coords, v_coords):
            chart_uv_local[(cid, vi)] = ((u - u_min) / w, (v - v_min) / h)
        chart_rects.append((cid, w, h))

    usable = texture_size - 2 * margin
    total_area = sum(w * h for _, w, h in chart_rects)
    fill_ratio = 0.65

    for attempt in range(8):
        scale = np.sqrt(fill_ratio * (usable ** 2) / max(1e-8, total_area))
        for _, w, h in chart_rects:
            if w * scale > usable:
                scale = usable / w
            if h * scale > usable:
                scale = usable / h

        scaled_rects = []
        for cid, w, h in chart_rects:
            pw = max(4, min(usable, int(np.ceil(w * scale))))
            ph = max(4, min(usable, int(np.ceil(h * scale))))
            scaled_rects.append((cid, pw, ph))

        order = sorted(range(num_charts), key=lambda i: scaled_rects[i][2], reverse=True)
        chart_pos = {}
        curr_x = margin
        curr_y = margin
        shelf_h = 0
        overflow = False
        for idx in order:
            cid, pw, ph = scaled_rects[idx]
            if curr_x + pw > texture_size - margin:
                curr_x = margin
                curr_y += shelf_h + gutter
                shelf_h = 0
            if curr_x + pw > texture_size - margin or curr_y + ph > texture_size - margin:
                overflow = True
                break
            chart_pos[cid] = (curr_x, curr_y, pw, ph)
            curr_x += pw + gutter
            shelf_h = max(shelf_h, ph)
        if not overflow:
            break
        fill_ratio *= 0.60

    if len(chart_pos) < num_charts:
        # Fallback grid placement to ensure all charts are packed within [0, 1]
        grid_dim = int(np.ceil(np.sqrt(num_charts)))
        cell_size = (usable - (grid_dim - 1) * gutter) // max(1, grid_dim)
        cell_size = max(4, cell_size)
        chart_pos = {}
        for cid in range(num_charts):
            row = cid // grid_dim
            col = cid % grid_dim
            cx = margin + col * (cell_size + gutter)
            cy = margin + row * (cell_size + gutter)
            chart_pos[cid] = (cx, cy, cell_size, cell_size)

    vert_map = {}
    render_verts = []
    render_uvs = []
    render_faces = []
    vertex_sources = []
    for fi, face in enumerate(f_arr):
        cid = face_chart[fi]
        cx, cy, pw, ph = chart_pos[cid]
        face_new = []
        for vi in face:
            key = (cid, vi)
            if key not in vert_map:
                vert_map[key] = len(render_verts)
                render_verts.append(v_arr[vi])
                vertex_sources.append(vi)
                lu, lv = chart_uv_local[key]
                u = np.clip((cx + lu * pw) / texture_size, 0.0, 1.0)
                v = np.clip((cy + lv * ph) / texture_size, 0.0, 1.0)
                render_uvs.append([u, v])
            face_new.append(vert_map[key])
        render_faces.append(face_new)

    return (
        np.array(render_verts, dtype=np.float32),
        np.array(render_faces, dtype=f_arr.dtype),
        np.array(render_uvs, dtype=np.float32),
        np.array(vertex_sources, dtype=np.int64),
    )
