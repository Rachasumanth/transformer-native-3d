"""CPU reference mesh-to-dual-grid conversion by direct triangle/cell clipping.

No SDF, flood fill, or largest-component filtering. Candidate enumeration is
per triangle bounding box and is intentionally a correctness reference, not the
production accelerator. Coincident/coplanar grid edges have no unique crossing
and are not marked. Multiple crossings of one edge collapse to one boolean;
multiple disconnected patches in a cell share one QEF vertex.
"""
from itertools import product
import numpy as np
from .sampling import _mesh
from .schema import VoxelAsset
from .qef import solve_qef


def _clip(points, lo, hi):
    poly = [p.copy() for p in points]
    for axis in range(3):
        for bound, sign in ((lo[axis], 1), (hi[axis], -1)):
            if not poly:
                return []
            result = []
            previous = poly[-1]
            d0 = sign * (previous[axis] - bound)
            for current in poly:
                d1 = sign * (current[axis] - bound)
                if (d0 >= 0) != (d1 >= 0):
                    result.append(previous + (current - previous) * (d0 / (d0 - d1)))
                if d1 >= 0:
                    result.append(current)
                previous, d0 = current, d1
            poly = result
    return poly


def _segment_clip(a, b, lo, hi):
    t0, t1 = 0., 1.
    delta = b - a
    for axis in range(3):
        if delta[axis] == 0:
            if a[axis] < lo[axis] or a[axis] > hi[axis]:
                return None
        else:
            u, v = sorted(((lo[axis] - a[axis]) / delta[axis],
                           (hi[axis] - a[axis]) / delta[axis]))
            t0, t1 = max(t0, u), min(t1, v)
            if t0 > t1:
                return None
    return a + .5 * (t0 + t1) * delta


def _crosses(triangle, normal, cell, axis):
    # Edge shared by the cell and its positive transverse neighbors.
    start = cell.astype(float) + 1
    start[axis] -= 1
    if abs(normal[axis]) < 1e-14:
        return False
    t = float(normal @ (triangle[0] - start) / normal[axis])
    if not -1e-12 <= t <= 1 + 1e-12:
        return False
    point = start.copy()
    point[axis] += t
    # Winding-independent oriented half-plane tests on the triangle.
    for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
        if np.dot(np.cross(b - a, point - a), normal) < -1e-10:
            return False
    return True


def mesh_to_voxels(vertices, faces, *, grid_size, aabb,
                   face_weight=1., boundary_weight=1., regularization_weight=.1):
    """Convert a triangle mesh to a canonically ordered geometry-only VoxelAsset.

    Vertices must be inside the explicit AABB. Mesh boundary edges add an
    in-surface plane perpendicular to the edge, helping retain open boundaries.
    Grid-space normals are computed after scaling, supporting anisotropic cells.
    """
    v, f, _, _ = _mesh(vertices, faces)
    # Reuse schema validation for grid and bounds before doing geometric work.
    empty = VoxelAsset(np.empty((0, 3), int), np.empty((0, 3)),
                       np.empty((0, 3), bool), grid_size, aabb)
    grid = np.array(empty.grid_size)
    if np.any(v < empty.aabb[0]) or np.any(v > empty.aabb[1]):
        raise ValueError('Mesh vertices must lie inside the AABB')
    weights = np.asarray([face_weight, boundary_weight, regularization_weight])
    if not np.isfinite(weights).all() or np.any(weights < 0) or face_weight <= 0:
        raise ValueError('Invalid QEF weights')
    positions = (v - empty.aabb[0]) / empty.voxel_size
    triangles = positions[f]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    incidence = {}
    for i, face in enumerate(f):
        for a, b in zip(face, np.roll(face, -1)):
            incidence.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
    boundaries = [[] for _ in f]
    for (a, b), owners in incidence.items():
        if len(owners) == 1:
            boundaries[owners[0]].append((positions[a], positions[b]))
    cells = {}
    for triangle, normal, boundary in zip(triangles, normals, boundaries):
        # Include both cells on integer planes; only retain actual intersections.
        low = np.maximum(0, np.ceil(triangle.min(axis=0)).astype(int) - 1)
        high = np.minimum(grid - 1, np.floor(triangle.max(axis=0)).astype(int))
        for key in product(*(range(low[a], high[a] + 1) for a in range(3))):
            cell = np.array(key)
            clipped = _clip(triangle, cell, cell + 1)
            if not clipped:
                continue
            data = cells.setdefault(key, [[], [], [], [], np.zeros(3, bool)])
            local = np.clip(np.asarray(clipped) - cell, 0, 1)
            # One plane sample per clipped polygon vertex.
            data[0].extend(local)
            data[1].extend([normal] * len(local))
            for a, b in boundary:
                midpoint = _segment_clip(a, b, cell, cell + 1)
                if midpoint is not None:
                    bn = np.cross(b - a, normal)
                    bn /= np.linalg.norm(bn)
                    data[2].append(np.clip(midpoint - cell, 0, 1))
                    data[3].append(bn)
            data[4] |= [_crosses(triangle, normal, cell, axis) for axis in range(3)]
    if not cells:
        return empty
    coords, dual, flags = [], [], []
    for key, (p, n, bp, bn, flag) in sorted(cells.items()):
        kwargs = dict(boundary_points=bp, boundary_normals=bn) if bp else {}
        dual.append(solve_qef(p, n, face_weight=face_weight, boundary_weight=boundary_weight,
                              regularization_weight=regularization_weight, **kwargs))
        coords.append(key)
        flags.append(flag)
    return VoxelAsset(np.array(coords), np.array(dual), np.array(flags),
                       empty.grid_size, empty.aabb).canonical()
