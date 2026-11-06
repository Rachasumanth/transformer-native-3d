"""CPU mesh orientation without discarding geometry or changing vertex indices."""
from collections import deque
import numpy as np
from .sampling import _mesh


def orient_faces(vertices, faces, *, outward=False):
    """Return consistently wound faces and a per-component audit report.

    Edge-connected components are solved by BFS parity constraints. Non-manifold
    edges, duplicate faces and non-orientable constraints are rejected. Geometry
    and face order are retained; only triangle corners 1 and 2 can be exchanged.
    Open components preserve their first face's winding. With outward=True,
    closed components are individually flipped to positive signed volume. This
    treats shells independently: cavity/nesting semantics require separate
    containment analysis and are not inferred here.
    """
    v, source, _, _ = _mesh(vertices, faces)
    f = source.astype(np.int64, copy=True)
    if len(np.unique(np.sort(f, axis=1), axis=0)) != len(f):
        raise ValueError('Duplicate faces must be cleaned before orientation')
    edges = {}
    for i, tri in enumerate(f):
        for a, b in zip(tri, np.roll(tri, -1)):
            key = tuple(sorted((int(a), int(b))))
            edges.setdefault(key, []).append((i, int(a < b)))
    if any(len(owners) > 2 for owners in edges.values()):
        raise ValueError('Non-manifold edge has more than two incident faces')
    adjacency = [[] for _ in f]
    boundary = np.zeros(len(f), dtype=bool)
    for owners in edges.values():
        if len(owners) == 1:
            boundary[owners[0][0]] = True
        else:
            (a, da), (b, db) = owners
            # Same original edge direction requires exactly one face flipped.
            parity = int(da == db)
            adjacency[a].append((b, parity))
            adjacency[b].append((a, parity))
    flips = np.full(len(f), -1, dtype=np.int8)
    components = []
    for root in range(len(f)):
        if flips[root] != -1:
            continue
        flips[root] = 0
        queue = deque([root])
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor, parity in adjacency[current]:
                expected = int(flips[current]) ^ parity
                if flips[neighbor] == -1:
                    flips[neighbor] = expected
                    queue.append(neighbor)
                elif flips[neighbor] != expected:
                    raise ValueError('Mesh has non-orientable face constraints')
        ids = np.array(component)
        changed = ids[flips[ids] == 1]
        f[changed] = f[changed][:, [0, 2, 1]]
        closed = not bool(boundary[ids].any())
        volume = None
        if closed:
            # Shift origin to the component to avoid cancellation under translation.
            positions = v[f[ids]]
            positions = positions - positions.reshape(-1, 3).mean(axis=0)
            volume = float(np.einsum('ij,ij->i', positions[:, 0],
                                    np.cross(positions[:, 1], positions[:, 2])).sum() / 6)
            if outward and volume == 0:
                raise ValueError('Closed component has zero signed volume; outward is ambiguous')
            if outward and volume < 0:
                f[ids] = f[ids][:, [0, 2, 1]]
                flips[ids] ^= 1
                volume = -volume
        components.append(dict(faces=len(ids), closed=closed, signed_volume=volume,
                               flipped_faces=int(np.count_nonzero(flips[ids]))))
    return f, dict(components=components, flipped_faces=int(np.count_nonzero(flips)),
                   outward_closed_components=bool(outward))
