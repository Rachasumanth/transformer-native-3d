"""CPU BPT reference: 128 bins, 8^3 coarse blocks, 16^3 local offsets.

Local vocabulary: center blocks 0..511, peripheral blocks 512..1023,
offsets 1024..5119. No BOS/EOS tokens; container count terminates the stream.
These explicit conventions do not claim compatibility with upstream checkpoints.
"""
import numpy as np
from .sampling import _mesh


def quantize(vertices):
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all() or np.any(v < -.5) or np.any(v > .5):
        raise ValueError('Vertices must be finite (V,3) in [-.5,.5]')
    return np.minimum(np.floor((v+.5)*128), 127).astype(np.int64)


def quantization_diagnostics(vertices, faces, *, bins=128):
    """Report whether uniform coordinate quantization makes any face degenerate.

    This is an ingestion preflight: it never drops faces or rewrites connectivity.
    ``bins=128`` matches the local BPT token vocabulary; other resolutions are
    diagnostic only because the current encoder has a fixed 5120-token grammar.
    """
    v,f,_,_=_mesh(vertices,faces)
    if not isinstance(bins,(int,np.integer)) or bins<2:
        raise ValueError('Quantization bins must be an integer >= 2')
    if np.any(v<-.5) or np.any(v>.5):
        raise ValueError('Vertices must lie in [-.5,.5] for BPT quantization')
    q=np.minimum(np.floor((v+.5)*bins),bins-1).astype(np.int64)
    triangles=q[f]
    collapsed=((triangles[:,0]==triangles[:,1]).all(1)|
               (triangles[:,1]==triangles[:,2]).all(1)|
               (triangles[:,2]==triangles[:,0]).all(1))
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    collinear=np.all(cross==0,axis=1)&~collapsed
    source_cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    source_area2=np.linalg.norm(source_cross,axis=1)
    reconstructed=(q+.5)/bins-.5
    error=v-reconstructed
    return dict(bins=int(bins),source_faces=int(len(f)),
        source_unique_positions=int(len(np.unique(v,axis=0))),
        quantized_unique_positions=int(len(np.unique(q,axis=0))),
        source_zero_area_faces=int(np.count_nonzero(source_area2==0)),
        source_tiny_faces_1e_8=int(np.count_nonzero(source_area2<1e-8)),
        collapsed_faces=int(np.count_nonzero(collapsed)),
        collinear_faces=int(np.count_nonzero(collinear)),
        eligible=not bool(collapsed.any() or collinear.any()),
        max_abs_coordinate_error=float(np.max(np.abs(error))),
        rms_coordinate_error=float(np.sqrt(np.mean(error**2))))


def _indices(q):
    coarse, local = q // 16, q % 16
    return int(coarse @ [64, 8, 1]), int(local @ [256, 16, 1])


def _coordinate(block, offset):
    return tuple(16*np.array([block//64, block//8 % 8, block % 8])
                 + np.array([offset//256, offset//16 % 16, offset % 16]))


def encode(vertices, faces):
    """Emit separator-free fan tokens, preserving oriented quantized triangles.

    Quantized vertices merge by location. Collapsed or duplicate triangles fail
    explicitly. Non-manifold branches/disconnected fans split into more patches.
    Canonical ordering is z/y/x; cyclic face rotations preserve winding.
    """
    v, f, _, _ = _mesh(vertices, faces)
    q, inverse = np.unique(quantize(v), axis=0, return_inverse=True)
    preflight=quantization_diagnostics(v,f,bins=128)
    if preflight['collapsed_faces']:
        n=preflight['collapsed_faces']
        raise ValueError(f"Quantization collapses {n} triangle{'s' if n!=1 else ''}")
    if preflight['collinear_faces']:
        n=preflight['collinear_faces']
        raise ValueError(f"Quantization makes {n} triangle{'s' if n!=1 else ''} collinear")
    order = np.lexsort((q[:, 0], q[:, 1], q[:, 2]))
    remap = np.empty(len(order), dtype=int)
    remap[order] = np.arange(len(order))
    q, f = q[order], remap[inverse[f]]
    canonical = []
    for face in f:
        if len(set(face)) != 3:
            raise ValueError('Quantization collapses a triangle')
        if not np.any(np.cross(q[face[1]]-q[face[0]], q[face[2]]-q[face[0]])):
            raise ValueError('Quantization makes a triangle collinear')
        canonical.append(tuple(np.roll(face, -int(np.argmin(face)))))
    f = np.array(sorted(canonical), dtype=np.int64)
    if len(np.unique(np.sort(f, axis=1), axis=0)) != len(f):
        raise ValueError('Duplicate quantized triangles')
    incident = [set() for _ in q]
    for i, face in enumerate(f):
        for vertex in face:
            incident[vertex].add(i)
    active = set(range(len(f)))
    tokens = []
    for seed in range(len(f)):
        if seed not in active:
            continue
        center = min(f[seed], key=lambda vertex: (-len(incident[vertex]), int(vertex)))
        # Each incident triangle is a directed edge between peripheral vertices.
        edges = {}
        for i in incident[center]:
            rotated = np.roll(f[i], -int(np.flatnonzero(f[i] == center)[0]))
            edges[i] = (int(rotated[1]), int(rotated[2]))
        first, last = edges[seed]
        ring, used = [first, last], {seed}
        while True:
            candidates = [i for i, (a, b) in edges.items() if a == ring[-1] and i not in used]
            if not candidates:
                break
            i = min(candidates)
            used.add(i)
            ring.append(edges[i][1])
        while ring[0] != ring[-1]:
            candidates = [i for i, (a, b) in edges.items() if b == ring[0] and i not in used]
            if not candidates:
                break
            i = min(candidates)
            used.add(i)
            ring.insert(0, edges[i][0])
        block, offset = _indices(q[center])
        tokens.extend([block, 1024+offset])
        previous = block
        for vertex in ring:
            block, offset = _indices(q[vertex])
            if block != previous:
                tokens.append(512+block)
            tokens.append(1024+offset)
            previous = block
        for i in used:
            active.remove(i)
            for vertex in f[i]:
                incident[vertex].remove(i)
    return np.asarray(tokens, dtype=np.uint16)


def decode(tokens):
    """Validate token grammar and return dequantized vertices and triangle indices."""
    t = np.asarray(tokens)
    if t.ndim != 1 or not np.issubdtype(t.dtype, np.integer) or np.any(t < 0) or np.any(t >= 5120):
        raise ValueError('Tokens must be integer vector within vocabulary')
    vertices, faces, lookup = [], [], {}
    patch = []
    block, pending = None, False
    def vertex_id(q):
        if q not in lookup:
            lookup[q] = len(vertices)
            vertices.append(q)
        return lookup[q]
    def finish():
        if not patch:
            return
        if len(patch) < 3:
            raise ValueError('Fan patch needs a center and at least two peripheral vertices')
        for a, b in zip(patch[1:-1], patch[2:]):
            if len({patch[0], a, b}) != 3:
                raise ValueError('Fan contains a degenerate triangle')
            faces.append([patch[0], a, b])
    for token in map(int, t):
        if token < 512:
            if pending:
                raise ValueError('Block token missing its offset')
            finish()
            patch = []
            block, pending = token, True
        elif token < 1024:
            if not patch or pending:
                raise ValueError('Unexpected peripheral block')
            block, pending = token-512, True
        else:
            if block is None:
                raise ValueError('Offset without block')
            patch.append(vertex_id(_coordinate(block, token-1024)))
            pending = False
    if pending:
        raise ValueError('Truncated block/offset pair')
    finish()
    q = np.asarray(vertices, dtype=float).reshape(-1, 3)
    return (q+.5)/128-.5, np.asarray(faces, dtype=np.int64).reshape(-1, 3)
