"""Count paper-defined EdgeRunner-style subsequence starts on local fixtures.

This is an independent counter derived from Algorithm 1 in EdgeRunner's paper.
It follows the paper's face/vertex visit rules and counts emitted B markers;
it does not encode coordinates or reproduce the authors' tokenizer runtime.
Boundary edges are treated as already-visited neighboring faces, matching the
paper's initialization of boundary vertices as visited. Nonmanifold edges and
inconsistently oriented adjacent faces are rejected.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from t3d.import_glb import load_glb
from t3d.normalization import normalize_asset
from t3d.bpt import decode, encode


class HalfEdge:
    __slots__ = ('face', 'start', 'end', 'opposite_vertex', 'next_twin', 'prev_twin')

    def __init__(self, face, start, end, opposite_vertex):
        self.face = face
        self.start = start
        self.end = end
        self.opposite_vertex = opposite_vertex
        self.next_twin = None
        self.prev_twin = None


def edge_topology(faces):
    incidences = defaultdict(list)
    for face_id, (a, b, c) in enumerate(np.asarray(faces, dtype=np.int64)):
        for start, end in ((a, b), (b, c), (c, a)):
            incidences[(min(int(start), int(end)), max(int(start), int(end)))].append(
                (face_id, int(start), int(end)))
    return incidences


def count_subsequences(faces, vertex_count=None):
    """Return B-marker count plus topology metadata for an oriented triangle mesh."""
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('faces must have shape (F, 3)')
    if len(faces) == 0:
        return dict(faces=0, subsequences=0, boundary_edges=0,
                    connected_components=0)
    if np.any(faces < 0) or (vertex_count is not None and np.any(faces >= vertex_count)):
        raise ValueError('face index is outside the supplied vertex range')
    if np.any(np.sort(faces, axis=1)[:, 1:] == np.sort(faces, axis=1)[:, :-1]):
        raise ValueError('degenerate face repeats a vertex index')

    incidence = edge_topology(faces)
    for edge, uses in incidence.items():
        if len(uses) > 2:
            raise ValueError(f'nonmanifold edge {edge} has {len(uses)} incident faces')
        if len(uses) == 2 and uses[0][1:] != uses[1][1:][::-1]:
            raise ValueError(f'adjacent faces have inconsistent winding at edge {edge}')

    # For each oriented face create half-edges with next/previous edge twins.
    halfedges = [[None, None, None] for _ in range(len(faces))]
    directed = {}
    for face_id, face in enumerate(faces):
        a, b, c = map(int, face)
        directed[(a, b)] = (face_id, 0)
        directed[(b, c)] = (face_id, 1)
        directed[(c, a)] = (face_id, 2)
        halfedges[face_id] = [HalfEdge(face_id, a, b, c),
                              HalfEdge(face_id, b, c, a),
                              HalfEdge(face_id, c, a, b)]
    for face_id in range(len(faces)):
        for edge_id, halfedge in enumerate(halfedges[face_id]):
            next_id, prev_id = (edge_id + 1) % 3, (edge_id - 1) % 3
            f, e = face_id, next_id
            a, b = map(int, (faces[f, e], faces[f, (e + 1) % 3]))
            halfedge.next_twin = _twin(directed, halfedges, a, b)
            e = prev_id
            a, b = map(int, (faces[f, e], faces[f, (e + 1) % 3]))
            halfedge.prev_twin = _twin(directed, halfedges, a, b)

    boundary_vertices = set()
    boundary_edges = 0
    for (a, b), uses in incidence.items():
        if len(uses) == 1:
            boundary_edges += 1
            boundary_vertices.update((a, b))

    face_visited = np.zeros(len(faces), dtype=bool)
    vertex_visited = set(boundary_vertices)
    starts = 0

    def face_is_visited(halfedge):
        # A missing twin is a boundary, equivalent to the paper's previsited
        # boundary vertices for the purposes of choosing traversal direction.
        return halfedge is None or face_visited[halfedge.face]

    def run_traverse(seed):
        nonlocal starts
        if seed is not None and not face_visited[seed.face]:
            starts += 1
            vertex_visited.update((seed.start, seed.end))
            actions.append(('tokenize', seed))

    actions = []
    for face_id in range(len(faces)):
        if face_visited[face_id]:
            continue
        run_traverse(halfedges[face_id][0])
        while actions:
            kind, current = actions.pop()
            if kind == 'traverse':
                run_traverse(current)
                continue
            face_visited[current.face] = True

            vertex = current.opposite_vertex
            if vertex not in vertex_visited:
                vertex_visited.add(vertex)
                if current.next_twin is not None:
                    actions.append(('tokenize', current.next_twin))
                continue

            next_seen = face_is_visited(current.next_twin)
            prev_seen = face_is_visited(current.prev_twin)
            if next_seen and prev_seen:
                continue
            if next_seen:
                actions.append(('tokenize', current.prev_twin))
            elif prev_seen:
                actions.append(('tokenize', current.next_twin))
            else:
                # The paper completes the next-side recursion before beginning
                # a new subsequence on the previous side.
                actions.append(('traverse', current.prev_twin))
                actions.append(('tokenize', current.next_twin))

    remaining = np.flatnonzero(~face_visited)
    if len(remaining):
        raise RuntimeError(f'traversal left {len(remaining)} faces unvisited')

    # Face-adjacency components are independent of traversal starts.
    adjacency = [[] for _ in range(len(faces))]
    for uses in incidence.values():
        if len(uses) == 2:
            a, b = uses[0][0], uses[1][0]
            adjacency[a].append(b)
            adjacency[b].append(a)
    components = 0
    seen = set()
    for seed in range(len(faces)):
        if seed in seen:
            continue
        components += 1
        stack = [seed]
        seen.add(seed)
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)

    return dict(faces=int(len(faces)), subsequences=int(starts),
                boundary_edges=int(boundary_edges),
                connected_components=int(components))


def _twin(directed, halfedges, start, end):
    found = directed.get((end, start))
    return None if found is None else halfedges[found[0]][found[1]]


def main():
    root = Path(__file__).resolve().parents[1]
    artifact_dir = root / 'artifacts/bpt'
    cleanup_report = json.loads((artifact_dir / 'quantized-cleanup-audit.json').read_text())
    cleaned_face_counts = {
        row['asset']: row['retained_faces'] for row in cleanup_report['cases']
        if row['cleanup_mode'] == 'all_24_rotations'
    }

    box = load_glb(root / 'artifacts/bpt-candidates/Box/Box.glb')
    box_vertices = np.clip(normalize_asset(box.vertices)[0], -.5, .5)
    box_tokens = encode(box_vertices, box.faces)
    box_roundtrip_vertices, box_roundtrip_faces = decode(box_tokens)
    grid_tokens = np.load(artifact_dir / 'grid_tokens.npy')
    grid_roundtrip_vertices, grid_roundtrip_faces = decode(grid_tokens)
    ico_tokens = np.load(artifact_dir / 'icosphere_tokens.npy')
    ico_roundtrip_vertices, ico_roundtrip_faces = decode(ico_tokens)

    cases = [
        ('Khronos Box', box_roundtrip_faces, len(box_roundtrip_vertices), box_tokens,
         'CC-BY 4.0 source after local BPT 128-bin quantization/round-trip'),
        ('planar-grid', grid_roundtrip_faces, len(grid_roundtrip_vertices), grid_tokens,
         'synthetic open surface after local BPT round-trip'),
        ('icosphere', ico_roundtrip_faces, len(ico_roundtrip_vertices), ico_tokens,
         'synthetic closed surface after local BPT round-trip'),
    ]
    for name, filename, asset in [
        ('Avocado', 'cleaned_Avocado_all_24_rotations.npz', 'Avocado'),
        ('Poly Haven Food Lime', 'cleaned_PolyHaven_food_lime_01_all_24_rotations.npz',
         'PolyHaven_food_lime_01'),
    ]:
        bundle = np.load(artifact_dir / filename)
        decoded_vertices, decoded_faces = decode(bundle['tokens'])
        cases.append((name, decoded_faces, len(decoded_vertices), bundle['tokens'],
                      f'cleaned 128-bin source; kept {cleaned_face_counts[asset]} faces'))

    rows = []
    for name, faces, vertices, tokens, provenance in cases:
        try:
            row = count_subsequences(faces, vertices)
            row.update(asset=name, provenance=provenance, status='counted',
                       bpt_patch_count=int(np.count_nonzero(tokens < 512)))
        except ValueError as exc:
            row = dict(asset=name, provenance=provenance, status='rejected',
                       faces=int(len(faces)),
                       bpt_patch_count=int(np.count_nonzero(tokens < 512)),
                       reason=str(exc))
        rows.append(row)

    report = dict(
        scope='Independent, paper-derived count of EdgeRunner B subsequence starts; local fixtures only',
        method='Count each B emitted by the paper appendix traversal: seed unvisited faces in array order, use next/previous adjacent faces, and start a new B traversal on the previous side when both sides are unvisited.',
        boundary_policy='Initialize boundary vertices as visited; treat missing adjacent faces as already visited when deciding traversal direction.',
        limitations=['The authors report 54.7 EdgeRunner and 199.5 AMT subsequences as dataset averages; this local count uses different fixtures and ordering.',
                     'The counter does not emit or validate vertex tokens and is not the authors’ implementation.',
                     'Nonmanifold edges and inconsistent winding are rejected.'],
        reference='EdgeRunner Appendix A.1.1, Algorithm 1 (arXiv:2409.18114v1)',
        cases=rows)
    output = artifact_dir / 'edgerunner-fragmentation-audit.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
