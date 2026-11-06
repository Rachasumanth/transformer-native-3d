"""Reproduce a clean, synthetic closed-surface BPT tokenizer fixture."""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import trimesh

from t3d.bpt import decode, encode, quantize, quantization_diagnostics


def oriented_faces(q, faces):
    result = []
    for face in np.asarray(faces):
        coords = [tuple(row) for row in q[face]]
        result.append(min(tuple(coords[i:] + coords[:i]) for i in range(3)))
    return Counter(result)


def edge_topology(faces):
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return dict(boundary_edges=int(np.count_nonzero(counts == 1)),
                nonmanifold_edges=int(np.count_nonzero(counts > 2)))


def main():
    mesh = trimesh.creation.icosphere(subdivisions=5, radius=0.5)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    diagnostics = quantization_diagnostics(vertices, faces, bins=128)
    if not diagnostics['eligible']:
        raise ValueError(f'Synthetic icosphere is not eligible: {diagnostics}')

    started = time.perf_counter()
    tokens = encode(vertices, faces)
    elapsed = time.perf_counter() - started
    decoded_vertices, decoded_faces = decode(tokens)
    preserved = (oriented_faces(quantize(vertices), faces) ==
                 oriented_faces(quantize(decoded_vertices), decoded_faces))
    if len(decoded_faces) != len(faces) or not preserved:
        raise ValueError('BPT round-trip changed oriented quantized icosphere faces')
    report = dict(fixture='trimesh icosphere, subdivisions=5, radius=0.5',
        provenance='Synthetic tokenizer fixture; not an artist asset or curated corpus',
        vertices=len(vertices), faces=len(faces), tokens=len(tokens),
        tokens_per_face=len(tokens) / len(faces),
        fraction_of_raw_9_tokens=len(tokens) / (9 * len(faces)),
        sequence_reduction=1 - len(tokens) / (9 * len(faces)),
        patches=int(np.count_nonzero(tokens < 512)), encode_seconds=elapsed,
        quantization=diagnostics,
        oriented_quantized_faces_preserved=preserved,
        decoded_vertices=len(decoded_vertices), decoded_faces=len(decoded_faces),
        source_edge_topology=edge_topology(faces),
        decoded_edge_topology=edge_topology(decoded_faces))
    out = Path(__file__).resolve().parents[1] / 'artifacts/bpt'
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'icosphere_tokens.npy', tokens)
    (out / 'icosphere_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
