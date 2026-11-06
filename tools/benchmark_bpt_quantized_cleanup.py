"""Measure explicit removal of quantization-degenerate faces as data filtering."""
from collections import Counter
import hashlib
import json
from itertools import permutations, product
from pathlib import Path

import numpy as np

from t3d.bpt import decode, encode, quantize, quantization_diagnostics
from t3d.import_glb import load_glb
from t3d.normalization import normalize_asset
from tools.audit_bpt_corpus import cube_rotation_quantization, load_geometry_only_glb
from tools.benchmark_bpt_decimation import fast_surface_comparison


def load_candidate(root, name, path):
    if path.suffix.lower() == '.gltf':
        vertices, faces, digest = load_geometry_only_glb(path)
    else:
        try:
            asset = load_glb(path)
            vertices, faces = asset.vertices, asset.faces
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except ValueError:
            vertices, faces, digest = load_geometry_only_glb(path)
    vertices = np.clip(normalize_asset(vertices)[0], -.5, .5)
    return vertices, np.asarray(faces, dtype=np.int64), digest


def filter_degenerate_faces(vertices, faces, *, all_rotations=False):
    q = quantize(vertices)
    bad = np.zeros(len(faces), dtype=bool)
    per_orientation_counts = []
    rotations = [np.eye(3, dtype=int)]
    if all_rotations:
        rotations = []
        for permutation in permutations(range(3)):
            basis = np.eye(3, dtype=int)[list(permutation)]
            for signs in product((-1, 1), repeat=3):
                rotation = np.diag(signs) @ basis
                if round(np.linalg.det(rotation)) == 1:
                    rotations.append(rotation)
    for rotation in rotations:
        qr = quantize(vertices @ rotation.T)
        triangles = qr[faces]
        collapsed = ((triangles[:, 0] == triangles[:, 1]).all(1) |
                     (triangles[:, 1] == triangles[:, 2]).all(1) |
                     (triangles[:, 2] == triangles[:, 0]).all(1))
        cross = np.cross(triangles[:, 1] - triangles[:, 0],
                         triangles[:, 2] - triangles[:, 0])
        collinear = np.all(cross == 0, axis=1) & ~collapsed
        bad |= collapsed | collinear
        per_orientation_counts.append(dict(collapsed=int(collapsed.sum()),
                                           collinear=int(collinear.sum())))
    keep = ~bad
    nondegenerate_ids = np.flatnonzero(keep)
    unordered = np.sort(triangles[keep], axis=1)
    _, first = np.unique(unordered, axis=0, return_index=True)
    duplicate_ids = np.ones(len(unordered), dtype=bool)
    duplicate_ids[first] = False
    kept_ids = nondegenerate_ids[~duplicate_ids]
    return faces[kept_ids], dict(bad_faces_any_orientation=int(bad.sum()),
        duplicate_quantized_faces=int(duplicate_ids.sum()),
        orientations_checked=len(rotations),
        per_orientation_degenerate_counts=per_orientation_counts,
        retained_source_face_indices=kept_ids)


def edge_topology(faces):
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return dict(boundary_edges=int(np.count_nonzero(counts == 1)),
                nonmanifold_edges=int(np.count_nonzero(counts > 2)))


def oriented_face_multiset(q, faces):
    result = []
    for face in q[np.asarray(faces)]:
        coords = [tuple(row) for row in face]
        result.append(min(tuple(coords[i:] + coords[:i]) for i in range(3)))
    return Counter(result)


def main():
    root = Path(__file__).resolve().parents[1]
    inputs = [
        ('Avocado', root / 'artifacts/native-inputs/Avocado.glb'),
        ('Box', root / 'artifacts/bpt-candidates/Box/Box.glb'),
        ('PolyHaven_food_lime_01', root / 'artifacts/bpt-candidates/PolyHaven_food_lime_01/food_lime_01_1k.gltf'),
    ]
    cases = []
    for name, path in inputs:
        vertices, faces, digest = load_candidate(root, name, path)
        for mode, all_rotations in (('default_orientation', False), ('all_24_rotations', True)):
            clean_faces, removed = filter_degenerate_faces(vertices, faces,
                                                          all_rotations=all_rotations)
            diagnostics = quantization_diagnostics(vertices, clean_faces, bins=128)
            _, vertex_remap = np.unique(quantize(vertices), axis=0, return_inverse=True)
            quantized_retained_faces = vertex_remap[clean_faces]
            tokens = encode(vertices, clean_faces)
            decoded_vertices, decoded_faces = decode(tokens)
            preserved = (oriented_face_multiset(quantize(vertices), clean_faces) ==
                         oriented_face_multiset(quantize(decoded_vertices), decoded_faces))
            if not preserved or len(decoded_faces) != len(clean_faces):
                raise ValueError(f'{name}: cleaned BPT round-trip changed retained oriented faces')
            metrics = fast_surface_comparison(vertices, faces, decoded_vertices,
                                              decoded_faces, samples=1024, seed=0)
            artifact = root/'artifacts/bpt'/f'cleaned_{name}_{mode}.npz'
            np.savez_compressed(artifact, vertices=vertices, faces=clean_faces,
                                tokens=tokens,
                                retained_source_face_indices=removed['retained_source_face_indices'])
            row = dict(asset=name, cleanup_mode=mode, source_sha256=digest,
                source_faces=len(faces), retained_faces=len(clean_faces),
                removed_faces=len(faces)-len(clean_faces),
                removed_fraction=float((len(faces)-len(clean_faces))/len(faces)),
                removal_breakdown={k:v for k,v in removed.items()
                                   if k not in ('retained_source_face_indices','per_orientation_degenerate_counts')},
                per_orientation_degenerate_counts=removed['per_orientation_degenerate_counts'],
                cleaned_quantization=diagnostics,
                cube_rotation_preflight=cube_rotation_quantization(vertices, clean_faces),
                tokens=len(tokens), compression_ratio=float(len(tokens)/(9*len(faces))),
                sequence_reduction=float(1-len(tokens)/(9*len(faces))),
                bpt_patches=int(np.count_nonzero(tokens < 512)),
                retained_oriented_faces_roundtrip=preserved,
                reproducible_data_artifact=str(artifact.relative_to(root)),
                source_index_edge_topology=edge_topology(faces),
                quantized_retained_edge_topology=edge_topology(quantized_retained_faces),
                decoded_edge_topology=edge_topology(decoded_faces),
                sampled_surface_metrics=metrics)
            cases.append(row)
            print(name,mode,{k:v for k,v in row.items() if k not in ('source_sha256','removal_breakdown','cleaned_quantization','per_orientation_degenerate_counts')},flush=True)
    report=dict(scope='Explicit quantized-degenerate/duplicate face filtering; no in-place source modification',
        filter='drop faces whose 128-bin quantized triangle has repeated vertices, zero quantized area, or duplicates an earlier unordered quantized face; measure default orientation and union over all 24 proper cube rotations',
        surface_metric='1024 area-weighted samples each direction, exact CPU triangle BVH, tolerance 0.01; exploratory only',
        cases=cases)
    out=root/'artifacts/bpt/quantized-cleanup-audit.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
