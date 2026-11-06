"""Prepare licensed, asset-disjoint BPT data with explicit cleanup rejection."""
import hashlib
import json
from pathlib import Path

import numpy as np

from t3d.bpt import encode, decode, quantize
from tools.benchmark_bpt_quantized_cleanup import (
    load_candidate, filter_degenerate_faces, oriented_face_multiset, edge_topology)


def points_on_mesh(vertices, faces, seed):
    rng = np.random.default_rng(seed)
    t = vertices[faces]
    area = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    t = t[rng.choice(len(t), 4096, p=area / area.sum())]
    uv = rng.random((4096, 2))
    uv[uv.sum(1) > 1] = 1 - uv[uv.sum(1) > 1]
    return (t[:, 0] + uv[:, :1] * (t[:, 1] - t[:, 0]) +
            uv[:, 1:] * (t[:, 2] - t[:, 0])).astype(np.float32)


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'artifacts/bpt-training-pilot'
    out.mkdir(exist_ok=True)
    sources = [
        ('Avocado', 'validation', 'artifacts/native-inputs/Avocado.glb', 'artifacts/import/avocado/SOURCE.md'),
        ('Box', 'control', 'artifacts/bpt-candidates/Box/Box.glb', 'artifacts/bpt-candidates/Box/LICENSE.md'),
        ('Duck', 'train', 'artifacts/bpt-candidates/Duck/Duck.glb', 'artifacts/bpt-candidates/Duck/LICENSE.md'),
        ('CesiumMilkTruck', 'train', 'artifacts/bpt-candidates/CesiumMilkTruck/CesiumMilkTruck.glb', 'artifacts/bpt-candidates/CesiumMilkTruck/LICENSE.md'),
        ('FoodLime', 'train', 'artifacts/bpt-candidates/PolyHaven_food_lime_01/food_lime_01_1k.gltf', 'artifacts/bpt-candidates/PolyHaven_food_lime_01/LICENSE.md'),
    ]
    rows = []
    for name, split, source, notice in sources:
        try:
            v, f, digest = load_candidate(root, name, root / source)
        except ValueError as exc:
            rows.append(dict(asset=name, split=split, source=source,
                source_sha256=hashlib.sha256((root / source).read_bytes()).hexdigest(),
                notice=notice, status='rejected', reason=str(exc), samples=[]))
            print(name, 'rejected', str(exc), flush=True)
            continue
        clean, details = filter_degenerate_faces(v, f, all_rotations=True)
        loss = 1 - len(clean) / len(f)
        row = dict(asset=name, split=split, source=source, source_sha256=digest,
            notice=notice, source_faces=len(f), retained_faces=len(clean),
            removed_fraction=loss, max_removed_fraction=.05, samples=[])
        if loss > .05:
            row.update(status='rejected', reason='quantized face loss exceeds 5%')
        else:
            row['status'] = 'accepted'
            rotated = v.copy()
            for turn in range(4):
                if turn:
                    rotated = rotated[:, [2, 1, 0]] * [1, 1, -1]
                tokens = encode(rotated, clean)
                dv, df = decode(tokens)
                if oriented_face_multiset(quantize(rotated), clean) != oriented_face_multiset(quantize(dv), df):
                    raise RuntimeError('BPT round trip changed retained oriented faces')
                filename = f'{name}-y{turn}.npz'
                path = out / filename
                np.savez_compressed(path, tokens=tokens, points=points_on_mesh(dv, df, turn),
                                    vertices=dv, faces=df,
                                    retained_source_face_indices=details['retained_source_face_indices'])
                row['samples'].append(dict(file=filename, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    quarter_turns_y=turn, tokens=len(tokens), faces=len(df), topology=edge_topology(df)))
        rows.append(row)
        print(name, row['status'], 'removed fraction', loss, flush=True)
    manifest = dict(scope='Small licensed data integration pilot; not a representative training corpus',
        split_policy='Fixed by source asset before augmentation; Avocado validation, Box simple control, other accepted assets training',
        cleanup_policy='Reject over 5% face deletion; otherwise explicitly filter 128-bin degeneracies/duplicates over 24 cube rotations',
        augmentation='Four proper Y-axis quarter turns per accepted source; condition points match each decoded mesh',
        assets=rows)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
