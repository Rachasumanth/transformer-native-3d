"""Measure exploratory BPT compression, empirical PTME and vertex locality."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from t3d.bpt import _coordinate, decode, encode
from t3d.import_glb import load_glb
from t3d.normalization import normalize_asset


def emitted_vertices(tokens):
    """Return every vertex occurrence in token order, including repeats."""
    result = []
    block = None
    for raw in tokens:
        token = int(raw)
        if token < 512:
            block = token
        elif token < 1024:
            block = token - 512
        else:
            if block is None:
                raise ValueError('Offset appears before a block token')
            q = np.asarray(_coordinate(block, token - 1024), dtype=np.float64)
            result.append((q + .5) / 128 - .5)
    return np.asarray(result, dtype=np.float64)


def avd(sequence, offsets=(5, 25, 50)):
    values = {}
    for t in offsets:
        if len(sequence) <= t:
            values[str(t)] = None
            continue
        distances = []
        for i in range(t, len(sequence)):
            prior = sequence[i - t:i]
            distances.append(np.sqrt(np.sum((prior - sequence[i]) ** 2, axis=1)).mean())
        values[str(t)] = float(np.mean(distances))
    return values


def entropy_bits(tokens, raw_token_count):
    counts = np.asarray(list(Counter(map(int, tokens)).values()), dtype=np.float64)
    probabilities = counts / len(tokens)
    return float(np.sum(-counts / raw_token_count * np.log2(probabilities)))


def main():
    root = Path(__file__).resolve().parents[1]
    artifact_dir = root / 'artifacts/bpt'
    grid_tokens = np.load(artifact_dir / 'grid_tokens.npy')
    grid_meta = json.loads((artifact_dir / 'grid_report.json').read_text())
    ico_tokens = np.load(artifact_dir / 'icosphere_tokens.npy')
    ico_meta = json.loads((artifact_dir / 'icosphere_report.json').read_text())
    avocado = np.load(artifact_dir / 'cleaned_Avocado_all_24_rotations.npz')
    lime = np.load(artifact_dir / 'cleaned_PolyHaven_food_lime_01_all_24_rotations.npz')
    cleanup_report = json.loads((artifact_dir / 'quantized-cleanup-audit.json').read_text())
    cleanup_sources = {row['asset']: row['source_faces'] for row in cleanup_report['cases']
                       if row['cleanup_mode'] == 'all_24_rotations'}

    box = load_glb(root / 'artifacts/bpt-candidates/Box/Box.glb')
    box_vertices = np.clip(normalize_asset(box.vertices)[0], -.5, .5)
    box_tokens = encode(box_vertices, box.faces)
    rows = []
    sequences = []
    cases = [
        ('Box', box_tokens, len(box.faces), len(box.faces), 'CC-BY 4.0 Khronos simple source control'),
        ('planar-grid', grid_tokens, grid_meta['faces'], grid_meta['faces'], 'synthetic open-surface fixture'),
        ('icosphere', ico_tokens, ico_meta['faces'], ico_meta['faces'], 'synthetic closed-surface fixture'),
        ('Avocado', avocado['tokens'], len(avocado['faces']), cleanup_sources['Avocado'],
         'CC0 Khronos source after removing one 128-bin-degenerate face, safe under 24 rotations'),
        ('PolyHaven_food_lime_01', lime['tokens'], len(lime['faces']), cleanup_sources['PolyHaven_food_lime_01'],
         'CC0 Poly Haven source after explicit 24-rotation quantization cleanup'),
    ]
    for name, tokens, faces, source_faces, provenance in cases:
        raw_count = 9 * int(faces)
        vertex_sequence = emitted_vertices(tokens)
        decoded_vertices, decoded_faces = decode(tokens)
        if len(decoded_faces) != faces:
            raise ValueError(f'{name}: decode returned a different face count')
        row = dict(asset=name, provenance=provenance, source_faces_before_cleanup=int(source_faces),
                   faces=int(faces), faces_removed=int(source_faces-faces),
                   tokens=int(len(tokens)), raw_coordinate_tokens=raw_count,
                   compression_ratio=float(len(tokens) / raw_count),
                   sequence_reduction=float(1 - len(tokens) / raw_count),
                   empirical_ptme_bits_per_raw_coordinate=entropy_bits(tokens, raw_count),
                   avd_mean_distance_previous_vertices_by_context=avd(vertex_sequence),
                   emitted_vertex_occurrences=len(vertex_sequence),
                   unique_decoded_vertices=len(decoded_vertices),
                   bpt_patch_count=int(np.count_nonzero(tokens < 512)))
        rows.append(row)
        sequences.append((tokens, raw_count))

    all_tokens = np.concatenate([tokens for tokens, _ in sequences])
    all_raw = sum(raw for _, raw in sequences)
    combined = dict(case_count=len(sequences),
                    empirical_ptme_bits_per_raw_coordinate=entropy_bits(all_tokens, all_raw),
                    compression_ratio=float(len(all_tokens) / all_raw),
                    provenance='exploratory mixed fixtures; not a representative corpus')
    report = dict(scope='Track B exploratory locality and entropy; local BPT profile',
        definitions=dict(compression_ratio='BPT token count / (9 * source face count)',
            ptme='sum over token types of -count/raw-coordinate-count * log2(empirical token probability); bits per raw coordinate',
            avd='mean, over emitted vertex occurrences with a full t-vertex history, of Euclidean distances to all previous t emitted vertex occurrences; dequantized coordinates in [-0.5, 0.5]'),
        cases=rows, mixed_fixture_aggregate=combined,
        fragmentation_note='Center-token patch counts are reported; comparison with EdgeRunner subsequence counts is not validated.')
    (artifact_dir / 'locality-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
