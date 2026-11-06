"""Independently audit constrained token streams and sampled target geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from t3d.bpt import decode, quantize
from tools.benchmark_bpt_decimation import fast_surface_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--samples', type=int, default=256)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError('Positive sample count required')
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'artifacts/bpt-training-pilot/manifest.json').read_text())
    report = json.loads((args.output / 'bpt-constraints-report.json').read_text())
    expected = {a['asset']: a for a in manifest['assets'] if a['status'] == 'accepted'}
    if set(report['cases']) != set(expected):
        raise ValueError('Incomplete or unexpected evaluation asset set')
    results = {}
    for name, asset in expected.items():
        token_path = args.output / f'{name}-constrained-tokens.npy'
        tokens = np.load(token_path)
        claim = report['cases'][name]['constrained']
        if len(tokens) != claim['geometry_tokens']:
            raise ValueError('Reported token count differs from artifact')
        entry = dict(tokens=len(tokens), token_sha256=hashlib.sha256(token_path.read_bytes()).hexdigest(),
                     stop_reason=claim['stop_reason'])
        try:
            vertices, faces = decode(tokens)
            if not len(faces):
                raise ValueError('No generated faces')
        except ValueError as exc:
            if claim['decodable']:
                raise RuntimeError('Reported decodable stream fails independent decode') from exc
            entry.update(decodable=False, reason=str(exc))
            results[name] = entry
            continue
        if not claim['decodable']:
            raise RuntimeError('Reported invalid stream independently decodes')
        saved = np.load(args.output / f'{name}-constrained-mesh.npz')
        if not np.array_equal(vertices, saved['vertices']) or not np.array_equal(faces, saved['faces']):
            raise ValueError('Saved mesh differs from decoded token stream')
        q = quantize(vertices)
        area = np.cross(q[faces[:, 1]] - q[faces[:, 0]], q[faces[:, 2]] - q[faces[:, 0]])
        entry.update(decodable=True, vertices=len(vertices), faces=len(faces),
            zero_area_faces=int(np.count_nonzero(np.all(area == 0, axis=1))),
            duplicate_unoriented_faces=int(len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0))))
        sample = asset['samples'][0]
        reference_path = root / 'artifacts/bpt-training-pilot' / sample['file']
        if hashlib.sha256(reference_path.read_bytes()).hexdigest() != sample['sha256']:
            raise ValueError('Reference fixture hash mismatch')
        reference = np.load(reference_path)
        entry['sampled_target_comparison'] = [fast_surface_comparison(
            reference['vertices'], reference['faces'], vertices, faces,
            samples=args.samples, seed=seed) for seed in (0, 1, 2)]
        results[name] = entry
        print(name, entry, flush=True)
    audit = dict(source_sha256=report['source_sha256'], checkpoint_sha256=report['checkpoint_sha256'],
        scope='Independent decode, saved mesh parity, topology and sampled distance to cleaned quantized target',
        samples_per_direction=args.samples, seeds=[0, 1, 2], fscore_tolerance=.01,
        limitations='Sampling does not prove all-surface or sharp-feature fidelity; context-limit outputs are incomplete regardless of decodability',
        cases=results)
    (args.output / 'independent-generation-audit.json').write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__':
    main()
