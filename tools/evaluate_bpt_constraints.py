"""Evaluate grammar masks on the completed multiasset checkpoint, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from t3d.bpt import decode
from t3d.bpt_ar import BPTAutoregressiveTransformer

TRAINING_SOURCE = 'e669da90c8b223552d8037b9b1a073baee50d5c07d98c2b09a612a350800a8b8'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Kaggle CUDA required')
    inputs = Path('/kaggle/input')
    checkpoints = list(inputs.rglob('bpt-multiasset.pt'))
    if len(checkpoints) != 1:
        raise RuntimeError('Expected one mounted multiasset checkpoint')
    path = checkpoints[0]
    checkpoint = torch.load(path, weights_only=True, map_location='cuda')
    if checkpoint['source_sha256'] != TRAINING_SOURCE:
        raise RuntimeError('Checkpoint does not match the completed baseline')
    baseline_path = path.parent / 'bpt-multiasset-report.json'
    baseline = json.loads(baseline_path.read_text())
    if baseline['source_sha256'] != TRAINING_SOURCE:
        raise RuntimeError('Baseline provenance mismatch')
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / 'artifacts/bpt-training-pilot/manifest.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != baseline['manifest_sha256']:
        raise RuntimeError('Evaluation manifest differs from training')
    manifest = json.loads(manifest_path.read_text())
    model = BPTAutoregressiveTransformer(**checkpoint['config']).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    report = dict(source_sha256=args.source_sha256, training_source_sha256=TRAINING_SOURCE,
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cases={},
        scope='Same weights and conditions as multiasset v1; grammar masking only; no retraining')
    output = root.parent
    for asset in manifest['assets']:
        if asset['status'] != 'accepted':
            continue
        row = asset['samples'][0]
        candidates = [p for p in inputs.rglob(row['file'])
                      if hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']]
        if not candidates:
            raise RuntimeError('Missing hash-matched input fixture')
        fixture = np.load(candidates[0])
        points = torch.tensor(fixture['points'], device='cuda')[None]
        result = model.generate(points, grammar_constrained=True)
        tokens = np.asarray(result.pop('tokens'), dtype=np.uint16)
        np.save(output / f"{asset['asset']}-constrained-tokens.npy", tokens)
        result['geometry_tokens'] = len(tokens)
        try:
            v, f = decode(tokens)
            if not len(f):
                raise ValueError('Empty generated mesh')
            np.savez_compressed(output / f"{asset['asset']}-constrained-mesh.npz", vertices=v, faces=f)
            result.update(decodable=True, faces=len(f), vertices=len(v))
        except ValueError as exc:
            result.update(decodable=False, reason=str(exc))
        report['cases'][asset['asset']] = dict(split=asset['split'],
            unconstrained=baseline['evaluation'][asset['asset']]['generation'], constrained=result)
        (output / 'bpt-constraints-report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(asset['asset'], result, flush=True)


if __name__ == '__main__':
    main()
