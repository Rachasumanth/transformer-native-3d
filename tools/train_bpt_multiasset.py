"""Asset-disjoint, augmented Track B learning experiment on Kaggle CUDA."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from t3d.bpt import decode
from t3d.bpt_ar import BPTAutoregressiveTransformer, teacher_forcing_batch


@torch.no_grad()
def evaluate(model, sample, window=512):
    x, y, points = sample
    total, correct = 0., 0
    for start in range(0, x.shape[1], window):
        target = y[:, start:start + window]
        logits = model(x[:, start:start + window], points, position_offset=start)
        total += float(torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction='sum'))
        correct += int((logits.argmax(-1) == target).sum())
    return dict(loss=total / y.numel(), accuracy=correct / y.numel(), targets=y.numel())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--steps', type=int, default=600)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError('Positive optimization steps required')
    if not torch.cuda.is_available():
        raise RuntimeError('Kaggle GPU required')
    torch.manual_seed(43)
    rng = np.random.default_rng(43)
    root = Path(__file__).resolve().parents[1]
    data = root / 'artifacts/bpt-training-pilot'
    manifest_path = data / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    ownership = {}
    for asset in manifest['assets']:
        digest = asset['source_sha256']
        if digest in ownership and ownership[digest] != asset['split']:
            raise RuntimeError('Source asset leaks between training and evaluation splits')
        ownership[digest] = asset['split']
    training, cases, longest = [], [], 0
    for asset in manifest['assets']:
        if asset['status'] != 'accepted':
            continue
        for row in asset['samples']:
            path = data / row['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise RuntimeError('Fixture hash mismatch')
            bundle = np.load(path)
            x, y = teacher_forcing_batch([bundle['tokens'].astype(np.int64)], 'cuda')
            points = torch.tensor(bundle['points'], device='cuda')[None]
            sample = (x, y, points)
            longest = max(longest, x.shape[1])
            if asset['split'] == 'train':
                training.append(sample)
            if row['quarter_turns_y'] == 0:
                cases.append((asset, row, sample))
    train_assets = {a['asset'] for a in manifest['assets'] if a['status'] == 'accepted' and a['split'] == 'train'}
    if len(train_assets) < 2:
        raise RuntimeError('At least two distinct training assets required')
    config = dict(width=96, heads=4, depth=2, condition_tokens=16, context_length=longest)
    model = BPTAutoregressiveTransformer(**config).cuda()
    model.eval()
    initial = {asset['asset']: evaluate(model, sample) for asset, _, sample in cases}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.1)
    losses = []
    model.train()
    for step in range(args.steps):
        x, y, points = training[int(rng.integers(len(training)))]
        size = min(512, x.shape[1])
        choice = float(rng.random())
        # Explicitly expose both BOS and EOS during long-stream crop training.
        start = (0 if choice < .25 else x.shape[1] - size if choice < .5
                 else int(rng.integers(x.shape[1] - size + 1)))
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(x[:, start:start + size], y[:, start:start + size], points,
                          position_offset=start)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError('Nonfinite optimization state')
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 50 == 0:
            print(step, losses[-1], flush=True)
    model.eval()
    output = root.parent
    checkpoint = output / 'bpt-multiasset.pt'
    torch.save(dict(config=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
        rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(), steps=args.steps,
        source_sha256=args.source_sha256), checkpoint)
    restored = BPTAutoregressiveTransformer(**config).cuda().eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True)['model'])
    results = {}
    for asset, row, sample in cases:
        name = asset['asset']
        final = evaluate(restored, sample)
        with torch.no_grad():
            torch.testing.assert_close(model(sample[0][:, :64], sample[2]),
                                       restored(sample[0][:, :64], sample[2]))
        generated = restored.generate(sample[2])
        tokens = np.asarray(generated.pop('tokens'), dtype=np.uint16)
        np.save(output / f'{name}-generated-tokens.npy', tokens)
        generated['tokens'] = len(tokens)
        try:
            vertices, faces = decode(tokens)
            if not len(faces):
                raise ValueError('No generated faces')
            np.savez_compressed(output / f'{name}-generated-mesh.npz', vertices=vertices, faces=faces)
            generated.update(decodable=True, vertices=len(vertices), faces=len(faces))
        except ValueError as exc:
            generated.update(decodable=False, reason=str(exc))
        results[name] = dict(split=asset['split'], initial=initial[name], final=final,
                            generation=generated)
        print(name, results[name], flush=True)
    report = dict(source_sha256=args.source_sha256,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        scope='Small licensed multiasset integration experiment; representative corpus still required',
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        config=config, steps=args.steps, training_assets=sorted(train_assets),
        training_rotated_streams=len(training), losses=losses, evaluation=results,
        crop_sampling='25% first crop, 25% final crop, 50% uniformly random crop; uniform rotated training stream selection',
        window_policy='512-token causal crops retain absolute stream position; evaluation covers each target once in nonoverlapping windows',
        generation_policy='Unconstrained greedy sampling from BOS, full KV context, explicit EOS or context limit; training crops have shorter history than inference',
        split_policy=manifest['split_policy'])
    (output / 'bpt-multiasset-report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
