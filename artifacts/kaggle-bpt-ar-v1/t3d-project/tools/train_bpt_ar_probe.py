"""Small CUDA optimization probe on one cleaned source; not corpus training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from t3d.bpt import decode
from t3d.bpt_ar import BPTAutoregressiveTransformer, teacher_forcing_batch


def sample_points(vertices, faces, seed=42):
    rng = np.random.default_rng(seed)
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1)
    selected = triangles[rng.choice(len(faces), 4096, p=areas / areas.sum())]
    uv = rng.random((4096, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    return selected[:, 0] + uv[:, :1] * (selected[:, 1] - selected[:, 0]) + uv[:, 1:] * (selected[:, 2] - selected[:, 0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError('Positive training steps required')
    if not torch.cuda.is_available():
        raise RuntimeError('Run this probe on a Kaggle GPU notebook')
    torch.manual_seed(42)
    root = Path(__file__).resolve().parents[1]
    fixture = root / 'artifacts/bpt/cleaned_Avocado_all_24_rotations.npz'
    tokens = np.load(fixture)['tokens'].astype(np.int64)
    vertices, faces = decode(tokens)
    points = torch.tensor(sample_points(vertices, faces), dtype=torch.float32, device='cuda')[None]
    inputs, targets = teacher_forcing_batch([tokens], 'cuda')
    config = dict(width=96, heads=4, depth=2, condition_tokens=16,
                  context_length=len(tokens) + 1)
    model = BPTAutoregressiveTransformer(**config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.1)
    losses = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(inputs, targets, points)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite training gradient')
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 10 == 0:
            print(step, losses[-1], flush=True)
    model.eval()
    with torch.no_grad():
        final_loss = float(model.loss(inputs, targets, points))
        logits = model(inputs, points)
        accuracy = float((logits.argmax(-1) == targets).float().mean())
    if final_loss >= losses[0]:
        raise RuntimeError('Optimization did not reduce next-token loss')
    output = root.parent
    torch.save(dict(config=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    source_sha256=args.source_sha256, steps=args.steps), output / 'bpt-ar-probe.pt')
    report = dict(scope='Single cleaned Avocado optimization probe, not held-out quality or corpus training',
        source_sha256=args.source_sha256, fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        config=config, conditioning_points=4096, geometry_tokens=len(tokens),
        steps=args.steps, initial_loss=losses[0], final_loss=final_loss,
        training_next_token_accuracy=accuracy, losses=losses,
        limitations=['No generated mesh evaluated', 'One training asset; no generalization claim',
                     'No Hourglass attention, KV cache or DPO training'])
    (output / 'bpt-ar-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'losses'}, indent=2))


if __name__ == '__main__':
    main()
