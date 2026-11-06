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


@torch.no_grad()
def evaluate_asset(model, fixture, output, name):
    tokens = np.load(fixture)['tokens'].astype(np.int64)
    vertices, faces = decode(tokens)
    points = torch.tensor(sample_points(vertices, faces), dtype=torch.float32, device='cuda')[None]
    inputs, targets = teacher_forcing_batch([tokens], 'cuda')
    total_loss, correct = 0., 0
    for start in range(0, inputs.shape[1], model.context_length):
        x = inputs[:, start:start + model.context_length]
        y = targets[:, start:start + model.context_length]
        logits = model(x, points)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='sum')
        total_loss += float(loss)
        correct += int((logits.argmax(-1) == y).sum())
    count = targets.numel()
    generated = model.generate(points)
    emitted = np.asarray(generated.pop('tokens'), dtype=np.uint16)
    np.save(output / f'{name}-generated-tokens.npy', emitted)
    generated.update(geometry_tokens=len(emitted),
                     exact_reference_tokens=bool(np.array_equal(emitted, tokens)))
    try:
        out_vertices, out_faces = decode(emitted)
        if not len(out_faces):
            raise ValueError('Stream contains no faces')
        np.savez(output / f'{name}-generated-mesh.npz', vertices=out_vertices, faces=out_faces)
        lines = ['v ' + ' '.join(map(str, v)) for v in out_vertices]
        lines += ['f ' + ' '.join(str(int(i) + 1) for i in f) for f in out_faces]
        (output / f'{name}-generated.obj').write_text('\n'.join(lines) + '\n')
        generated.update(decodable=True, vertices=len(out_vertices), faces=len(out_faces))
    except ValueError as exc:
        generated.update(decodable=False, decode_error=str(exc))
    return dict(fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        reference_faces=len(faces), reference_tokens=len(tokens),
        teacher_forced_loss=total_loss / count, teacher_forced_accuracy=correct / count,
        evaluated_targets=count,
        window_policy='Nonoverlapping context-sized windows; learned positions reset in each window; EOS only at actual stream end',
        generation=generated)


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
    restored = BPTAutoregressiveTransformer(**config).cuda().eval()
    restored.load_state_dict(torch.load(output / 'bpt-ar-probe.pt', weights_only=True)['model'])
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs, points), logits)
    evaluation = {}
    for name, filename in (
        ('train-avocado', 'cleaned_Avocado_all_24_rotations.npz'),
        ('heldout-box', 'cleaned_Box_all_24_rotations.npz'),
        ('heldout-lime', 'cleaned_PolyHaven_food_lime_01_all_24_rotations.npz')):
        evaluation[name] = evaluate_asset(restored, root / 'artifacts/bpt' / filename, output, name)
    report = dict(scope='Single cleaned Avocado optimization probe, not held-out quality or corpus training',
        source_sha256=args.source_sha256, fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        config=config, conditioning_points=4096, geometry_tokens=len(tokens),
        steps=args.steps, initial_loss=losses[0], final_loss=final_loss,
        training_next_token_accuracy=accuracy, losses=losses,
        checkpoint_reloaded=True, evaluation=evaluation,
        limitations=['One training asset; two held-out fixtures do not establish corpus quality',
                     'Unconstrained greedy generation; decode validity is distinct from complete EOS termination',
                     'No Hourglass attention, sliding context or DPO training'])
    (output / 'bpt-ar-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'losses'}, indent=2))


if __name__ == '__main__':
    main()
