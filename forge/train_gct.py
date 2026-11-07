"""Train the FORGE-Net geometry completion transformer (GCT).

Environment-overridable; designed for Kaggle dual T4 (DataParallel) and CPU
smoke tests. Writes resumable checkpoints plus JSONL metrics.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .constants import FINE, GRID
from .data import ForgeDataset, collate
from .model_gct import GeometryCompletionTransformer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=os.environ.get('FORGE_MANIFEST',
                                                             'artifacts/forge3d/corpus/manifest.json'))
    parser.add_argument('--out', default=os.environ.get('FORGE_OUT', 'artifacts/forge3d/gct'))
    parser.add_argument('--resume', default=os.environ.get('FORGE_RESUME', ''))
    parser.add_argument('--steps', type=int, default=int(os.environ.get('FORGE_STEPS', '20000')))
    parser.add_argument('--batch', type=int, default=int(os.environ.get('FORGE_BATCH', '4')))
    parser.add_argument('--lr', type=float, default=float(os.environ.get('FORGE_LR', '2e-4')))
    parser.add_argument('--workers', type=int, default=int(os.environ.get('FORGE_WORKERS', '4')))
    parser.add_argument('--width', type=int, default=int(os.environ.get('FORGE_WIDTH', '384')))
    parser.add_argument('--heads', type=int, default=int(os.environ.get('FORGE_HEADS', '8')))
    parser.add_argument('--depth', type=int, default=int(os.environ.get('FORGE_DEPTH', '8')))
    parser.add_argument('--val-every', type=int, default=int(os.environ.get('FORGE_VAL_EVERY', '1000')))
    parser.add_argument('--ckpt-every', type=int, default=int(os.environ.get('FORGE_CKPT_EVERY', '2000')))
    parser.add_argument('--val-assets', type=int, default=int(os.environ.get('FORGE_VAL_ASSETS', '48')))
    parser.add_argument('--seed', type=int, default=int(os.environ.get('FORGE_SEED', '42')))
    parser.add_argument('--limit', type=int, default=int(os.environ.get('FORGE_LIMIT', '0')))
    return parser.parse_args()


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def dice_loss(logits, target, eps=1.0):
    probability = torch.sigmoid(logits)
    numerator = 2.0 * (probability * target).sum() + eps
    denominator = probability.sum() + target.sum() + eps
    return 1.0 - numerator / denominator


def save_checkpoint(path, model, optimizer, scheduler, scaler, step, config, metrics,
                    *, half=False):
    state = unwrap(model).state_dict()
    if half:
        state = {key: value.half() if value.is_floating_point() else value
                 for key, value in state.items()}
    payload = {
        'model': state,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'step': step,
        'config': config,
        'metrics': metrics,
        'format': 'forge3d-gct-v1',
    }
    torch.save(payload, path)
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest(), path.stat().st_size


@torch.no_grad()
def evaluate(model, loader, device, max_batches=2):
    model.eval()
    sdf_errors, ious = [], []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        occ = batch['occ'].to(device, non_blocking=True)
        target = batch['target_occ'].to(device, non_blocking=True)
        pts = batch['pts'].to(device, non_blocking=True)
        query = batch['sdf_coords'].to(device, non_blocking=True)
        sdf_target = batch['sdf'].to(device, non_blocking=True)
        occ_logits, sdf = model(occ, pts, query)
        sdf_errors.append(float((sdf * FINE - sdf_target * FINE).abs().mean()))
        predicted = torch.sigmoid(occ_logits) > 0.5
        truth = target > 0.5
        union = (predicted | truth).sum().clamp_min(1)
        ious.append(float((predicted & truth).sum() / union))
    model.train()
    return float(np.mean(sdf_errors)), float(np.mean(ious))


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch {torch.__version__} cuda {torch.version.cuda} '
          f'available {torch.cuda.is_available()} count {torch.cuda.device_count()}',
          flush=True)
    if device.type == 'cuda':
        print(f'device: {torch.cuda.get_device_name(0)}', flush=True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    else:
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    train_ds = ForgeDataset(args.manifest, 'train', task='gct', limit=args.limit,
                            augment=True)
    val_ds = ForgeDataset(args.manifest, 'validation', task='gct', limit=args.val_assets)
    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate,
                              pin_memory=pin, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=min(args.batch, len(val_ds)),
                            shuffle=False, num_workers=0, collate_fn=collate)
    print(f'data: {len(train_ds)} train / {len(val_ds)} validation assets', flush=True)

    config = dict(width=args.width, heads=args.heads, depth=args.depth,
                  batch=args.batch, lr=args.lr, steps=args.steps)
    model = GeometryCompletionTransformer(width=args.width, heads=args.heads, depth=args.depth)
    parameters = sum(p.numel() for p in model.parameters())
    print(f'model: {parameters / 1e6:.2f}M parameters on {device}', flush=True)

    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        start_step = int(checkpoint.get('step', 0))
        print(f'resumed from {args.resume} at step {start_step}', flush=True)

    model.to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f'using DataParallel across {torch.cuda.device_count()} GPUs', flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = 500
    horizon = max(1, args.steps - start_step)

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, horizon - warmup))
        return 0.5 * (1.0 + np.cos(np.pi * progress)) * 0.9 + 0.1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    if args.resume:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        except (KeyError, ValueError):
            pass
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / 'train-log.jsonl'
    best = float('inf')
    step = start_step
    running = []
    started = time.time()
    data_time = time.time()

    def next_batch(iterator):
        nonlocal data_time
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        data_time = time.time()
        return batch, iterator

    iterator = iter(train_loader)
    model.train()
    while step < args.steps:
        batch, iterator = next_batch(iterator)
        occ = batch['occ'].to(device, non_blocking=True)
        target_occ = batch['target_occ'].to(device, non_blocking=True)
        pts = batch['pts'].to(device, non_blocking=True)
        query = batch['sdf_coords'].to(device, non_blocking=True)
        sdf_target = batch['sdf'].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == 'cuda'):
            occ_logits, sdf = model(occ, pts, query)
            loss_occ = F.binary_cross_entropy_with_logits(occ_logits, target_occ) \
                + dice_loss(occ_logits, target_occ)
            loss_sdf = F.huber_loss(sdf * FINE, sdf_target * FINE, delta=0.5)
            loss = loss_sdf + 0.5 * loss_occ
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        running.append((float(loss_sdf.detach()), float(loss_occ.detach()),
                            float(loss.detach())))
        if step % 20 == 0:
            window = running[-20:]
            mean_sdf = float(np.mean([row[0] for row in window]))
            mean_occ = float(np.mean([row[1] for row in window]))
            rate = (time.time() - started) / max(1, step - start_step)
            print(f'step {step}/{args.steps} sdf {mean_sdf:.3f} occ {mean_occ:.3f} '
                  f'lr {scheduler.get_last_lr()[0]:.2e} {rate:.2f}s/step', flush=True)
            with open(log_path, 'a') as handle:
                handle.write(json.dumps({'step': step, 'sdf': mean_sdf, 'occ': mean_occ,
                                         'lr': scheduler.get_last_lr()[0],
                                         'seconds_per_step': rate}) + '\n')
        if step % args.val_every == 0 or step == args.steps:
            sdf_mae, iou = evaluate(model, val_loader, device)
            print(f'validation step {step}: sdf_mae_voxels {sdf_mae:.4f} coarse_iou {iou:.4f}',
                  flush=True)
            with open(log_path, 'a') as handle:
                handle.write(json.dumps({'step': step, 'val_sdf_mae': sdf_mae,
                                         'val_coarse_iou': iou}) + '\n')
            metrics = {'val_sdf_mae': sdf_mae, 'val_coarse_iou': iou}
            if sdf_mae < best:
                best = sdf_mae
                digest, size = save_checkpoint(out / 'forge-gct-best.pt', model, optimizer,
                                               scheduler, scaler, step, config, metrics)
                (out / 'forge-gct-best.json').write_text(json.dumps(
                    {'step': step, 'metrics': metrics, 'sha256': digest,
                     'bytes': size}, indent=2) + '\n')
        if step % args.ckpt_every == 0 or step == args.steps:
            digest, size = save_checkpoint(out / f'forge-gct-{step}.pt', model, optimizer,
                                           scheduler, scaler, step, config,
                                           {'best_val_sdf_mae': best})
            print(f'checkpoint {step}: {size / 1e6:.1f} MB sha256 {digest[:16]}', flush=True)
            (out / 'latest.json').write_text(json.dumps(
                {'step': step, 'checkpoint': f'forge-gct-{step}.pt',
                 'sha256': digest, 'bytes': size}, indent=2) + '\n')
    final = out / f'forge-gct-{step}.pt'
    if final.exists():
        checkpoint = torch.load(final, map_location='cpu', weights_only=False)
        half_digest, half_size = save_checkpoint(out / f'forge-gct-{step}-fp16.pt',
                                                 model, optimizer, scheduler, scaler,
                                                 step, config, checkpoint.get('metrics', {}),
                                                 half=True)
        print(f'fp16 export: {half_size / 1e6:.1f} MB sha256 {half_digest[:16]}', flush=True)
    (out / 'complete.json').write_text(json.dumps(
        {'step': step, 'best_val_sdf_mae': best, 'params': parameters}, indent=2) + '\n')


if __name__ == '__main__':
    main()
