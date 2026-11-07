"""Train the FORGE-Net material transformer on the FORGE-3D corpus."""
import json
import math
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ForgeDataset, collate_mat
from .model_mat import MaterialTransformer


def _cfg(name, default, cast=str):
    value = os.environ.get(name)
    return cast(value) if value not in (None, '') else default


class MaterialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weights', torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 0.1]))

    def forward(self, predicted, target):
        difference = (predicted - target).abs()
        color = F.huber_loss(predicted[..., :3], target[..., :3], delta=0.1,
                             reduction='none').mean((1, 2))
        scalar = difference[..., 3:].mean((1, 2))
        return (color * self.weights[0] + scalar * self.weights[3:].mean()).mean()


def validate(model, loader, device, limit=None):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if limit is not None and step >= limit:
                break
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            predicted = model(batch['occ'], batch['coords'], batch['normals'],
                              batch['hint'], batch['has_tex'])
            target = batch['pbr']
            error = (predicted - target).abs()
            total += float(error[..., :3].mean() * target.shape[0])
            count += target.shape[0]
    model.train()
    return total / max(count, 1)


def main():
    manifest = Path(_cfg('FORGE_MANIFEST', '/home/sumanth/forge-data/corpus-v1/manifest.json'))
    out = Path(_cfg('FORGE_OUT', '/home/sumanth/forge-data/mat-run'))
    steps = _cfg('FORGE_STEPS', 8000, int)
    batch = _cfg('FORGE_BATCH', 8, int)
    lr = _cfg('FORGE_LR', 2e-4, float)
    width = _cfg('FORGE_WIDTH', 256, int)
    heads = _cfg('FORGE_HEADS', 8, int)
    depth = _cfg('FORGE_DEPTH', 6, int)
    window = _cfg('FORGE_WINDOW', 128, int)
    workers = _cfg('FORGE_WORKERS', 4, int)
    val_every = _cfg('FORGE_VAL_EVERY', 500, int)
    ckpt_every = _cfg('FORGE_CKPT_EVERY', 1000, int)
    val_assets = _cfg('FORGE_VAL_ASSETS', 64, int)
    seed = _cfg('FORGE_SEED', 42, int)
    resume = os.environ.get('FORGE_RESUME', '')

    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch {torch.__version__} cuda {torch.version.cuda} '
          f'available {torch.cuda.is_available()} count {torch.cuda.device_count()}',
          flush=True)
    if device.type == 'cuda':
        print(f'device: {torch.cuda.get_device_name(0)}', flush=True)
    train_set = ForgeDataset(manifest, split='train', task='mat')
    val_set = ForgeDataset(manifest, split='validation', task='mat', max_assets=val_assets)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,
                              num_workers=workers, collate_fn=collate_mat,
                              drop_last=True, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_set, batch_size=max(1, batch // 2), shuffle=False,
                            num_workers=max(1, workers // 2), collate_fn=collate_mat)
    print(f'data: {len(train_set)} train / {len(val_set)} validation assets', flush=True)

    model = MaterialTransformer(width=width, heads=heads, depth=depth, window=window).to(device)
    config = dict(width=width, heads=heads, depth=depth, window=window)
    print(f'model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters '
          f'on {device}', flush=True)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / 200) *
        (0.5 * (1 + math.cos(math.pi * step / max(steps, 1)))))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    loss_fn = MaterialLoss().to(device)

    start_step, best = 0, float('inf')
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=False)
        target = model.module if isinstance(model, nn.DataParallel) else model
        target.load_state_dict(state['model'])
        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        if 'scheduler' in state:
            scheduler.load_state_dict(state['scheduler'])
        start_step = int(state.get('step', 0))
        best = float(state.get('best', best))
        print(f'resumed from {resume} at step {start_step}', flush=True)

    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'train-log.jsonl', 'a')
    running = []
    step = start_step
    loader = iter(train_loader)
    started = time.time()
    model.train()
    while step < steps:
        try:
            batch_data = next(loader)
        except StopIteration:
            loader = iter(train_loader)
            batch_data = next(loader)
        batch_data = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in batch_data.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == 'cuda'):
            predicted = model(batch_data['occ'], batch_data['coords'],
                              batch_data['normals'], batch_data['hint'],
                              batch_data['has_tex'])
            loss = loss_fn(predicted, batch_data['pbr'])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        running.append(float(loss.detach()))
        if step % 50 == 0:
            record = {'step': step, 'loss': sum(running) / len(running),
                      'lr': scheduler.get_last_lr()[0],
                      'seconds': round(time.time() - started, 1),
                      'device': str(device), 'gpus': torch.cuda.device_count()}
            print('train step', step, json.dumps(record), flush=True)
            log.write(json.dumps(record) + '\n')
            log.flush()
            running = []
        if step % val_every == 0 or step == steps:
            score = validate(model, val_loader, device)
            print(f'validation step {step}: color_mae {score:.4f}', flush=True)
            if score < best:
                best = score
                torch.save({'format': 'forge3d-mat-v1', 'model': model.state_dict(),
                            'config': config, 'step': step, 'best': best},
                           out / 'forge-mat-best.pt')
        if step % ckpt_every == 0 or step == steps:
            state = {'format': 'forge3d-mat-v1', 'model': model.state_dict(),
                     'config': config, 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'step': step, 'best': best}
            path = out / f'forge-mat-{step}.pt'
            torch.save(state, path)
            print(f'checkpoint {step}: {path.stat().st_size / 1e6:.1f} MB', flush=True)
            latest = {'step': step, 'best': best, 'checkpoint': path.name}
            (out / 'latest.json').write_text(json.dumps(latest))
    log.close()
    target = model.module if isinstance(model, nn.DataParallel) else model
    export = {k: v.half() for k, v in target.state_dict().items()}
    torch.save({'format': 'forge3d-mat-v1', 'model': export, 'config': config,
                'step': step}, out / 'forge-mat-fp16.pt')
    (out / 'complete.json').write_text(json.dumps({'step': step, 'best': best,
                                                   'seconds': round(time.time() - started, 1)}))
    print('done', step, 'best', round(best, 4), flush=True)


if __name__ == '__main__':
    main()
