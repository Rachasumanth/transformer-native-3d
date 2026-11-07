"""Train Stage 2 shape-latent DiT on a full 4x4x4 latent grid.

Instead of feeding only Stage 1 support coordinates, this variant always feeds
all 64 latent cells and supervises empty cells with zero targets. The SC-VAE
zero latent is known to decode to empty space, so the model learns occupancy
and geometry jointly and removes the Stage 1 recall bottleneck.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from t3d.conditioning import PromptConditioner
from t3d.container import load
from t3d.dataset import AssetDataset, batch_arrays
from t3d.extraction import voxels_to_mesh
from t3d.flow import SparseFlowDiT, flow_pair, sample_flow
from t3d.metrics import surface_comparison
from t3d.sc_vae import SparseCompressionVAE
from tools.generate_corpus_prompts import id_to_prompt

STEPS = int(os.environ.get('T3D_STEPS', '12000'))
GRID = 4
CHECKPOINT_STEPS = sorted({min(STEPS, x) for x in (2000, 4000, 6000, 8000, 10000, 12000)})
if STEPS < 2000:
    CHECKPOINT_STEPS = [STEPS]
LIMIT = int(os.environ.get('T3D_LIMIT', '0'))
BENCH = () if os.environ.get('T3D_SMOKE') else (
    'kenney-pirate-kit-boat-row-large',
    'kenney-pirate-kit-bottle',
    'kenney-pirate-kit-ship-medium',
    'kenney-pirate-kit-rocks-a',
    'kenney-space-kit-craft-speedera',
    'kenney-space-kit-rocket-basea',
    'kenney-space-kit-monorail-trainfront',
    'kenney-space-kit-barrel',
)


def find_corpus_manifest():
    matches = list(Path('/kaggle/input').rglob('manifest.json'))
    preferred = [p for p in matches if 'kenney-corpus-64' in str(p)]
    if preferred:
        return preferred[0]
    for candidate in (Path('/content/data/kenney-corpus-64/manifest.json'), Path('artifacts/kenney-corpus-64/manifest.json'), *matches):
        if candidate.exists() and '64' in str(candidate):
            return candidate
    raise FileNotFoundError('Mounted 64³ Kenney corpus manifest not found')


def find_vae_checkpoint():
    candidates = list(Path('/kaggle/input').rglob('triton-kenney-corpus-64-*.pt'))
    candidates += list(Path('/content/data/checkpoints').glob('*.pt'))
    candidates += list(Path('artifacts/kaggle-kenney-corpus-64-training').glob('*.pt'))
    if not candidates:
        raise FileNotFoundError('64³ SC-VAE checkpoint not found')
    def step(path):
        return int(path.stem.rsplit('-', 1)[-1]) if path.stem.rsplit('-', 1)[-1].isdigit() else 0
    return sorted(candidates, key=step)[-1]


def encode_asset(vae, ds, index, device):
    asset = ds[index]
    coords, feats = batch_arrays([asset], 'shape')
    posterior = vae.encode(torch.from_numpy(coords).to(device), torch.from_numpy(feats).to(device))
    return posterior


def full_grid_target(posterior, grid=GRID):
    cells = torch.cartesian_prod(*(torch.arange(grid, device=posterior.mean.device) for _ in range(3)))
    target = torch.zeros((len(cells), posterior.mean.shape[1]), device=posterior.mean.device, dtype=posterior.mean.dtype)
    occ = torch.zeros(len(cells), device=posterior.mean.device, dtype=torch.bool)
    src = posterior.coords[:, 1:]
    flat = (src[:, 0] * grid + src[:, 1]) * grid + src[:, 2]
    assert torch.unique(flat).numel() == len(flat), 'Duplicate latent coordinates'
    target[flat] = posterior.mean
    occ[flat] = True
    return cells, target, occ


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    if device == 'cuda':
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f'device={device}', flush=True)

    manifest = find_corpus_manifest()
    train_ds = AssetDataset(manifest, split='train')
    val_ds = AssetDataset(manifest, split='validation')
    if LIMIT:
        train_ds.records = train_ds.records[:LIMIT]
        val_ds.records = val_ds.records[:LIMIT]
    print(f'corpus={manifest} train={len(train_ds)} val={len(val_ds)}', flush=True)

    vae_path = find_vae_checkpoint()
    vae_data = torch.load(vae_path, map_location=device, weights_only=False)
    cfg = {k: v for k, v in vae_data['config'].items() if k in (
        'mode', 'channels', 'latent_channels', 'levels', 'downsample_block', 'upsample_block', 'convolution_backend')}
    if 'input.weight' in vae_data['model']:
        n = max(int(k.split('.')[1]) for k in vae_data['model'] if k.startswith('encoder_blocks.'))
        cfg['channels'] = tuple(vae_data['model'][f'encoder_blocks.{i}.scale'].shape[0] for i in range(n + 1))
    if device == 'cpu':
        cfg['convolution_backend'] = 'reference'
    vae = SparseCompressionVAE(**cfg).to(device).eval()
    vae.load_state_dict(vae_data['model'])
    for p in vae.parameters():
        p.requires_grad = False
    latent_channels = int(vae_data['config']['latent_channels'])
    print(f'vae={vae_path} latent_channels={latent_channels}', flush=True)

    conditioner = PromptConditioner.from_pretrained(
        image_model='facebook/dinov2-small',
        image_revision='ed25f3a31f01632728cabb09d1542f84ab7b0056',
        text_model='openai/clip-vit-base-patch32',
        text_revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
        output_channels=32,
    ).to(device)

    width, heads, depth = 144, 6, 6
    model = SparseFlowDiT(
        channels=latent_channels, condition_channels=32,
        width=width, heads=heads, depth=depth, patch_size=64,
    ).to(device)
    params = list(model.parameters()) + list(conditioner.text_projection.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=1e-5)

    print('caching text and full-grid latent targets...', flush=True)
    raw_text = {}
    targets = {}
    with torch.no_grad():
        for rec in list(train_ds.records) + list(val_ds.records):
            inputs = conditioner.tokenizer(id_to_prompt(rec.asset_id), return_tensors='pt', padding=False, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
            raw_text[rec.asset_id] = conditioner.text_encoder(**inputs).last_hidden_state[0].detach()
        for i, rec in enumerate(train_ds.records):
            posterior = encode_asset(vae, train_ds, i, device)
            targets[rec.asset_id] = full_grid_target(posterior)
            if (i + 1) % 100 == 0:
                print(f'encoded {i + 1}/{len(train_ds)}', flush=True)

    cells = torch.cartesian_prod(*(torch.arange(GRID, device=device) for _ in range(3)))
    out_dir = Path(os.environ.get('T3D_OUT', '/kaggle/working'))
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        out_dir = Path('/content/output')
        out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    saved = {}
    start = time.perf_counter()
    rng = np.random.default_rng(42)
    model.train()
    conditioner.train()

    for step in range(1, STEPS + 1):
        rec = train_ds.records[int(rng.integers(len(train_ds)))]
        _, target, occ = targets[rec.asset_id]
        tokens = conditioner.text_projection(raw_text[rec.asset_id])

        time_t = torch.rand((), device=device)
        noise = torch.randn_like(target)
        noisy, v_target = flow_pair(target, noise, time_t)
        drop = torch.rand((), device=device) < 0.1
        pred_v = model(noisy, cells, time_t, None if drop else tokens)
        flow = F.mse_loss(pred_v, v_target)
        pred_clean = noisy + (1.0 - time_t) * pred_v
        tweedie = F.mse_loss(pred_clean, target)

        norm = pred_clean.abs().sum(-1)
        occ_loss = F.binary_cross_entropy_with_logits(norm - 5.0, occ.float())
        prob = torch.sigmoid((norm - 5.0) / 2.0)
        inter = (prob * occ.float()).sum()
        dice = 1.0 - (2.0 * inter + 1.0) / (prob.sum() + occ.float().sum() + 1.0)
        zero = F.mse_loss(pred_clean[~occ], target[~occ]) if (~occ).any() else torch.tensor(0.0, device=device)
        loss = flow + tweedie + 0.5 * dice + 1.0 * zero + 0.5 * occ_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            row = dict(step=step, asset=rec.asset_id, loss=round(float(loss), 5), flow=round(float(flow), 5),
                       tweedie=round(float(tweedie), 5), dice=round(float(dice), 5), occ=round(float(occ_loss), 5), zero=round(float(zero), 5),
                       lr=round(float(scheduler.get_last_lr()[0]), 6))
            history.append(row)
            print(json.dumps(row), flush=True)

        if step in CHECKPOINT_STEPS:
            path = out_dir / f'geometry-dit-64-fullgrid-{step}.pt'
            torch.save(dict(
                version=1, step=step, model=model.state_dict(),
                text_projection=conditioner.text_projection.state_dict(),
                manifest_sha256=train_ds.fingerprint, latent_channels=latent_channels,
                resolution=64, width=width, heads=heads, depth=depth,
                full_grid=True, grid_size=GRID, models=conditioner.model_provenance,
            ), path)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            saved[path.name] = sha
            print(f'saved {path.name} sha256={sha}', flush=True)

    model.eval()
    conditioner.eval()
    benchmark = []
    with torch.no_grad():
        for asset_id in BENCH:
            matches = [i for i, r in enumerate(val_ds.records) if r.asset_id == asset_id]
            if not matches:
                continue
            i = matches[0]
            asset = val_ds[i]
            gt_v, gt_f = voxels_to_mesh(asset)
            posterior = encode_asset(vae, val_ds, i, device)
            _, gt_target, gt_occ = full_grid_target(posterior)
            tokens = conditioner.text_projection(raw_text[asset_id])
            noise = torch.randn((len(cells), latent_channels), device=device, generator=torch.Generator(device=device).manual_seed(42))
            pred = sample_flow(model, noise, cells, tokens, steps=20, guidance=2.5, method='midpoint')
            pred_occ = pred.abs().sum(-1) > 5.0
            inter = int((pred_occ & gt_occ).sum())
            iou = inter / max(1, int((pred_occ | gt_occ).sum()))
            bcoords = torch.cat([torch.zeros((len(cells), 1), dtype=torch.long, device=device), cells], dim=1)
            out, raw, _ = vae.decode(bcoords, pred, threshold=0.46)
            if len(out) == 0:
                benchmark.append(dict(asset_id=asset_id, error='empty decode'))
                continue
            valid = (out[:, 1:] < 64).all(1)
            out, raw = out[valid], raw[valid]
            from t3d.schema import VoxelAsset
            pa = VoxelAsset(
                coords=out[:, 1:].cpu().numpy().astype(np.int32),
                dual_vertices=raw[:, :3].sigmoid().cpu().numpy(),
                intersected_flag=(raw[:, 3:6].sigmoid() > 0.46).cpu().numpy(),
                split_weight=raw[:, 6:].cpu().numpy(),
                aabb=np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=np.float32),
                grid_size=(64, 64, 64),
            )
            pv, pf = voxels_to_mesh(pa, resolve_nonmanifold=True)
            comp = surface_comparison(gt_v, gt_f, pv, pf, samples=512, tolerance=0.01)
            row = dict(asset_id=asset_id, occupancy_iou=round(iou, 4),
                       pred_cells=int(pred_occ.sum()), gt_cells=int(gt_occ.sum()),
                       mean_distance=round(float(comp['mean_distance']), 5),
                       fscore=round(float(comp['fscore']), 4), triangles=len(pf),
                       nonmanifold=int(comp.get('nonmanifold_edges', 0)))
            benchmark.append(row)
            print(json.dumps(row), flush=True)

    report = dict(
        scope='Full 4³ latent grid Stage 2 DiT (all 64 cells, zero-target empty cells) at 64³',
        device=torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
        elapsed_seconds=round(time.perf_counter() - start, 1),
        steps=STEPS, train_count=len(train_ds), validation_count=len(val_ds),
        latent_channels=latent_channels, width=width, heads=heads, depth=depth,
        checkpoints=saved, benchmark=benchmark,
        mean_surface_distance=round(float(np.mean([b['mean_distance'] for b in benchmark if 'mean_distance' in b])), 5) if benchmark else None,
        mean_fscore=round(float(np.mean([b['fscore'] for b in benchmark if 'fscore' in b])), 4) if benchmark else None,
        history=history,
    )
    report_path = out_dir / 'geometry-dit-64-fullgrid-report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'report={report_path}', flush=True)


if __name__ == '__main__':
    main()
