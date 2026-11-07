"""Train Unified Single-Stage Flow DiT for joint occupancy and geometry generation on full 4³ grid.

Eliminates the Stage 1 -> Stage 2 cascading bottleneck by predicting both coarse cell occupancy
and continuous geometry latents simultaneously on all 64 grid cells using Rectified Flow Matching.
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
from t3d.dataset import AssetDataset, batch_arrays
from t3d.extraction import voxels_to_mesh
from t3d.metrics import surface_comparison
from t3d.sc_vae import SparseCompressionVAE
from t3d.unified_dit import (
    UnifiedFlowDiT,
    get_grid_cells,
    pack_unified_token,
    unpack_unified_token,
    sample_logit_normal_time,
    sample_unified_flow,
    GRID_SIZE,
    TOTAL_CELLS,
)
from tools.generate_corpus_prompts import id_to_prompt

STEPS = int(os.environ.get('T3D_STEPS', '20000'))
CHECKPOINT_STEPS = sorted({min(STEPS, x) for x in (2500, 5000, 10000, 15000, 20000)})
if STEPS < 2500:
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
    env = os.environ.get('T3D_MANIFEST')
    if env and Path(env).exists():
        return Path(env)
    matches = list(Path('/kaggle/input').rglob('manifest.json'))
    preferred = [p for p in matches if 'kenney-corpus-64' in str(p)]
    if preferred:
        return preferred[0]
    for candidate in (
        Path('artifacts/kenney-corpus-64-id-expanded/manifest.json'),
        Path('/content/data/kenney-corpus-64/manifest.json'),
        Path('artifacts/kenney-corpus-64/manifest.json'),
        *matches
    ):
        if candidate.exists() and '64' in str(candidate):
            return candidate
    raise FileNotFoundError('Mounted 64³ Kenney corpus manifest not found')


def find_vae_checkpoint():
    env = os.environ.get('T3D_VAE')
    if env and Path(env).exists():
        return Path(env)
    candidates = list(Path('/kaggle/input').rglob('triton-kenney-corpus-64-*.pt'))
    candidates += list(Path('/content/data/checkpoints').glob('*.pt'))
    candidates += list(Path('artifacts/kaggle-kenney-corpus-64-training').glob('*.pt'))
    if not candidates:
        raise FileNotFoundError('64³ SC-VAE checkpoint not found')
    def step(path):
        return int(path.stem.rsplit('-', 1)[-1]) if path.stem.rsplit('-', 1)[-1].isdigit() else 0
    return sorted(candidates, key=step)[-1]


def load_or_compute_latent_stats(vae, ds, device):
    for stats_file in (Path('artifacts/latent_stats_64.json'), Path('/kaggle/working/latent_stats_64.json'), Path('latent_stats_64.json')):
        if stats_file.exists():
            d = json.loads(stats_file.read_text())
            mean = torch.tensor(d['mean'], device=device, dtype=torch.float32)
            std = torch.tensor(d['std'], device=device, dtype=torch.float32)
            print(f'Loaded precomputed latent stats from {stats_file}')
            return mean, std
    if False:
        d = json.loads(stats_file.read_text())
        mean = torch.tensor(d['mean'], device=device, dtype=torch.float32)
        std = torch.tensor(d['std'], device=device, dtype=torch.float32)
        print(f"Loaded precomputed latent stats from {stats_file}")
        return mean, std

    print("Computing latent stats across dataset...")
    all_means = []
    with torch.no_grad():
        for i in range(min(500, len(ds))):
            asset = ds[i]
            coords, feats = batch_arrays([asset], 'shape')
            post = vae.encode(torch.from_numpy(coords).to(device), torch.from_numpy(feats).to(device))
            all_means.append(post.mean)
    cat = torch.cat(all_means, dim=0)
    mean = cat.mean(dim=0)
    std = cat.std(dim=0).clamp_min(1e-4)
    return mean, std


def compute_surface_weights(asset, grid_size=GRID_SIZE, device=torch.device('cpu')):
    """Compute per-octant surface ratio weighting."""
    coords = asset.coords  # (N_voxels, 3) in [0, 63]
    octants = coords // 16
    flat = (octants[:, 0] * grid_size + octants[:, 1]) * grid_size + octants[:, 2]
    counts = np.bincount(flat, minlength=grid_size**3)
    max_voxels_per_octant = 16 * 16 * 16  # 4096
    surface_ratios = counts / max_voxels_per_octant

    # Active weight: 1.0 + 2.0 * surface_ratio; Void weight: 0.5
    weights = np.where(counts > 0, 1.0 + 2.0 * surface_ratios, 0.5).astype(np.float32)
    return torch.from_numpy(weights).to(device)


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

    mean_stats, std_stats = load_or_compute_latent_stats(vae, train_ds, device)

    conditioner = PromptConditioner.from_env(output_channels=32).to(device)

    width, heads, depth = 144, 6, 6  # head_dim=24, RoPE compliant
    model = UnifiedFlowDiT(
        latent_channels=latent_channels,
        condition_channels=32,
        width=width,
        heads=heads,
        depth=depth,
        patch_size=64,
    ).to(device)
    params = list(model.parameters()) + list(conditioner.text_projection.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=1e-5)

    print('Caching text tokens, targets, and cell weights...', flush=True)
    raw_text = {}
    cached_targets = {}
    cached_weights = {}

    with torch.no_grad():
        for rec in list(train_ds.records) + list(val_ds.records):
            inputs = conditioner.tokenizer(id_to_prompt(rec.asset_id), return_tensors='pt', padding=False, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
            raw_text[rec.asset_id] = conditioner.text_encoder(**inputs).last_hidden_state[0].detach()

        for i, rec in enumerate(train_ds.records):
            asset = train_ds[i]
            coords, feats = batch_arrays([asset], 'shape')
            post = vae.encode(torch.from_numpy(coords).to(device), torch.from_numpy(feats).to(device))
            target, occ_mask, cells = pack_unified_token(
                post.coords, post.mean, mean_stats, std_stats, device=device
            )
            weights = compute_surface_weights(asset, device=device)
            cached_targets[rec.asset_id] = (target, occ_mask)
            cached_weights[rec.asset_id] = weights
            if (i + 1) % 200 == 0 or i == len(train_ds) - 1:
                print(f'Encoded {i + 1}/{len(train_ds)}', flush=True)

    grid_cells = get_grid_cells(device)
    out_env = os.environ.get('T3D_OUT')
    if out_env:
        out_dir = Path(out_env)
    elif Path('/kaggle/working').exists():
        out_dir = Path('/kaggle/working')
    elif Path('/content').exists():
        out_dir = Path('/content/output')
    else:
        out_dir = Path('artifacts/kaggle-unified-dit-v1')
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    saved = {}
    rng = np.random.default_rng(42)
    model.train()
    conditioner.train()

    print(f"Starting training for {STEPS} steps...", flush=True)
    t0 = time.time()

    for step in range(1, STEPS + 1):
        rec = train_ds.records[int(rng.integers(len(train_ds)))]
        target, occ = cached_targets[rec.asset_id]  # target: (64, 17)
        weights = cached_weights[rec.asset_id]      # weights: (64,)
        tokens = conditioner.text_projection(raw_text[rec.asset_id])

        # Logit-Normal time sampling
        time_t = sample_logit_normal_time((1,), device=device, sigma=1.0)
        noise = torch.randn_like(target)

        # Straight-line Rectified Flow: x_t = (1-t)*x_0 + t*epsilon, v_target = epsilon - x_0
        noisy = (1.0 - time_t) * target + time_t * noise
        v_target = noise - target

        # CFG drop (15% drop rate)
        drop = torch.rand((), device=device) < 0.15
        pred_v = model(noisy, grid_cells, time_t, None if drop else tokens)

        # Weighted Flow matching loss
        cell_weights = weights.unsqueeze(-1)  # (64, 1)
        flow_loss = (cell_weights * (pred_v - v_target) ** 2).mean()

        # Tweedie clean prediction: x_0 = x_t - t * v
        pred_clean = noisy - time_t * pred_v
        tweedie_loss = (cell_weights * (pred_clean - target) ** 2).mean()

        # Occupancy consistency on channel 0
        pred_occ_logits = pred_clean[:, 0]
        occ_target = occ.float()
        bce_loss = F.binary_cross_entropy_with_logits(pred_occ_logits, occ_target)
        prob = torch.sigmoid(pred_occ_logits)
        inter = (prob * occ_target).sum()
        dice_loss = 1.0 - (2.0 * inter + 1.0) / (prob.sum() + occ_target.sum() + 1.0)

        # Void geometry suppression: empty cells should have zero continuous latents
        void_mask = ~occ
        void_loss = (pred_clean[void_mask, 1:] ** 2).mean() if void_mask.any() else torch.tensor(0.0, device=device)

        total_loss = flow_loss + 0.5 * tweedie_loss + 0.5 * bce_loss + 0.5 * dice_loss + 1.0 * void_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            row = dict(
                step=step, asset=rec.asset_id,
                loss=round(float(total_loss), 5),
                flow=round(float(flow_loss), 5),
                tweedie=round(float(tweedie_loss), 5),
                bce=round(float(bce_loss), 5),
                dice=round(float(dice_loss), 5),
                void=round(float(void_loss), 5),
                lr=round(float(scheduler.get_last_lr()[0]), 6),
            )
            history.append(row)
            print(json.dumps(row), flush=True)

        if step in CHECKPOINT_STEPS:
            ckpt_path = out_dir / f'unified-dit-64-{step}.pt'
            torch.save(dict(
                version=1,
                step=step,
                model=model.state_dict(),
                text_projection=conditioner.text_projection.state_dict(),
                latent_mean=mean_stats.cpu().tolist(),
                latent_std=std_stats.cpu().tolist(),
                manifest_sha256=train_ds.fingerprint,
                latent_channels=latent_channels,
                total_channels=model.total_channels,
                resolution=64,
                width=width,
                heads=heads,
                depth=depth,
                unified=True,
                full_grid=True,
                grid_size=GRID_SIZE,
                models=conditioner.model_provenance,
            ), ckpt_path)
            sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
            saved[ckpt_path.name] = sha
            print(f'Saved checkpoint {ckpt_path.name} sha256={sha}', flush=True)

    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s", flush=True)

    # Benchmark evaluation on the 8 benchmark assets
    model.eval()
    conditioner.eval()
    benchmark_results = []
    print("Running benchmark evaluation on hold-out assets...", flush=True)

    with torch.no_grad():
        for asset_id in BENCH:
            matches = [i for i, r in enumerate(val_ds.records) if r.asset_id == asset_id]
            if not matches:
                continue
            i = matches[0]
            asset = val_ds[i]
            gt_v, gt_f = voxels_to_mesh(asset)

            tokens = conditioner.text_projection(raw_text[asset_id])
            pred_tokens = sample_unified_flow(
                model, tokens, coords=grid_cells, steps=30, guidance=3.5, method='midpoint',
                cfg_min=0.25, cfg_max=0.85, seed=42, device=device
            )

            bcoords, active_latents, active_mask = unpack_unified_token(
                pred_tokens, mean_stats, std_stats, occ_threshold=0.0
            )

            if len(bcoords) == 0:
                benchmark_results.append(dict(asset_id=asset_id, error='empty decode'))
                continue

            out_vox, raw_feats, _ = vae.decode(bcoords, active_latents, threshold=0.46)
            if len(out_vox) == 0:
                benchmark_results.append(dict(asset_id=asset_id, error='empty voxel decode'))
                continue

            valid = (out_vox[:, 1:] < 64).all(dim=1)
            out_vox, raw_feats = out_vox[valid], raw_feats[valid]

            from t3d.schema import VoxelAsset
            from t3d.reconstruction import dual_contouring
            pa = VoxelAsset(
                coords=out_vox[:, 1:].cpu().numpy().astype(np.int32),
                dual_vertices=raw_feats[:, :3].sigmoid().cpu().numpy(),
            )
            pred_v, pred_f = dual_contouring(pa)

            if len(pred_v) == 0 or len(pred_f) == 0:
                benchmark_results.append(dict(asset_id=asset_id, error='empty mesh'))
                continue

            metrics = surface_comparison(gt_v, gt_f, pred_v, pred_f)
            rec = {
                'asset_id': asset_id,
                'active_cells': int(active_mask.sum()),
                'decoded_voxels': len(out_vox),
                'triangles': len(pred_f),
                'surface_distance': round(float(metrics.mean_distance), 5),
                'fscore_001': round(float(metrics.fscore), 5),
            }
            benchmark_results.append(rec)
            print(f"  {asset_id}: dist={rec['surface_distance']}, F@0.01={rec['fscore_001']}, active={rec['active_cells']}", flush=True)

    report_path = out_dir / 'unified-dit-64-report.json'
    summary = {
        'steps': STEPS,
        'elapsed_seconds': round(elapsed, 1),
        'saved_checkpoints': saved,
        'benchmark': benchmark_results,
        'history': history,
    }
    report_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(f"Saved report to {report_path}", flush=True)


if __name__ == '__main__':
    main()
