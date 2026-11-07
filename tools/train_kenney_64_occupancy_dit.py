"""Train Stage 1 text-conditioned coarse occupancy DiT on the 64³ Kenney corpus."""
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from t3d.conditioning import PromptConditioner
from t3d.dataset import AssetDataset
from t3d.flow import SparseFlowDiT, flow_pair, sample_flow
from tools.generate_corpus_prompts import id_to_prompt


def build_latent_occupancy(asset, grid_dim=4):
    """Downsample active 64³ voxel coordinates into a 4³ binary latent occupancy grid."""
    lat_coords = asset.coords // 16
    flat = (lat_coords[:, 0] * grid_dim + lat_coords[:, 1]) * grid_dim + lat_coords[:, 2]
    unique_flat = np.unique(flat)
    target = np.zeros((grid_dim ** 3, 1), dtype=np.float32)
    target[unique_flat] = 1.0
    return torch.from_numpy(target)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    else:
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    # Discover dataset
    env_manifest = os.environ.get('T3D_MANIFEST')
    if env_manifest:
        manifest_path = Path(env_manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f'T3D_MANIFEST not found: {manifest_path}')
    else:
        matches = list(Path('/kaggle/input').rglob('manifest.json'))
        corpus_matches = [
            p for p in matches
            if 't3d-kenney-corpus-64' in str(p) or (p.parent / 'manifest.json').exists() and '64' in str(p)
        ]
        if not corpus_matches:
            candidates = [
                Path('artifacts/kenney-corpus-64/manifest.json'),
                *list(Path('/kaggle/input').rglob('manifest.json'))
            ]
            corpus_matches = [p for p in candidates if p.exists() and '64' in str(p)]
        if not corpus_matches:
            raise FileNotFoundError('Mounted t3d-kenney-corpus-64 dataset not found')
        manifest_path = corpus_matches[0]
    print(f'Using 64³ corpus manifest: {manifest_path}', flush=True)

    train_ds = AssetDataset(manifest_path, split='train')
    val_ds = AssetDataset(manifest_path, split='validation')
    limit = int(os.environ.get('T3D_LIMIT', '0'))
    if limit:
        train_ds.records = train_ds.records[:limit]
        val_ds.records = val_ds.records[:min(limit, len(val_ds.records))]
    print(f'Loaded {len(train_ds)} train, {len(val_ds)} validation assets (64³)', flush=True)

    grid_dim = 4
    grid_coords = torch.cartesian_prod(*(torch.arange(grid_dim, device=device) for _ in range(3)))

    cond_dim = 32
    print('Initializing CLIP prompt conditioner...', flush=True)
    conditioner = PromptConditioner.from_env(output_channels=cond_dim).to(device)

    # DiT Flow Backbone (head_dim = width // heads = 144 // 6 = 24, divisible by 6 for 3D RoPE; 2.8M params)
    model = SparseFlowDiT(channels=1, condition_channels=cond_dim, width=144, heads=6, depth=6, patch_size=64).to(device)
    total_steps = int(os.environ.get('T3D_STEPS', '4000'))
    batch_size = int(os.environ.get('T3D_BATCH', '4'))
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(conditioner.text_projection.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4)

    # Pre-cache raw text representations
    print('Extracting text embeddings for all 64³ corpus assets...', flush=True)
    raw_text_cache = {}
    with torch.no_grad():
        for rec in list(train_ds.records) + list(val_ds.records):
            prompt = id_to_prompt(rec.asset_id)
            inputs = conditioner.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
            tokens = conditioner.text_encoder(**inputs).last_hidden_state[0]
            raw_text_cache[rec.asset_id] = tokens.detach()

    # Pre-compute 4³ latent occupancy targets
    print('Precomputing 4³ latent occupancy targets for training set...', flush=True)
    occupancy_targets = {
        rec.asset_id: build_latent_occupancy(train_ds[i], grid_dim=grid_dim).to(device)
        for i, rec in enumerate(train_ds.records)
    }

    print(f'Starting 64³ Occupancy DiT Flow training on {len(train_ds)} assets ({total_steps} steps, batch size {batch_size}, 2.8M params with composite Tweedie + soft Dice + volume loss)...', flush=True)
    model.train()
    conditioner.train()

    rng = np.random.default_rng(42)
    history = []
    start = time.perf_counter()

    checkpoint_steps = sorted({max(1, total_steps * k // 4) for k in (1, 2, 3, 4)})
    out_dir = Path(os.environ.get('T3D_OUT', '/kaggle/working' if Path('/kaggle/working').exists() else 'artifacts/kaggle-kenney-64-occupancy-dit'))
    out_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        acc_loss = 0.0
        acc_flow = 0.0
        acc_tweedie = 0.0
        acc_dice = 0.0
        acc_vol = 0.0

        for _ in range(batch_size):
            idx = int(rng.integers(len(train_ds)))
            rec = train_ds.records[idx]
            target = occupancy_targets[rec.asset_id]
            tokens = conditioner.text_projection(raw_text_cache[rec.asset_id])

            time_t = torch.rand((), device=device)
            noise = torch.randn_like(target)
            noisy, v_target = flow_pair(target, noise, time_t)

            drop = torch.rand((), device=device) < 0.1
            cond_input = None if drop else tokens

            pred_v = model(noisy, grid_coords, time_t, cond_input)

            # Composite flow loss: velocity MSE + Tweedie clean target pos-weighted MSE + soft Dice loss + volume loss
            flow_loss = F.mse_loss(pred_v, v_target)
            pred_clean = noisy + (1.0 - time_t) * pred_v

            # Balanced Tweedie MSE
            pos_weight = torch.ones_like(target)
            tweedie_loss = torch.mean(pos_weight * (pred_clean - target) ** 2)

            # Soft Dice loss directly optimizing IoU/F1 of active occupancy cells
            pred_sig = torch.clamp(pred_clean, 0.0, 1.0)
            inter = (pred_sig * target).sum()
            denom = pred_sig.sum() + target.sum()
            dice_loss = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)

            # Explicit volume loss: penalize discrepancy between predicted active volume fraction and GT active volume fraction
            vol_loss = F.mse_loss(pred_clean.mean(), target.mean())

            loss = (flow_loss + 1.0 * tweedie_loss + 1.0 * dice_loss + 2.0 * vol_loss) / batch_size
            loss.backward()

            acc_loss += float(loss.detach()) * batch_size
            acc_flow += float(flow_loss.detach()) / batch_size
            acc_tweedie += float(tweedie_loss.detach()) / batch_size
            acc_dice += float(dice_loss.detach()) / batch_size
            acc_vol += float(vol_loss.detach()) / batch_size

        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(conditioner.text_projection.parameters()), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            history.append(dict(
                step=step, loss=round(acc_loss, 5),
                flow_loss=round(acc_flow, 5),
                tweedie_loss=round(acc_tweedie, 5),
                dice_loss=round(acc_dice, 5),
                vol_loss=round(acc_vol, 5),
                lr=float(scheduler.get_last_lr()[0])
            ))
            print(json.dumps(dict(
                step=step, loss=round(acc_loss, 5),
                flow=round(acc_flow, 5),
                tweedie=round(acc_tweedie, 5),
                dice=round(acc_dice, 5),
                vol=round(acc_vol, 5),
                lr=round(float(scheduler.get_last_lr()[0]), 6)
            )), flush=True)

        if step in checkpoint_steps:
            torch.save(dict(
                version=1,
                model=model.state_dict(),
                text_projection=conditioner.text_projection.state_dict(),
                manifest_sha256=train_ds.fingerprint,
                grid_size=grid_dim,
                resolution=64,
                width=144,
                heads=6,
                depth=6,
                step=step,
                models=conditioner.model_provenance
            ), out_dir / f'occupancy-dit-64-{step}.pt')

    print('Training complete. Running held-out validation evaluation on 64³ assets...', flush=True)
    model.eval()
    conditioner.eval()

    val_groups = {}
    for i, r in enumerate(val_ds.records):
        val_groups.setdefault(r.source_group, []).append(i)
    val_subset_indices = []
    for g in sorted(val_groups):
        val_subset_indices.extend(val_groups[g][:12])

    eval_results = []
    with torch.no_grad():
        for vi in val_subset_indices:
            rec = val_ds.records[vi]
            asset = val_ds[vi]
            target = build_latent_occupancy(asset, grid_dim=grid_dim).to(device)
            target_mask = target[:, 0] > 0.5

            prompt = id_to_prompt(rec.asset_id)
            tokens = conditioner.text_projection(raw_text_cache[rec.asset_id])

            mismatched_rec = train_ds.records[(vi * 17) % len(train_ds)]
            mismatched_prompt = id_to_prompt(mismatched_rec.asset_id)
            mismatched_tokens = conditioner.text_projection(raw_text_cache[mismatched_rec.asset_id])

            gen_noise = torch.randn((len(grid_coords), 1), device=device, generator=torch.Generator(device=device).manual_seed(9000 + vi))

            pred_correct = sample_flow(model, gen_noise, grid_coords, tokens, steps=20, guidance=2.5)
            correct_mask = pred_correct[:, 0] > 0.5
            if not correct_mask.any():
                topk = torch.topk(pred_correct[:, 0], k=min(16, len(pred_correct))).indices
                correct_mask[topk] = True

            pred_mismatched = sample_flow(model, gen_noise, grid_coords, mismatched_tokens, steps=20, guidance=2.5)
            mismatched_mask = pred_mismatched[:, 0] > 0.5
            if not mismatched_mask.any():
                topk = torch.topk(pred_mismatched[:, 0], k=min(16, len(pred_mismatched))).indices
                mismatched_mask[topk] = True

            def calc_metrics(pred_m, gt_m):
                inter = int((pred_m & gt_m).sum())
                union = int((pred_m | gt_m).sum())
                pred_cnt = int(pred_m.sum())
                gt_cnt = int(gt_m.sum())
                iou = inter / max(1, union)
                prec = inter / max(1, pred_cnt)
                rec_val = inter / max(1, gt_cnt)
                return dict(predicted=pred_cnt, target=gt_cnt, intersection=inter, iou=round(iou, 4), precision=round(prec, 4), recall=round(rec_val, 4))

            diff_count = int((correct_mask ^ mismatched_mask).sum())
            eval_results.append(dict(
                asset_id=rec.asset_id,
                group=rec.source_group,
                prompt=prompt,
                mismatched_prompt=mismatched_prompt,
                correct=calc_metrics(correct_mask, target_mask),
                mismatched=calc_metrics(mismatched_mask, target_mask),
                prompt_difference_cells=diff_count
            ))

    mean_iou = sum(e['correct']['iou'] for e in eval_results) / len(eval_results)
    mean_prec = sum(e['correct']['precision'] for e in eval_results) / len(eval_results)
    mean_rec = sum(e['correct']['recall'] for e in eval_results) / len(eval_results)
    mean_diff = sum(e['prompt_difference_cells'] for e in eval_results) / len(eval_results)
    print(json.dumps(dict(mean_val_occupancy_iou=round(mean_iou, 4), mean_val_precision=round(mean_prec, 4), mean_val_recall=round(mean_rec, 4), mean_prompt_diff_cells=round(mean_diff, 1))), flush=True)

    ckpt_path = out_dir / f'occupancy-dit-64-{total_steps}.pt'
    ckpt_payload = dict(
        version=1,
        model=model.state_dict(),
        text_projection=conditioner.text_projection.state_dict(),
        manifest_sha256=train_ds.fingerprint,
        grid_size=grid_dim,
        resolution=64,
        width=144,
        heads=6,
        depth=6,
        step=total_steps,
        models=conditioner.model_provenance
    )
    torch.save(ckpt_payload, ckpt_path)
    # Also save as 2000 for compatibility if needed
    first_checkpoint = out_dir / f'occupancy-dit-64-{checkpoint_steps[0]}.pt'
    if not first_checkpoint.exists():
        torch.save(ckpt_payload, first_checkpoint)

    report = dict(
        scope=f'Text-conditioned Stage 1 4³ latent occupancy DiT scaled to 2.8M params (width 144, heads 6, depth 6) trained on {len(train_ds)} Kenney 64³ train assets for {total_steps} steps with mini-batching (batch size {batch_size}) and composite flow + Tweedie + soft Dice + volume loss; evaluated on {len(eval_results)} held-out assets with prompt sensitivity checks',
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        elapsed_seconds=time.perf_counter() - start,
        train_count=len(train_ds),
        validation_eval_count=len(eval_results),
        mean_val_occupancy_iou=round(mean_iou, 4),
        mean_val_precision=round(mean_prec, 4),
        mean_val_recall=round(mean_rec, 4),
        mean_prompt_difference_cells=round(mean_diff, 1),
        checkpoint_sha256=hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        history=history,
        eval_results=eval_results
    )
    (out_dir / 'occupancy-dit-64-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Report written to occupancy-dit-64-report.json', flush=True)


if __name__ == '__main__':
    main()
