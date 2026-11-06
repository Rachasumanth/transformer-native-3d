"""Train Stage 2 text-conditioned shape latent geometry DiT on the 64³ Kenney corpus."""
import hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from t3d.conditioning import PromptConditioner
from t3d.dataset import AssetDataset, batch_arrays
from t3d.extraction import voxels_to_mesh
from t3d.metrics import surface_comparison
from t3d.flow import SparseFlowDiT, flow_pair, sample_flow
from t3d.sc_vae import SparseCompressionVAE
from t3d.schema import VoxelAsset
from tools.generate_corpus_prompts import id_to_prompt


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for DiT geometry training')

    device = 'cuda'
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Locate 64³ dataset
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
    print(f'Loaded {len(train_ds)} train, {len(val_ds)} validation assets (64³)', flush=True)

    # Locate trained 64³ SC-VAE checkpoint
    ckroot_candidates = [
        *list(Path('/kaggle/input').rglob('triton-kenney-corpus-64-*.pt')),
        *list(Path('artifacts/kaggle-kenney-corpus-64-training').rglob('*.pt')),
        *list(Path('/kaggle/input').rglob('*.pt'))
    ]
    vae_ckpts = [p for p in ckroot_candidates if '64' in p.name and 'triton' in p.name]
    if not vae_ckpts:
        # Fallback to any 64 shape checkpoint or latest
        vae_ckpts = [p for p in ckroot_candidates if 'triton' in p.name or 'kenney' in p.name]
    if not vae_ckpts:
        raise FileNotFoundError('Trained 64³ SC-VAE checkpoint not found')
    
    import re
    def _step_key(p):
        m = re.search(r'(\d+)\.pt$', p.name)
        return int(m.group(1)) if m else 0

    vae_ckpt_path = sorted(vae_ckpts, key=_step_key)[-1]
    print(f'Using 64³ SC-VAE checkpoint: {vae_ckpt_path}', flush=True)

    # Load frozen SC-VAE
    vae_data = torch.load(vae_ckpt_path, map_location=device, weights_only=False)
    cfg = {k: v for k, v in vae_data['config'].items() if k in ('mode', 'channels', 'latent_channels', 'levels', 'downsample_block', 'upsample_block', 'convolution_backend')}
    # Infer true channels directly from model weights to guard against metadata discrepancy
    if 'input.weight' in vae_data['model']:
        enc_blocks = max(int(k.split('.')[1]) for k in vae_data['model'] if k.startswith('encoder_blocks.'))
        channels = tuple(vae_data['model'][f'encoder_blocks.{i}.scale'].shape[0] for i in range(enc_blocks + 1))
        cfg['channels'] = channels

    vae_model = SparseCompressionVAE(**cfg).to(device).eval()
    vae_model.load_state_dict(vae_data['model'])
    for param in vae_model.parameters():
        param.requires_grad = False

    cond_dim = 32
    print('Initializing CLIP prompt conditioner...', flush=True)
    conditioner = PromptConditioner.from_pretrained(
        image_model='facebook/dinov2-small',
        image_revision='ed25f3a31f01632728cabb09d1542f84ab7b0056',
        text_model='openai/clip-vit-base-patch32',
        text_revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
        output_channels=cond_dim
    ).to(device)

    # Stage 2 Geometry DiT (latent channels from SC-VAE, width 144, 6 heads -> head_dim 24 divisible by 6, depth 6 -> 2.8M params)
    latent_channels = int(vae_data['config']['latent_channels'])
    model_width = 144
    model_heads = 6
    model_depth = 6
    model = SparseFlowDiT(channels=latent_channels, condition_channels=cond_dim, width=model_width, heads=model_heads, depth=model_depth, patch_size=64).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(conditioner.text_projection.parameters()), lr=1e-3, weight_decay=1e-4)
    total_steps = 4000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=5e-5)

    # Pre-cache text tokens and pre-encode shape latents
    print('Extracting text embeddings and encoding shape latents for 64³ training models...', flush=True)
    raw_text_cache = {}
    encoded_latents = {}

    with torch.no_grad():
        for rec in list(train_ds.records) + list(val_ds.records):
            prompt = id_to_prompt(rec.asset_id)
            inputs = conditioner.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
            tokens = conditioner.text_encoder(**inputs).last_hidden_state[0]
            raw_text_cache[rec.asset_id] = tokens.detach()

        for i, rec in enumerate(train_ds.records):
            asset = train_ds[i]
            c, f = batch_arrays([asset], 'shape')
            c_t = torch.from_numpy(c).to(device)
            f_t = torch.from_numpy(f).to(device)
            posterior = vae_model.encode(c_t, f_t)
            # Use posterior mean as training target
            encoded_latents[rec.asset_id] = (posterior.coords[:, 1:].detach(), posterior.mean.detach())

    print(f'Encoded latents for {len(encoded_latents)} training assets (64³). Starting Stage 2 DiT training...', flush=True)
    model.train()
    conditioner.train()

    rng = np.random.default_rng(42)
    history = []
    start = time.perf_counter()

    out_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('artifacts/kaggle-kenney-64-geometry-dit')
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_checkpoints = {}

    for step in range(1, total_steps + 1):
        idx = int(rng.integers(len(train_ds)))
        rec = train_ds.records[idx]
        coords, target_latents = encoded_latents[rec.asset_id]
        tokens = conditioner.text_projection(raw_text_cache[rec.asset_id])

        # Coordinate subsampling augmentation (simulate Stage 1 partial recall)
        if len(coords) > 8 and torch.rand((), device=device) < 0.20:
            sub_k = int(rng.integers(int(len(coords) * 0.85), len(coords)))
            sub_idx = torch.randperm(len(coords), device=device)[:sub_k]
            coords = coords[sub_idx]
            target_latents = target_latents[sub_idx]

        # Background distractor coordinate augmentation (simulate Stage 1 false-positive cells)
        has_distractors = False
        if len(coords) < 60 and torch.rand((), device=device) < 0.85:
            all_grid = torch.cartesian_prod(*(torch.arange(4, device=device) for _ in range(3)))
            dist_matrix = (all_grid[:, None, :] == coords[None, :, :]).all(dim=-1)
            bg_mask = ~dist_matrix.any(dim=1)
            bg_indices = bg_mask.nonzero(as_tuple=False).squeeze(-1)
            if len(bg_indices) > 0:
                k_bg = min(int(rng.integers(4, 12)), len(bg_indices))
                chosen_bg = bg_indices[torch.randperm(len(bg_indices), device=device)[:k_bg]]
                bg_coords = all_grid[chosen_bg]
                bg_latents = torch.zeros((k_bg, target_latents.shape[1]), device=device)
                coords = torch.cat([coords, bg_coords], dim=0)
                target_latents = torch.cat([target_latents, bg_latents], dim=0)
                has_distractors = True

        time_t = torch.rand((), device=device)
        noise = torch.randn_like(target_latents)
        noisy, v_target = flow_pair(target_latents, noise, time_t)

        drop = torch.rand((), device=device) < 0.1
        cond_input = None if drop else tokens

        optimizer.zero_grad(set_to_none=True)
        pred_v = model(noisy, coords, time_t, cond_input)
        flow_loss = torch.nn.functional.mse_loss(pred_v, v_target)

        # Composite flow loss: velocity MSE + Tweedie clean target reconstruction + distractor zero suppression
        pred_clean = noisy + (1.0 - time_t) * pred_v
        tweedie_loss = torch.nn.functional.mse_loss(pred_clean, target_latents)
        if has_distractors:
            zero_mask = target_latents.abs().sum(dim=-1) == 0
            zero_loss = torch.nn.functional.mse_loss(pred_clean[zero_mask], target_latents[zero_mask]) if zero_mask.any() else torch.tensor(0.0, device=device)
        else:
            zero_loss = torch.tensor(0.0, device=device)

        loss = flow_loss + 1.0 * tweedie_loss + 2.0 * zero_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(conditioner.text_projection.parameters()), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            rec_loss = float(loss.detach())
            flow_l = float(flow_loss.detach())
            tw_l = float(tweedie_loss.detach())
            z_l = float(zero_loss.detach())
            history.append(dict(step=step, asset_id=rec.asset_id, loss=rec_loss, flow_loss=flow_l, tweedie_loss=tw_l, zero_loss=z_l))
            print(json.dumps(dict(step=step, asset=rec.asset_id, loss=round(rec_loss, 5), flow=round(flow_l, 5), tweedie=round(tw_l, 5), zero=round(z_l, 5), lr=round(scheduler.get_last_lr()[0], 6))), flush=True)

        if step in (1000, 2000, 3000, 4000):
            ckpt_path = out_dir / f'geometry-dit-64-{step}.pt'
            torch.save(dict(
                version=1,
                step=step,
                model=model.state_dict(),
                text_projection=conditioner.text_projection.state_dict(),
                manifest_sha256=train_ds.fingerprint,
                latent_channels=latent_channels,
                resolution=64,
                width=model_width,
                heads=model_heads,
                depth=model_depth,
                models=conditioner.model_provenance
            ), ckpt_path)
            sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
            saved_checkpoints[ckpt_path.name] = sha
            print(f"Saved checkpoint: {ckpt_path.name} (SHA-256: {sha})", flush=True)

    print('Training complete. Running held-out validation evaluation and mesh generation on 64³ assets...', flush=True)
    model.eval()
    conditioner.eval()

    val_groups = {}
    for i, r in enumerate(val_ds.records):
        val_groups.setdefault(r.source_group, []).append(i)
    val_subset_indices = []
    for g in sorted(val_groups):
        val_subset_indices.extend(val_groups[g][:8])

    eval_results = []
    with torch.no_grad():
        for vi in val_subset_indices:
            rec = val_ds.records[vi]
            asset = val_ds[vi]
            c, f = batch_arrays([asset], 'shape')
            c_t = torch.from_numpy(c).to(device)
            f_t = torch.from_numpy(f).to(device)
            posterior = vae_model.encode(c_t, f_t)
            gt_coords = posterior.coords[:, 1:]
            gt_latents = posterior.mean

            prompt = id_to_prompt(rec.asset_id)
            tokens = conditioner.text_projection(raw_text_cache[rec.asset_id])

            gen_noise = torch.randn_like(gt_latents)
            pred_latents = sample_flow(model, gen_noise, gt_coords, tokens, steps=20, guidance=2.5)
            latent_mse = float(torch.nn.functional.mse_loss(pred_latents, gt_latents).item())

            # Evaluate steerability by comparing against mismatched prompt
            mismatched_prompt = "an alien sci-fi space asset" if "pirate" in prompt else "a barrel pirate theme asset"
            alt_inputs = conditioner.tokenizer(mismatched_prompt, return_tensors='pt', padding=False, truncation=True)
            alt_inputs = {k: v.to(device) for k, v in alt_inputs.items() if torch.is_tensor(v)}
            alt_raw = conditioner.text_encoder(**alt_inputs).last_hidden_state[0]
            alt_tokens = conditioner.text_projection(alt_raw)
            alt_latents = sample_flow(model, gen_noise, gt_coords, alt_tokens, steps=20, guidance=2.5)
            steerability_delta = float(torch.nn.functional.mse_loss(pred_latents, alt_latents).item())

            # Decode into mesh and evaluate surface metrics
            batched_coords = posterior.coords
            out, raw, _ = vae_model.decode(batched_coords, pred_latents, threshold=0.5)
            mesh_verts, mesh_tris = 0, 0
            mesh_dist, mesh_fscore = None, None
            if len(out) > 0:
                valid = (out[:, 1:] < out.new_tensor(asset.grid_size)).all(dim=1)
                out, raw = out[valid], raw[valid]
                if len(out) > 0:
                    decoded_asset = VoxelAsset(
                        out[:, 1:].cpu().numpy().astype(int),
                        raw[:, :3].sigmoid().cpu().numpy(),
                        (raw[:, 3:6].sigmoid() > 0.5).cpu().numpy(),
                        asset.grid_size, asset.aabb,
                        split_weight=raw[:, 6:].cpu().numpy()
                    )
                    v, faces = voxels_to_mesh(decoded_asset, resolve_nonmanifold=True)
                    mesh_verts, mesh_tris = len(v), len(faces)
                    gt_v, gt_faces = voxels_to_mesh(asset)
                    comp = surface_comparison(gt_v, gt_faces, v, faces, samples=512, tolerance=0.01)
                    mesh_dist = round(float(comp['mean_distance']), 5) if comp.get('valid') and 'mean_distance' in comp else None
                    mesh_fscore = round(float(comp['fscore']), 4) if comp.get('valid') and 'fscore' in comp else None

            eval_results.append(dict(
                asset_id=rec.asset_id,
                group=rec.source_group,
                prompt=prompt,
                latent_mse=round(latent_mse, 5),
                steerability_delta=round(steerability_delta, 5),
                decoded_voxels=len(out),
                mesh_triangles=mesh_tris,
                surface_distance=mesh_dist,
                fscore_001=mesh_fscore
            ))

    mean_mse = sum(e['latent_mse'] for e in eval_results) / len(eval_results)
    mean_steer = sum(e['steerability_delta'] for e in eval_results) / len(eval_results)
    mean_tris = sum(e['mesh_triangles'] for e in eval_results) / len(eval_results)
    valid_dists = [e['surface_distance'] for e in eval_results if e['surface_distance'] is not None]
    valid_fscores = [e['fscore_001'] for e in eval_results if e['fscore_001'] is not None]
    mean_dist = sum(valid_dists) / len(valid_dists) if valid_dists else None
    mean_fscore = sum(valid_fscores) / len(valid_fscores) if valid_fscores else None
    print(json.dumps(dict(
        mean_val_latent_mse=round(mean_mse, 5),
        mean_steerability_delta=round(mean_steer, 5),
        mean_mesh_triangles=round(mean_tris, 1),
        mean_surface_distance=round(mean_dist, 5) if mean_dist is not None else None,
        mean_fscore_001=round(mean_fscore, 4) if mean_fscore is not None else None
    )), flush=True)

    final_ckpt = out_dir / f'geometry-dit-64-{total_steps}.pt'
    final_sha = saved_checkpoints.get(final_ckpt.name, '')

    report = dict(
        scope='Text-conditioned Stage 2 shape latent geometry DiT (2.8M param, width 144, heads 6, depth 6) trained on 649 Kenney train assets at 64³ with composite flow + Tweedie loss + distractor zero-suppression; evaluated on 16 held-out assets with latent MSE, steerability, and mesh extraction',
        device=torch.cuda.get_device_name(0),
        elapsed_seconds=time.perf_counter() - start,
        total_steps=total_steps,
        model_parameters=sum(p.numel() for p in model.parameters()),
        model_width=model_width,
        model_heads=model_heads,
        model_depth=model_depth,
        train_count=len(train_ds),
        validation_eval_count=len(eval_results),
        mean_val_latent_mse=mean_mse,
        mean_steerability_delta=mean_steer,
        mean_mesh_triangles=mean_tris,
        mean_surface_distance=mean_dist,
        mean_fscore_001=mean_fscore,
        final_checkpoint=final_ckpt.name,
        final_checkpoint_sha256=final_sha,
        saved_checkpoints=saved_checkpoints,
        history=history,
        eval_results=eval_results
    )
    (out_dir / 'geometry-dit-64-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Report written to geometry-dit-64-report.json', flush=True)


if __name__ == '__main__':
    main()
