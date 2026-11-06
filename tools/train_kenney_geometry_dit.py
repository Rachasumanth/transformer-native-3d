"""Train Stage 2 text-conditioned shape latent geometry DiT on the 868-asset Kenney corpus."""
import hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from t3d.conditioning import PromptConditioner
from t3d.dataset import AssetDataset, batch_arrays
from t3d.extraction import voxels_to_mesh
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

    # Locate dataset
    matches = list(Path('/kaggle/input').rglob('manifest.json'))
    corpus_matches = [p for p in matches if 't3d-kenney-corpus-32' in str(p) or (p.parent / 'kenney-car-kit-ambulance.t3d').exists()]
    if not corpus_matches:
        raise FileNotFoundError('Mounted t3d-kenney-corpus-32 dataset not found')
    manifest_path = corpus_matches[0]
    print(f'Using corpus manifest: {manifest_path}', flush=True)

    train_ds = AssetDataset(manifest_path, split='train')
    val_ds = AssetDataset(manifest_path, split='validation')
    print(f'Loaded {len(train_ds)} train, {len(val_ds)} validation assets', flush=True)

    # Locate trained SC-VAE checkpoint
    ck_matches = list(Path('/kaggle/input').rglob('triton-kenney-corpus-1900.pt'))
    if not ck_matches:
        # Fallback to other available checkpoints
        ck_matches = list(Path('/kaggle/input').rglob('*.pt'))
    vae_ckpt_path = [p for p in ck_matches if 'triton' in p.name or 'kenney' in p.name or 'shape' in p.name][0]
    print(f'Using SC-VAE checkpoint: {vae_ckpt_path}', flush=True)

    # Load frozen SC-VAE
    vae_data = torch.load(vae_ckpt_path, map_location=device, weights_only=False)
    cfg = {k: v for k, v in vae_data['config'].items() if k in ('mode', 'channels', 'latent_channels', 'levels', 'downsample_block', 'upsample_block', 'convolution_backend')}
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

    # Stage 2 Geometry DiT (latent channels from SC-VAE, width 96, 4 heads -> head_dim 24 divisible by 6)
    latent_channels = int(vae_data['config']['latent_channels'])
    model = SparseFlowDiT(channels=latent_channels, condition_channels=cond_dim, width=96, heads=4, depth=4, patch_size=64).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(conditioner.text_projection.parameters()), lr=1e-3, weight_decay=1e-4)

    # Pre-cache text tokens and pre-encode shape latents
    print('Extracting text embeddings and encoding shape latents for training models...', flush=True)
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

    print(f'Encoded latents for {len(encoded_latents)} training assets. Starting Stage 2 DiT training...', flush=True)
    model.train()
    conditioner.train()

    rng = np.random.default_rng(42)
    history = []
    start = time.perf_counter()

    for step in range(1, 601):
        idx = int(rng.integers(len(train_ds)))
        rec = train_ds.records[idx]
        coords, target_latents = encoded_latents[rec.asset_id]
        tokens = conditioner.text_projection(raw_text_cache[rec.asset_id])

        time_t = torch.rand((), device=device)
        noise = torch.randn_like(target_latents)
        noisy, v_target = flow_pair(target_latents, noise, time_t)

        drop = torch.rand((), device=device) < 0.1
        cond_input = None if drop else tokens

        optimizer.zero_grad(set_to_none=True)
        pred_v = model(noisy, coords, time_t, cond_input)
        loss = torch.nn.functional.mse_loss(pred_v, v_target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(conditioner.text_projection.parameters()), 1.0)
        optimizer.step()

        if step % 50 == 0:
            rec_loss = float(loss.detach())
            history.append(dict(step=step, asset_id=rec.asset_id, loss=rec_loss))
            print(json.dumps(dict(step=step, asset=rec.asset_id, loss=round(rec_loss, 5))), flush=True)

    print('Training complete. Running held-out validation evaluation and mesh generation...', flush=True)
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

            # Decode into mesh
            batched_coords = posterior.coords
            out, raw, _ = vae_model.decode(batched_coords, pred_latents, threshold=0.5)
            mesh_verts, mesh_tris = 0, 0
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
                    v, faces = voxels_to_mesh(decoded_asset)
                    mesh_verts, mesh_tris = len(v), len(faces)

            eval_results.append(dict(
                asset_id=rec.asset_id,
                group=rec.source_group,
                prompt=prompt,
                latent_mse=round(latent_mse, 5),
                decoded_voxels=len(out),
                mesh_triangles=mesh_tris
            ))

    mean_mse = sum(e['latent_mse'] for e in eval_results) / len(eval_results)
    mean_tris = sum(e['mesh_triangles'] for e in eval_results) / len(eval_results)
    print(json.dumps(dict(mean_val_latent_mse=round(mean_mse, 5), mean_mesh_triangles=round(mean_tris, 1))), flush=True)

    out_dir = Path('/kaggle/working')
    ckpt_path = out_dir / 'geometry-dit-600.pt'
    torch.save(dict(
        version=1,
        model=model.state_dict(),
        text_projection=conditioner.text_projection.state_dict(),
        manifest_sha256=train_ds.fingerprint,
        latent_channels=latent_channels,
        models=conditioner.model_provenance
    ), ckpt_path)

    report = dict(
        scope='Text-conditioned Stage 2 shape latent geometry DiT trained on 649 Kenney train assets; evaluated on 16 held-out assets with latent MSE and mesh extraction',
        device=torch.cuda.get_device_name(0),
        elapsed_seconds=time.perf_counter() - start,
        train_count=len(train_ds),
        validation_eval_count=len(eval_results),
        mean_val_latent_mse=mean_mse,
        mean_mesh_triangles=mean_tris,
        checkpoint_sha256=hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        history=history,
        eval_results=eval_results
    )
    (out_dir / 'geometry-dit-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Report written to geometry-dit-report.json', flush=True)


if __name__ == '__main__':
    main()
