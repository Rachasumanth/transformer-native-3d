"""Train Stage 3 Shape-Conditioned Material Flow DiT on the multi-asset PBR corpus."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F

from t3d.conditioning import PromptConditioner
from t3d.dataset import AssetDataset, batch_arrays
from t3d.flow import SparseFlowDiT, flow_pair, sample_flow
from t3d.sc_vae import SparseCompressionVAE


def asset_id_to_prompt(asset_id: str) -> str:
    name = asset_id.lower()
    if 'avocado' in name:
        return 'a fresh green ripe avocado fruit with seed and textured skin'
    elif 'waterbottle' in name:
        return 'a stainless steel insulated sports water bottle with blue strap'
    elif 'fish' in name:
        return 'a silvery barramundi fish with detailed scales and fins'
    elif 'duck' in name:
        return 'a yellow rubber bath duck toy'
    elif 'boombox' in name:
        return 'a retro 1980s portable boombox cassette player stereo'
    elif 'box' in name:
        return 'a brown wooden storage crate shipping container'
    elif 'lantern' in name:
        return 'an antique iron metal lantern lamp with glass panels'
    return f'a 3d object of {asset_id}'


def evaluate_validation(model, conditioner, val_ds, cached_tokens, cached_latents, mat_vae, device, steps=20, guidance=3.0):
    model.eval()
    val_latent_mses = []
    val_channel_maes = []

    with torch.no_grad():
        for i in range(len(val_ds)):
            rec = val_ds.records[i]
            asset = val_ds[i]
            coords, shape_latents, target_mat_latents, shape_occ = cached_latents[rec.asset_id]
            tokens = conditioner.text_projection(cached_tokens[rec.asset_id])

            noise = torch.randn_like(target_mat_latents)
            pred_mat = sample_flow(
                model, noise, coords, tokens,
                shape=shape_latents, steps=steps, guidance=guidance
            )

            latent_mse = float(F.mse_loss(pred_mat, target_mat_latents).item())
            val_latent_mses.append(latent_mse)

            # Decode predicted material latents through frozen Material SC-VAE
            batched_coords = torch.cat([torch.zeros((len(coords), 1), device=device, dtype=coords.dtype), coords], dim=1)
            out_mat, raw_mat, _ = mat_vae.decode(batched_coords, pred_mat, teacher_occupancy=shape_occ)

            c_mat, f_mat = batch_arrays([asset], 'material')
            sorted_coords, inverse = torch.unique(torch.from_numpy(c_mat).to(device), dim=0, sorted=True, return_inverse=True)
            target_pbr = torch.from_numpy(f_mat).to(device)[torch.argsort(inverse)]

            pbr_mae = (raw_mat.sigmoid() - target_pbr).abs().mean(dim=0).cpu().numpy()
            val_channel_maes.append(pbr_mae)

    mean_latent_mse = float(np.mean(val_latent_mses))
    mean_maes = np.mean(val_channel_maes, axis=0)
    mae_dict = dict(zip(('r', 'g', 'b', 'metallic', 'roughness', 'alpha'), [float(x) for x in mean_maes]))
    mae_dict['mean'] = float(np.mean(mean_maes))

    return {
        'latent_mse': round(mean_latent_mse, 5),
        'pbr_mae': mae_dict
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Stage 3 Material Flow DiT training on {device}...", flush=True)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Locate dataset manifest
    possible_manifests = [
        Path('/kaggle/working/material-train/artifacts/material-multiasset-32/manifest.json'),
        Path('/kaggle/working/artifacts/material-multiasset-32/manifest.json'),
        Path('artifacts/material-multiasset-32/manifest.json'),
        *Path('/kaggle/input').rglob('manifest.json')
    ]
    manifest_path = None
    for p in possible_manifests:
        if p.exists() and ('material' in str(p) or (p.parent / 'avocado-rot0.t3d').exists()):
            manifest_path = p
            break

    if manifest_path is None:
        raise FileNotFoundError("Could not find multiasset material manifest.json")
    print(f"Using manifest: {manifest_path}", flush=True)

    train_ds = AssetDataset(manifest_path, split='train')
    val_ds = AssetDataset(manifest_path, split='validation')
    print(f"Loaded {len(train_ds)} train assets, {len(val_ds)} validation assets", flush=True)

    # 2. Locate frozen Shape SC-VAE and Material SC-VAE checkpoints
    all_ckpts = list(Path('/kaggle/input').rglob('*.pt')) + list(Path('artifacts').rglob('*.pt'))
    shape_ckpts = [p for p in all_ckpts if 'triton-kenney-corpus-1900' in p.name or 'shape' in p.name or 'triton' in p.name]
    mat_ckpts = [p for p in all_ckpts if 'material-multiasset-300' in p.name or 'material-multiasset' in p.name]

    if not shape_ckpts:
        raise FileNotFoundError("Shape SC-VAE checkpoint not found")
    if not mat_ckpts:
        raise FileNotFoundError("Material SC-VAE checkpoint not found")

    shape_ckpt_path = shape_ckpts[0]
    mat_ckpt_path = mat_ckpts[0]
    print(f"Using Shape SC-VAE: {shape_ckpt_path}", flush=True)
    print(f"Using Material SC-VAE: {mat_ckpt_path}", flush=True)

    # Load frozen Shape SC-VAE
    backend = 'triton' if torch.cuda.is_available() else 'reference'
    shape_data = torch.load(shape_ckpt_path, map_location=device, weights_only=False)
    s_cfg = {k: v for k, v in shape_data['config'].items() if k in ('mode', 'channels', 'latent_channels', 'levels', 'downsample_block', 'upsample_block', 'convolution_backend')}
    s_cfg['convolution_backend'] = backend
    shape_vae = SparseCompressionVAE(**s_cfg).to(device).eval()
    shape_vae.load_state_dict(shape_data['model'])
    for p in shape_vae.parameters():
        p.requires_grad = False

    # Load frozen Material SC-VAE
    mat_data = torch.load(mat_ckpt_path, map_location=device, weights_only=False)
    m_cfg = {k: v for k, v in mat_data['config'].items() if k in ('mode', 'channels', 'latent_channels', 'levels', 'downsample_block', 'upsample_block', 'convolution_backend')}
    m_cfg['convolution_backend'] = backend
    mat_vae = SparseCompressionVAE(**m_cfg).to(device).eval()
    mat_vae.load_state_dict(mat_data['model'])
    for p in mat_vae.parameters():
        p.requires_grad = False

    # 3. Prompt conditioner
    cond_dim = 32
    print("Initializing CLIP PromptConditioner...", flush=True)
    conditioner = PromptConditioner.from_pretrained(
        image_model='facebook/dinov2-small',
        image_revision='ed25f3a31f01632728cabb09d1542f84ab7b0056',
        text_model='openai/clip-vit-base-patch32',
        text_revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
        output_channels=cond_dim
    ).to(device)

    # If occupancy or geometry checkpoint exists, load initialized text projection
    occ_ckpts = [p for p in all_ckpts if 'occupancy-dit-600' in p.name or 'occupancy' in p.name]
    if occ_ckpts:
        try:
            occ_data = torch.load(occ_ckpts[0], map_location=device, weights_only=False)
            if 'text_projection' in occ_data:
                conditioner.text_projection.load_state_dict(occ_data['text_projection'])
                print(f"Loaded initialized text_projection from {occ_ckpts[0]}", flush=True)
        except Exception as e:
            print(f"Warning: could not load text_projection from occupancy checkpoint: {e}", flush=True)

    # 4. Initialize Stage 3 SparseFlowDiT
    latent_channels = int(mat_data['config']['latent_channels'])
    model = SparseFlowDiT(
        channels=latent_channels,
        condition_channels=cond_dim,
        width=96,
        heads=4,
        depth=4,
        patch_size=64,
        shape_channels=latent_channels
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Initialized Stage 3 Material Flow DiT ({param_count:,} parameters)", flush=True)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(conditioner.text_projection.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )

    # 5. Pre-extract text tokens and pre-encode shape & material latents
    print("Pre-encoding latents and text embeddings...", flush=True)
    cached_tokens = {}
    cached_latents = {}

    with torch.no_grad():
        all_records = list(train_ds.records) + list(val_ds.records)
        for rec in all_records:
            prompt = asset_id_to_prompt(rec.asset_id)
            inputs = conditioner.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
            tokens = conditioner.text_encoder(**inputs).last_hidden_state[0]
            cached_tokens[rec.asset_id] = tokens.detach()

        for ds in (train_ds, val_ds):
            for i in range(len(ds)):
                rec = ds.records[i]
                asset = ds[i]

                c_shape, f_shape = batch_arrays([asset], 'shape')
                c_mat, f_mat = batch_arrays([asset], 'material')

                c_shape_t = torch.from_numpy(c_shape).to(device)
                f_shape_t = torch.from_numpy(f_shape).to(device)
                c_mat_t = torch.from_numpy(c_mat).to(device)
                f_mat_t = torch.from_numpy(f_mat).to(device)

                shape_post = shape_vae.encode(c_shape_t, f_shape_t)
                mat_post = mat_vae.encode(c_mat_t, f_mat_t)

                coords = shape_post.coords[:, 1:].detach()
                shape_lat = shape_post.mean.detach()
                mat_lat = mat_post.mean.detach()
                shape_occ = [m.detach() for m in shape_post.occupancy]

                cached_latents[rec.asset_id] = (coords, shape_lat, mat_lat, shape_occ)

    print(f"Pre-encoded {len(cached_latents)} assets. Starting Stage 3 DiT training...", flush=True)

    # 6. Training loop
    model.train()
    conditioner.train()
    rng = np.random.default_rng(42)
    history = []
    total_steps = 600
    start_time = time.perf_counter()

    for step in range(1, total_steps + 1):
        idx = int(rng.integers(len(train_ds)))
        rec = train_ds.records[idx]
        coords, shape_lat, target_mat, _ = cached_latents[rec.asset_id]

        tokens = conditioner.text_projection(cached_tokens[rec.asset_id])

        # Flow matching loss with classifier-free guidance drop probability 0.1
        t = torch.rand((), device=device, dtype=target_mat.dtype)
        noise = torch.randn_like(target_mat)
        noisy_mat, target_v = flow_pair(target_mat, noise, t)

        drop = torch.rand((), device=device) < 0.1
        cond_input = None if drop else tokens

        pred_v = model(noisy_mat, coords, t, cond_input, shape=shape_lat)
        loss = F.mse_loss(pred_v, target_v)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 150 == 0 or step == total_steps:
            val_results = evaluate_validation(
                model, conditioner, val_ds, cached_tokens, cached_latents, mat_vae, device
            )
            elapsed = time.perf_counter() - start_time
            record = {
                'step': step,
                'train_loss': round(float(loss.item()), 5),
                'val_latent_mse': val_results['latent_mse'],
                'val_pbr_mae': val_results['pbr_mae'],
                'elapsed_seconds': round(elapsed, 2)
            }
            history.append(record)
            print(
                f"Step {step:04d}/{total_steps:04d} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Latent MSE: {val_results['latent_mse']:.4f} | "
                f"Val PBR MAE: {val_results['pbr_mae']['mean']:.4f} | "
                f"Time: {elapsed:.1f}s",
                flush=True
            )

            # Save checkpoint
            out_ckpt = Path(f"/kaggle/working/material-flow-dit-{step}.pt")
            if not out_ckpt.parent.exists():
                out_ckpt = Path(f"artifacts/material-flow-dit-{step}.pt")
                out_ckpt.parent.mkdir(parents=True, exist_ok=True)

            ckpt_payload = {
                'version': 1,
                'step': step,
                'model': model.state_dict(),
                'text_projection': conditioner.text_projection.state_dict(),
                'config': {
                    'channels': latent_channels,
                    'condition_channels': cond_dim,
                    'shape_channels': latent_channels,
                    'width': 96,
                    'heads': 4,
                    'depth': 4,
                    'patch_size': 64
                },
                'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            }
            torch.save(ckpt_payload, out_ckpt)
            print(f"Saved checkpoint: {out_ckpt} ({out_ckpt.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)

    # 7. Write final training report
    report_path = Path("/kaggle/working/material-flow-dit-report.json")
    if not report_path.parent.exists():
        report_path = Path("artifacts/material-flow-dit-report.json")

    report_data = {
        'model': 'SparseFlowDiT-Stage3-Material',
        'device': str(device),
        'parameters': param_count,
        'train_assets': len(train_ds),
        'validation_assets': len(val_ds),
        'total_steps': total_steps,
        'history': history,
        'final_checkpoint': f"material-flow-dit-{total_steps}.pt",
        'final_val_latent_mse': history[-1]['val_latent_mse'],
        'final_val_pbr_mae': history[-1]['val_pbr_mae']
    }
    report_path.write_text(json.dumps(report_data, indent=2) + '\n')
    print(f"Wrote report to {report_path}", flush=True)


if __name__ == '__main__':
    main()
