"""Train Stage 1 text-conditioned coarse occupancy DiT on the 868-asset Kenney corpus."""
import hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from t3d.conditioning import PromptConditioner
from t3d.dataset import AssetDataset
from t3d.flow import SparseFlowDiT, flow_pair, sample_flow
from tools.generate_corpus_prompts import id_to_prompt


def build_coarse_occupancy(asset, coarse_grid=16, input_grid=32):
    """Downsample active sparse voxel coordinates into a coarse binary occupancy grid."""
    scale = input_grid // coarse_grid
    coarse_coords = asset.coords // scale
    unique_coords = np.unique(coarse_coords, axis=0)
    flat_indices = (unique_coords[:, 0] * coarse_grid + unique_coords[:, 1]) * coarse_grid + unique_coords[:, 2]
    occupancy = np.zeros((coarse_grid ** 3, 1), dtype=np.float32)
    occupancy[flat_indices] = 1.0
    return torch.from_numpy(occupancy)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for DiT flow training')

    device = 'cuda'
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Discover dataset
    matches = list(Path('/kaggle/input').rglob('manifest.json'))
    corpus_matches = [p for p in matches if 't3d-kenney-corpus-32' in str(p) or (p.parent / 'kenney-car-kit-ambulance.t3d').exists()]
    if not corpus_matches:
        raise FileNotFoundError('Mounted t3d-kenney-corpus-32 dataset not found')
    manifest_path = corpus_matches[0]
    corpus_dir = manifest_path.parent
    print(f'Using corpus manifest: {manifest_path}', flush=True)

    train_ds = AssetDataset(manifest_path, split='train')
    val_ds = AssetDataset(manifest_path, split='validation')
    print(f'Loaded {len(train_ds)} train, {len(val_ds)} validation assets', flush=True)

    coarse_grid = 16
    grid_coords = torch.cartesian_prod(*(torch.arange(coarse_grid, device=device) for _ in range(3)))

    # Condition channels and model setup
    cond_dim = 32
    print('Initializing CLIP prompt conditioner...', flush=True)
    conditioner = PromptConditioner.from_pretrained(
        image_model='facebook/dinov2-small',
        image_revision='ed25f3a31f01632728cabb09d1542f84ab7b0056',
        text_model='openai/clip-vit-base-patch32',
        text_revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
        output_channels=cond_dim
    ).to(device)

    # DiT Flow Backbone (head_dim = width // heads must be divisible by 6 for 3D RoPE)
    model = SparseFlowDiT(channels=1, condition_channels=cond_dim, width=96, heads=4, depth=3, patch_size=64).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(conditioner.text_projection.parameters()), lr=1e-3, weight_decay=1e-4)

    # Pre-cache raw text representations for faster training steps
    print('Extracting text embeddings for all corpus assets...', flush=True)
    raw_text_cache = {}
    with torch.no_grad():
        for rec in list(train_ds.records) + list(val_ds.records):
            prompt = id_to_prompt(rec.asset_id)
            inputs = conditioner.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}
            tokens = conditioner.text_encoder(**inputs).last_hidden_state[0]
            raw_text_cache[rec.asset_id] = tokens.detach()

    # Pre-compute coarse occupancy targets
    print('Precomputing coarse occupancy targets...', flush=True)
    occupancy_targets = {rec.asset_id: build_coarse_occupancy(train_ds[i], coarse_grid=coarse_grid).to(device)
                         for i, rec in enumerate(train_ds.records)}

    print(f'Starting Occupancy DiT Flow training on {len(train_ds)} assets...', flush=True)
    model.train()
    conditioner.train()

    rng = np.random.default_rng(42)
    history = []
    start = time.perf_counter()

    for step in range(1, 601):
        idx = int(rng.integers(len(train_ds)))
        rec = train_ds.records[idx]
        target = occupancy_targets[rec.asset_id]
        tokens = conditioner.text_projection(raw_text_cache[rec.asset_id])

        # Random time and linear rectified flow target
        time_t = torch.rand((), device=device)
        noise = torch.randn_like(target)
        noisy, v_target = flow_pair(target, noise, time_t)

        # 10% prompt dropout for CFG
        drop = torch.rand((), device=device) < 0.1
        cond_input = None if drop else tokens

        optimizer.zero_grad(set_to_none=True)
        pred_v = model(noisy, grid_coords, time_t, cond_input)
        loss = torch.nn.functional.mse_loss(pred_v, v_target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(conditioner.text_projection.parameters()), 1.0)
        optimizer.step()

        if step % 50 == 0:
            rec_loss = float(loss.detach())
            history.append(dict(step=step, asset_id=rec.asset_id, loss=rec_loss))
            print(json.dumps(dict(step=step, asset=rec.asset_id, loss=round(rec_loss, 5))), flush=True)

    print('Training complete. Running held-out validation evaluation...', flush=True)
    model.eval()
    conditioner.eval()

    # Deterministic validation subset: 12 from pirate-kit, 12 from space-kit
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
            target = build_coarse_occupancy(asset, coarse_grid=coarse_grid).to(device)
            target_mask = target[:, 0] > 0.5

            prompt = id_to_prompt(rec.asset_id)
            tokens = conditioner.text_projection(raw_text_cache[rec.asset_id])

            # Also test an explicitly different prompt to check prompt-sensitivity
            mismatched_rec = train_ds.records[(vi * 17) % len(train_ds)]
            mismatched_prompt = id_to_prompt(mismatched_rec.asset_id)
            mismatched_tokens = conditioner.text_projection(raw_text_cache[mismatched_rec.asset_id])

            gen_noise = torch.randn((len(grid_coords), 1), device=device, generator=torch.Generator(device=device).manual_seed(9000 + vi))

            # Sample with correct prompt
            pred_correct = sample_flow(model, gen_noise, grid_coords, tokens, steps=15, guidance=2.0)
            correct_mask = pred_correct[:, 0] > 0.5

            # Sample with mismatched prompt
            pred_mismatched = sample_flow(model, gen_noise, grid_coords, mismatched_tokens, steps=15, guidance=2.0)
            mismatched_mask = pred_mismatched[:, 0] > 0.5

            def calc_metrics(pred_m, gt_m):
                inter = int((pred_m & gt_m).sum())
                union = int((pred_m | gt_m).sum())
                iou = inter / max(1, union)
                return dict(predicted=int(pred_m.sum()), target=int(gt_m.sum()), iou=round(iou, 4))

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
    mean_diff = sum(e['prompt_difference_cells'] for e in eval_results) / len(eval_results)
    print(json.dumps(dict(mean_val_occupancy_iou=round(mean_iou, 4), mean_prompt_diff_cells=round(mean_diff, 1))), flush=True)

    out_dir = Path('/kaggle/working')
    ckpt_path = out_dir / 'occupancy-dit-600.pt'
    torch.save(dict(
        version=1,
        model=model.state_dict(),
        text_projection=conditioner.text_projection.state_dict(),
        manifest_sha256=train_ds.fingerprint,
        grid_size=coarse_grid,
        models=conditioner.model_provenance
    ), ckpt_path)

    report = dict(
        scope='Text-conditioned Stage 1 coarse occupancy DiT (16³ grid) trained on 649 Kenney train assets; evaluated on 24 held-out assets with prompt sensitivity checks',
        device=torch.cuda.get_device_name(0),
        elapsed_seconds=time.perf_counter() - start,
        train_count=len(train_ds),
        validation_eval_count=len(eval_results),
        mean_val_occupancy_iou=mean_iou,
        mean_prompt_difference_cells=mean_diff,
        checkpoint_sha256=hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        history=history,
        eval_results=eval_results
    )
    (out_dir / 'occupancy-dit-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Report written to occupancy-dit-report.json', flush=True)


if __name__ == '__main__':
    main()
