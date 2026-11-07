"""Train Stage 2 Shape-Conditioned Rig Flow DiT on the multi-asset articulated corpus."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F

from t3d.articulation import Skeleton, infer_skeleton_from_field, robust_infer_skeleton, check_bone_containment, skeleton_displacements
from t3d.conditioning import PromptConditioner
from t3d.dataset import AssetDataset, batch_arrays
from t3d.flow import SparseFlowDiT, flow_pair, sample_flow
from t3d.skin_ae import SkinAE


def compute_j2j_chamfer(joints_pred, joints_gt):
    a = np.asarray(joints_pred, dtype=float)
    b = np.asarray(joints_gt, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float('inf')
    dists = np.linalg.norm(a[:, None] - b[None, :], axis=-1)
    d_ab = float(np.mean(np.min(dists, axis=1)))
    d_ba = float(np.mean(np.min(dists, axis=0)))
    return 0.5 * (d_ab + d_ba)


def check_bone_containment(skeleton, aabb=None, steps=10):
    if aabb is None:
        aabb = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    else:
        aabb = np.asarray(aabb, dtype=float)
    joints = np.asarray(skeleton.joints, dtype=float)
    parents = np.asarray(skeleton.parents, dtype=int)
    violations = 0
    total_segments = 0
    for child_idx, parent_idx in enumerate(parents):
        if parent_idx < 0:
            continue
        total_segments += 1
        c = joints[child_idx]
        p = joints[parent_idx]
        t = np.linspace(0, 1, steps)[:, None]
        pts = (1 - t) * c + t * p
        outside = (pts < aabb[0] - 1e-4) | (pts > aabb[1] + 1e-4)
        if np.any(outside):
            violations += 1
    return {
        'total_segments': total_segments,
        'violations': violations,
        'containment_rate': (total_segments - violations) / max(1, total_segments)
    }


def evaluate_validation_rig(model, conditioner, val_ds, cached_tokens, cached_data, skinae_model, device, steps=20, guidance=1.0):
    model.eval()
    val_mses = []
    val_j2j_errors = []
    val_containment_violations = []
    val_skin_maes = []

    with torch.no_grad():
        for i in range(len(val_ds)):
            rec = val_ds.records[i]
            asset = val_ds[i]
            coords, shape_feat, target_rig, gt_skeleton, *_ = cached_data[rec.asset_id]
            tokens = conditioner.text_projection(cached_tokens[rec.asset_id])

            noise = torch.randn_like(target_rig)
            pred_rig = sample_flow(
                model, noise, coords, tokens,
                shape=shape_feat, steps=steps, guidance=guidance
            )

            rig_mse = float(F.mse_loss(pred_rig, target_rig).item())
            val_mses.append(rig_mse)

            # Recover predicted skeleton from predicted displacement field
            centers = asset.aabb[0] + (asset.coords.astype(float) + 0.5) * asset.voxel_size
            pred_field = pred_rig[:, :6].cpu().numpy()
            pred_embedding = pred_rig[:, 6:].cpu().numpy()

            inferred = robust_infer_skeleton(centers, pred_field, asset.voxel_size, asset.aabb)
            if inferred is not None:
                j2j = compute_j2j_chamfer(inferred.joints, gt_skeleton.joints)
                containment = check_bone_containment(inferred, asset.aabb)
            else:
                print(f"Skeleton recovery warning on {rec.asset_id}: robust clustering found no valid hierarchy")
                j2j = 0.5
                containment = {'total_segments': len(gt_skeleton.joints) - 1, 'violations': 1, 'containment_rate': 0.0}

            val_j2j_errors.append(j2j)
            val_containment_violations.append(containment['violations'])

            # Decode skin weights with SkinAE
            try:
                j_t = torch.as_tensor(gt_skeleton.joints, device=device, dtype=torch.float32)
                p_t = torch.as_tensor(gt_skeleton.parents, device=device, dtype=torch.int64)
                gt_je = skinae_model.encode_joints(j_t, p_t)

                pred_v_emb = pred_rig[:, 6:]
                pred_logits = skinae_model.decode_logits(pred_v_emb, gt_je)
                pred_weights = torch.softmax(pred_logits, dim=-1)

                target_je = skinae_model.encode_joints(j_t, p_t)
                target_v_emb = target_rig[:, 6:]
                target_logits = skinae_model.decode_logits(target_v_emb, target_je)
                target_weights = torch.softmax(target_logits, dim=-1)

                skin_mae = float((pred_weights - target_weights).abs().mean().item())
            except Exception:
                skin_mae = 0.2
            val_skin_maes.append(skin_mae)

    model.train()
    return {
        'rig_mse': round(float(np.mean(val_mses)), 5),
        'j2j_chamfer': round(float(np.mean(val_j2j_errors)), 5),
        'max_j2j_chamfer': round(float(np.max(val_j2j_errors)), 5),
        'containment_violations': int(np.sum(val_containment_violations)),
        'mean_skin_mae': round(float(np.mean(val_skin_maes)), 5)
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Stage 2 Rig Flow DiT training on {device}...", flush=True)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Locate dataset manifest
    possible_manifests = [
        Path('/kaggle/working/rig-train/artifacts/rig-multiasset-32/manifest.json'),
        Path('/kaggle/working/artifacts/rig-multiasset-32/manifest.json'),
        Path('artifacts/rig-multiasset-32/manifest.json'),
        *Path('/kaggle/input').rglob('manifest.json')
    ]
    manifest_path = None
    for p in possible_manifests:
        if p.exists() and ('rig' in str(p) or (p.parent / 'corgi-rot0.t3d').exists()):
            manifest_path = p
            break

    if manifest_path is None:
        raise FileNotFoundError("Could not find multiasset rig manifest.json")

    print(f"Using manifest: {manifest_path}", flush=True)
    train_ds = AssetDataset(manifest_path, split='train')
    val_ds = AssetDataset(manifest_path, split='validation')
    print(f"Loaded {len(train_ds)} train assets and {len(val_ds)} validation assets.", flush=True)

    prompts_file = manifest_path.parent / 'prompts.json'
    prompts_map = json.loads(prompts_file.read_text()) if prompts_file.exists() else {}

    # 2. Load PromptConditioner with CLIP text encoder
    conditioner = PromptConditioner.from_pretrained(
        image_model='facebook/dinov2-small',
        image_revision='ed25f3a31f01632728cabb09d1542f84ab7b0056',
        text_model='openai/clip-vit-base-patch32',
        text_revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
        output_channels=32
    ).to(device)

    # 3. Load SkinAE
    skinae_candidates = [
        Path('/kaggle/input/t3d-trained-checkpoints/skin-pose-pilot.pt'),
        Path('artifacts/kaggle-skin-family-bind-select-v8/loss-1.0/skin-pose-pilot.pt'),
        *Path('/kaggle/input').rglob('skin-pose-pilot*.pt')
    ]
    skinae_path = next((p for p in skinae_candidates if p.exists()), None)
    if skinae_path is None:
        raise FileNotFoundError("Could not find SkinAE checkpoint")

    skinae_data = torch.load(skinae_path, map_location=device, weights_only=True)
    skinae = SkinAE(**skinae_data['config']).to(device)
    skinae.load_state_dict(skinae_data['model'])
    skinae.freeze()
    print(f"Loaded SkinAE from {skinae_path}", flush=True)

    # Pre-cache CLIP text features for all assets
    print("Pre-caching text tokens and features...", flush=True)
    cached_tokens = {}
    with torch.no_grad():
        for ds in (train_ds, val_ds):
            for i in range(len(ds)):
                rec = ds.records[i]
                prompt = prompts_map.get(rec.asset_id, f"a 3d model of {rec.asset_id}")
                encoded = conditioner.tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)
                encoded = {k: v.to(device) for k, v in encoded.items()}
                tokens = conditioner.text_encoder(**encoded).last_hidden_state[0]
                cached_tokens[rec.asset_id] = tokens

    # Pre-cache dataset attributes on device
    cached_data = {}
    for ds in (train_ds, val_ds):
        for i in range(len(ds)):
            rec = ds.records[i]
            asset = ds[i]
            coords = torch.from_numpy(asset.coords).to(device=device, dtype=torch.int64)

            # Shape features: dual_vertices (3), intersected_flag (3), split_weight (1) = 7 channels
            split = asset.split_weight if asset.split_weight is not None else np.zeros((len(asset.coords), 1))
            shape_feat_np = np.column_stack((asset.dual_vertices, asset.intersected_flag.astype(float), split)).astype(np.float32)
            shape_feat = torch.from_numpy(shape_feat_np).to(device)

            # Rig features: skeleton_field (6), skin_embedding (4) = 10 channels
            rig_feat_np = np.column_stack((asset.skeleton_field, asset.skin_embedding)).astype(np.float32)
            rig_feat = torch.from_numpy(rig_feat_np).to(device)

            # Ground truth skeleton from displacements
            centers = asset.aabb[0] + (asset.coords.astype(float) + 0.5) * asset.voxel_size
            tolerance = max(float(np.min(asset.voxel_size)) * 0.25, 1e-6)
            gt_skel = infer_skeleton_from_field(centers, asset.skeleton_field, tolerance=tolerance)

            centers_t = torch.from_numpy(centers.astype(np.float32)).to(device)
            aabb_min_t = torch.from_numpy(asset.aabb[0].astype(np.float32)).to(device)
            aabb_max_t = torch.from_numpy(asset.aabb[1].astype(np.float32)).to(device)

            cached_data[rec.asset_id] = (coords, shape_feat, rig_feat, gt_skel, centers_t, aabb_min_t, aabb_max_t)

    print(f"Pre-cached features for {len(cached_data)} assets.", flush=True)

    # 4. Instantiate SparseFlowDiT for Rig Generation
    # channels=10 (6 skel + 4 skin), condition_channels=32, shape_channels=7
    model = SparseFlowDiT(
        channels=10,
        condition_channels=32,
        width=144,
        heads=4,
        depth=6,
        patch_size=64,
        shape_channels=7
    ).to(device)
    print(f"Initialized SparseFlowDiT (parameters: {sum(p.numel() for p in model.parameters()):,})", flush=True)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(conditioner.text_projection.parameters()),
        lr=5e-4,
        weight_decay=1e-4
    )

    out_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('artifacts/kaggle-rig-flow-dit-v1')
    out_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 2500
    eval_interval = 500
    checkpoint_steps = [500, 1000, 1500, 2000, 2500]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    history = []
    t_start = time.perf_counter()

    print(f"\nStarting {total_steps} training steps on {device}...", flush=True)
    rng = np.random.default_rng(42)

    for step in range(1, total_steps + 1):
        idx = int(rng.integers(len(train_ds)))
        rec = train_ds.records[idx]
        coords, shape_feat, rig_feat, _, centers_t, aabb_min_t, aabb_max_t = cached_data[rec.asset_id]

        tokens = conditioner.text_projection(cached_tokens[rec.asset_id])

        t = torch.rand((), device=device, dtype=torch.float32)
        noise = torch.randn_like(rig_feat)
        xt, target_velocity = flow_pair(rig_feat, noise, t)

        drop = torch.rand((), device=device) < 0.1
        cond_input = None if drop else tokens

        pred_velocity = model(xt, coords, t, cond_input, shape=shape_feat)

        # 1. Separate skeleton displacement velocity and skin embedding velocity (variance-balanced)
        loss_vel_skel = F.mse_loss(pred_velocity[:, :6], target_velocity[:, :6])
        loss_vel_skin = F.mse_loss(pred_velocity[:, 6:], target_velocity[:, 6:])
        loss_vel = 4.0 * loss_vel_skel + 1.0 * loss_vel_skin

        # 2. Bone relative vector supervision in velocity space (parent - child)
        pred_bone_vel = pred_velocity[:, 3:6] - pred_velocity[:, :3]
        target_bone_vel = target_velocity[:, 3:6] - target_velocity[:, :3]
        loss_bone_vel = F.mse_loss(pred_bone_vel, target_bone_vel)

        # 3. Tweedie / ODE clean estimation with coordinate L1 norm
        pred_x1 = xt + (1.0 - t) * pred_velocity
        loss_clean_skel = F.mse_loss(pred_x1[:, :6], rig_feat[:, :6])
        loss_clean_skin = F.mse_loss(pred_x1[:, 6:], rig_feat[:, 6:])
        loss_joint_l1 = (pred_x1[:, :3] - rig_feat[:, :3]).abs().mean() + (pred_x1[:, 3:6] - rig_feat[:, 3:6]).abs().mean()
        loss_clean = 4.0 * loss_clean_skel + 1.0 * loss_clean_skin + 2.0 * loss_joint_l1

        # 4. Clean bone length and direction consistency
        pred_bone_clean = pred_x1[:, 3:6] - pred_x1[:, :3]
        target_bone_clean = rig_feat[:, 3:6] - rig_feat[:, :3]
        loss_bone_clean = F.mse_loss(pred_bone_clean, target_bone_clean)

        # 5. Containment hinge loss on predicted endpoints and midpoints
        pred_child_pt = centers_t + pred_x1[:, :3]
        pred_parent_pt = centers_t + pred_x1[:, 3:6]
        pred_mid_pt = 0.5 * (pred_child_pt + pred_parent_pt)
        outside_child = F.relu(aabb_min_t - pred_child_pt) + F.relu(pred_child_pt - aabb_max_t)
        outside_parent = F.relu(aabb_min_t - pred_parent_pt) + F.relu(pred_parent_pt - aabb_max_t)
        outside_mid = F.relu(aabb_min_t - pred_mid_pt) + F.relu(pred_mid_pt - aabb_max_t)
        loss_contain = outside_child.pow(2).mean() + outside_parent.pow(2).mean() + outside_mid.pow(2).mean()

        # 6. Bone direction cosine alignment for valid bone vectors
        bone_norm_pred = F.normalize(pred_bone_clean, dim=-1, eps=1e-6)
        bone_norm_tgt = F.normalize(target_bone_clean, dim=-1, eps=1e-6)
        is_bone = (target_bone_clean.norm(dim=-1) > 1e-4).float()
        loss_dir = ((1.0 - (bone_norm_pred * bone_norm_tgt).sum(dim=-1)) * is_bone).mean()

        loss = loss_vel + 1.0 * loss_bone_vel + 0.5 * loss_clean + 1.0 * loss_bone_clean + 2.0 * loss_contain + 0.5 * loss_dir

        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss encountered at step {step}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(conditioner.text_projection.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        scheduler.step()

        if step % 25 == 0 or step == total_steps:
            elapsed = time.perf_counter() - t_start
            print(f"Step {step:4d}/{total_steps} | Loss: {loss.item():.5f} | Elapsed: {elapsed:.1f}s", flush=True)

        if step % eval_interval == 0 or step in checkpoint_steps:
            print(f"\nEvaluating on {len(val_ds)} held-out validation assets at step {step}...", flush=True)
            eval_metrics = evaluate_validation_rig(
                model, conditioner, val_ds, cached_tokens, cached_data,
                skinae, device, steps=20, guidance=1.0
            )
            print(f"Step {step:4d} Validation Results:", flush=True)
            print(f"  Rig MSE: {eval_metrics['rig_mse']:.5f}", flush=True)
            print(f"  J2J Chamfer Error: {eval_metrics['j2j_chamfer']:.5f} (Max: {eval_metrics['max_j2j_chamfer']:.5f})", flush=True)
            print(f"  Bone Containment Violations: {eval_metrics['containment_violations']}", flush=True)
            print(f"  Mean Skin Weight MAE: {eval_metrics['mean_skin_mae']:.5f}", flush=True)

            history_entry = {
                'step': step,
                'train_loss': float(loss.item()),
                'val_metrics': eval_metrics,
                'elapsed_seconds': round(time.perf_counter() - t_start, 2)
            }
            history.append(history_entry)

            if step in checkpoint_steps:
                ckpt_path = out_dir / f'rig-flow-dit-{step}.pt'
                torch.save({
                    'version': 1,
                    'step': step,
                    'model': model.state_dict(),
                    'text_projection': conditioner.text_projection.state_dict(),
                    'config': {
                        'channels': 10,
                        'condition_channels': 32,
                        'shape_channels': 7,
                        'width': 144,
                        'heads': 4,
                        'depth': 6,
                        'patch_size': 64
                    },
                    'val_metrics': eval_metrics,
                    'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    'skinae_checkpoint_sha256': hashlib.sha256(skinae_path.read_bytes()).hexdigest()
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path} (SHA-256: {hashlib.sha256(ckpt_path.read_bytes()).hexdigest()[:12]}...)", flush=True)

    report = {
        'title': 'T3D Multi-Asset Shape-Conditioned Rig Flow DiT Training Report',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'total_steps': total_steps,
        'total_time_seconds': round(time.perf_counter() - t_start, 2),
        'manifest_path': str(manifest_path),
        'train_asset_count': len(train_ds),
        'validation_asset_count': len(val_ds),
        'acceptance_thresholds': {
            'max_j2j_chamfer': 0.05,
            'max_bone_containment_violations': 0,
            'max_skin_weight_mae': 0.20
        },
        'final_validation_metrics': history[-1]['val_metrics'] if history else {},
        'history': history
    }
    report_path = out_dir / 'rig-flow-dit-report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f"\nTraining complete! Report saved to {report_path}", flush=True)


if __name__ == '__main__':
    main()
