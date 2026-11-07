"""Train Material SC-VAE on the 26-asset multi-asset PBR corpus."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from t3d.dataset import AssetDataset
from t3d.training import VAETrainer


def evaluate_dataset(trainer, dataset):
    """Evaluate teacher-forced loss and per-channel MAE across all assets in dataset."""
    trainer.model.eval()
    losses = []
    channel_maes = []

    for i in range(len(dataset)):
        asset = dataset[i]
        coords, features = trainer.batch([asset])
        with torch.inference_mode():
            loss = trainer.evaluate([asset])
            losses.append(loss['total'])

            posterior = trainer.model.encode(coords, features)
            out, raw, _ = trainer.model.decode(posterior.coords, posterior.mean, teacher_occupancy=posterior.occupancy)
            sorted_coords, inverse = torch.unique(coords, dim=0, sorted=True, return_inverse=True)
            if not torch.equal(out, sorted_coords):
                raise ValueError(f"Teacher support mismatch for {asset.asset_id if hasattr(asset, 'asset_id') else i}")
            target = features[torch.argsort(inverse)]
            error = (raw.sigmoid() - target).abs().mean(dim=0)
            channel_maes.append(error.cpu().numpy())

    mean_loss = float(np.mean(losses))
    mean_mae = np.mean(channel_maes, axis=0)
    mae_dict = dict(zip(('r', 'g', 'b', 'metallic', 'roughness', 'alpha'), [float(x) for x in mean_mae]))
    mae_dict['mean'] = float(np.mean(mean_mae))
    return dict(loss=round(mean_loss, 5), mae=mae_dict)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backend = 'triton' if torch.cuda.is_available() else 'reference'

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Locate manifest
    possible_paths = [
        Path('/kaggle/working/material-train/artifacts/material-multiasset-32/manifest.json'),
        Path('/kaggle/working/artifacts/material-multiasset-32/manifest.json'),
        Path('artifacts/material-multiasset-32/manifest.json'),
        *Path('/kaggle/input').rglob('manifest.json')
    ]
    manifest_path = None
    for p in possible_paths:
        if p.exists() and ('material' in str(p) or (p.parent / 'avocado-rot0.t3d').exists()):
            manifest_path = p
            break

    if manifest_path is None:
        raise FileNotFoundError('Multi-asset material manifest not found')

    print(f'Using material manifest: {manifest_path}', flush=True)

    train_ds = AssetDataset(manifest_path, split='train', require_pbr=True)
    val_ds = AssetDataset(manifest_path, split='validation', require_pbr=True)
    print(f'Loaded {len(train_ds)} train assets, {len(val_ds)} validation assets', flush=True)

    trainer = VAETrainer(
        mode='material',
        channels=(8, 16, 32, 64, 128),
        latent_channels=16,
        device=device,
        learning_rate=1e-3,
        manifest_sha256=train_ds.fingerprint,
        convolution_backend=backend
    )

    out_dir = Path('/kaggle/working' if Path('/kaggle/working').exists() else 'artifacts/kaggle-material-multiasset-32')
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_steps = (0, 150, 300, 600, 900, 1200)
    checkpoint_steps = (300, 600, 900, 1200)
    records = []
    rng = np.random.default_rng(42)
    start_time = time.perf_counter()

    for target_step in eval_steps:
        while trainer.steps < target_step:
            idx = int(rng.integers(len(train_ds)))
            train_asset = train_ds[idx]
            loss_dict = trainer.update([train_asset])
            if trainer.steps % 50 == 0:
                print(f"Step {trainer.steps}: train_loss={loss_dict['total']:.4f} (attr={loss_dict['attribute']:.4f}, occ={loss_dict['occupancy']:.4f})", flush=True)

        print(f"Evaluating at step {trainer.steps}...", flush=True)
        train_eval = evaluate_dataset(trainer, train_ds)
        val_eval = evaluate_dataset(trainer, val_ds)

        rec = dict(
            step=trainer.steps,
            train=train_eval,
            validation=val_eval
        )

        if target_step in checkpoint_steps:
            ckpt_path = out_dir / f'material-multiasset-{target_step}.pt'
            trainer.save(ckpt_path)
            rec['checkpoint'] = ckpt_path.name
            rec['checkpoint_sha256'] = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
            print(f"Saved checkpoint: {ckpt_path.name} (SHA256: {rec['checkpoint_sha256'][:16]}...)", flush=True)

        records.append(rec)
        print(f"Step {trainer.steps} Summary: Val Loss={val_eval['loss']:.4f}, Val Mean MAE={val_eval['mae']['mean']:.4f} (RGB=[{val_eval['mae']['r']:.3f},{val_eval['mae']['g']:.3f},{val_eval['mae']['b']:.3f}], Metal={val_eval['mae']['metallic']:.3f}, Rough={val_eval['mae']['roughness']:.3f}, Alpha={val_eval['mae']['alpha']:.3f})", flush=True)

    total_time = time.perf_counter() - start_time
    report = dict(
        title="T3D Multi-Asset Material SC-VAE Training",
        manifest_sha256=train_ds.fingerprint,
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        backend=backend,
        train_assets=len(train_ds),
        val_assets=len(val_ds),
        total_steps=trainer.steps,
        elapsed_seconds=round(total_time, 2),
        history=records
    )

    report_path = out_dir / 'material-multiasset-report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Training completed in {total_time:.2f}s. Report saved to {report_path}", flush=True)


if __name__ == '__main__':
    main()
