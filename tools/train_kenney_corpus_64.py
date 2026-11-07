"""Train shape VAE on the 64³ Kenney CC0 corpus with Triton sparse backend."""
import hashlib
import json
from pathlib import Path
import random
import time
import torch

from t3d.container import dumps
from t3d.dataset import AssetDataset
from t3d.reconstruction import reconstruct_asset
from t3d.training import VAETrainer
from tools.audit_triton_paired_reconstruction import metrics


def select_eval_subset(dataset, per_group=6, seed=2026):
    """Deterministically select a balanced subset of validation assets for periodic auditing."""
    rng = random.Random(seed)
    by_group = {}
    for i, rec in enumerate(dataset.records):
        by_group.setdefault(rec.source_group, []).append(i)
    selected = []
    for grp in sorted(by_group):
        indices = sorted(by_group[grp])
        rng.shuffle(indices)
        selected.extend(indices[:per_group])
    return sorted(selected)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for Triton SC-VAE training')

    # Locate mounted Kenney 64 corpus dataset
    matches = list(Path('/kaggle/input').rglob('manifest.json'))
    corpus_matches = [
        p for p in matches
        if 't3d-kenney-corpus-64' in str(p) or (p.parent / 'manifest.json').exists() and '64' in str(p)
    ]
    if not corpus_matches:
        # Fallback search
        corpus_matches = [p for p in matches if 'kenney' in str(p) and '64' in str(p)]
    if not corpus_matches:
        # If running locally or test mount
        candidates = [
            Path('artifacts/kenney-corpus-64/manifest.json'),
            *list(Path('/kaggle/input').rglob('manifest.json'))
        ]
        corpus_matches = [p for p in candidates if p.exists()]

    if not corpus_matches:
        raise FileNotFoundError('Mounted t3d-kenney-corpus-64 dataset not found')

    manifest_path = corpus_matches[0]
    print(f'Using 64³ corpus manifest: {manifest_path}', flush=True)

    sets = {s: AssetDataset(manifest_path, split=s) for s in ('train', 'validation')}
    print(f"64³ Corpus loaded: {len(sets['train'])} train, {len(sets['validation'])} validation", flush=True)

    eval_indices = {'validation': select_eval_subset(sets['validation'], per_group=6, seed=2026)}
    print(f"Selected {len(eval_indices['validation'])} balanced validation assets for periodic auditing", flush=True)

    # Check for warm-start checkpoint from 32³ corpus
    ckroot_candidates = [
        *list(Path('/kaggle/input').rglob('triton-kenney-corpus-1900.pt')),
        Path('artifacts/kaggle-kenney-corpus-training-v35/triton-kenney-corpus-1900.pt'),
        Path('artifacts/checkpoints/triton-kenney-corpus-1900.pt')
    ]
    warm_ckpt = next((p for p in ckroot_candidates if p.exists()), None)

    if warm_ckpt is not None:
        print(f"Found warm start checkpoint: {warm_ckpt}", flush=True)
        # Load weights into fresh trainer configured for 64³ dataset
        ckpt_data = torch.load(warm_ckpt, map_location='cuda')
        channels = tuple(ckpt_data['config']['channels'])
        latent_channels = int(ckpt_data['config']['latent_channels'])
        trainer = VAETrainer(
            mode='shape',
            channels=channels,
            latent_channels=latent_channels,
            convolution_backend='triton',
            manifest_sha256=sets['train'].fingerprint,
            device='cuda',
            learning_rate=1e-4
        )
        trainer.model.load_state_dict(ckpt_data['model'])
        trainer.steps = 0
        print(f"Warm-started model weights from {warm_ckpt} with channels={channels}", flush=True)
    else:
        print("Initializing fresh VAETrainer with Triton backend...", flush=True)
        trainer = VAETrainer(
            mode='shape',
            channels=(8, 16, 32, 64, 128),
            latent_channels=16,
            convolution_backend='triton',
            manifest_sha256=sets['train'].fingerprint,
            device='cuda',
            learning_rate=2e-4
        )

    trainer.batch_cursor = 0
    trainer.epoch = 0
    trainer.shuffle_seed = 2026
    trainer.batch_size = 1

    train_len = len(sets['train'])
    rows = []
    start = time.perf_counter()

    out_dir = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('artifacts/kaggle-kenney-corpus-64-training')
    out_dir.mkdir(parents=True, exist_ok=True)

    target_steps = [300, 600, 900, 1200]
    for target in target_steps:
        while trainer.steps < target:
            idx = trainer.next_indices(train_len, 1)[0]
            loss = trainer.update([sets['train'][idx]])
            if trainer.steps % 50 == 0:
                print(json.dumps(dict(
                    step=trainer.steps,
                    epoch=trainer.epoch,
                    asset=sets['train'].records[idx].asset_id,
                    loss=round(loss['total'], 5)
                )), flush=True)

        # Audit evaluation subset
        cases = []
        for i in eval_indices['validation']:
            asset = sets['validation'][i]
            rec = sets['validation'].records[i]
            teacher = trainer.evaluate([asset])
            decoded = reconstruct_asset(asset, trainer.model, support='predicted')
            free_occ = metrics(asset, decoded)
            case = dict(
                split='validation',
                asset=rec.asset_id,
                group=rec.source_group,
                teacher_forced_loss=teacher,
                free_occupancy=free_occ
            )
            if target == target_steps[-1]:
                data = dumps(decoded)
                (out_dir / f'{rec.asset_id}-predicted-64.t3d').write_bytes(data)
                case['reconstruction_sha256'] = hashlib.sha256(data).hexdigest()
            cases.append(case)

        mean_val_iou = sum(c['free_occupancy']['iou'] for c in cases) / len(cases)
        mean_val_loss = sum(c['teacher_forced_loss']['total'] for c in cases) / len(cases)
        print(json.dumps(dict(
            step=trainer.steps,
            mean_val_iou=round(mean_val_iou, 4),
            mean_val_loss=round(mean_val_loss, 4),
            eval_count=len(cases)
        )), flush=True)

        row = dict(
            step=trainer.steps,
            mean_val_iou=mean_val_iou,
            mean_val_loss=mean_val_loss,
            cases=cases
        )

        ckpt_file = out_dir / f'triton-kenney-corpus-64-{trainer.steps}.pt'
        torch.save(dict(
            step=trainer.steps,
            model=trainer.model.state_dict(),
            config=dict(
                mode='shape',
                channels=tuple(trainer.channels) if hasattr(trainer, 'channels') else (8, 16, 32, 64, 128),
                latent_channels=getattr(trainer, 'latent_channels', 16),
                convolution_backend='triton'
            ),
            manifest_sha256=sets['train'].fingerprint,
            mean_val_iou=mean_val_iou,
            mean_val_loss=mean_val_loss
        ), ckpt_file)
        row['checkpoint_sha256'] = hashlib.sha256(ckpt_file.read_bytes()).hexdigest()
        rows.append(row)

    report_path = out_dir / 'triton-kenney-corpus-64.json'
    report_path.write_text(json.dumps(dict(
        title='T3D 64³ Kenney Corpus SC-VAE Training Report',
        total_time_seconds=round(time.perf_counter() - start, 2),
        train_manifest_sha256=sets['train'].fingerprint,
        validation_manifest_sha256=sets['validation'].fingerprint,
        rows=rows
    ), indent=2) + '\n')
    print(f"Training complete. Report written to {report_path}", flush=True)


if __name__ == '__main__':
    main()
