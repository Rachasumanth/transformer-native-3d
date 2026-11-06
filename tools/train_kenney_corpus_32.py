"""Train shape VAE on the full pack-split 868-asset Kenney CC0 corpus."""
import hashlib, json, time
from pathlib import Path
import torch
from t3d.container import dumps
from t3d.dataset import AssetDataset
from t3d.reconstruction import reconstruct_asset
from t3d.training import VAETrainer
from tools.audit_triton_paired_reconstruction import metrics


def select_eval_subset(dataset, per_group=12, seed=2026):
    """Deterministically select a balanced subset of validation assets for periodic auditing."""
    import random
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
        raise RuntimeError('CUDA required')

    ckroot = Path('/kaggle/input/t3d-mixed-pilot-shape-checkpoint')
    report = json.loads((ckroot / 'triton-mixed-32.json').read_text())
    checkpoint = ckroot / 'triton-mixed-600.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected = next(x['checkpoint_sha256'] for x in report['rows'] if x['step'] == 600)
    if digest != expected:
        raise ValueError('Checkpoint hash mismatch')

    old = AssetDataset('artifacts/mixed-shape-pilot-32/manifest.json', split='train')

    # Locate mounted Kenney corpus dataset
    matches = list(Path('/kaggle/input').rglob('manifest.json'))
    corpus_matches = [p for p in matches if 't3d-kenney-corpus-32' in str(p) or (p.parent / 'kenney-car-kit-ambulance.t3d').exists()]
    if not corpus_matches:
        raise FileNotFoundError('Mounted t3d-kenney-corpus-32 dataset not found')
    manifest_path = corpus_matches[0]
    print(f'Using corpus manifest: {manifest_path}', flush=True)

    sets = {s: AssetDataset(manifest_path, split=s) for s in ('train', 'validation')}
    print(f"Corpus loaded: {len(sets['train'])} train, {len(sets['validation'])} validation", flush=True)

    eval_indices = {'validation': select_eval_subset(sets['validation'], per_group=12, seed=2026)}
    print(f"Selected {len(eval_indices['validation'])} balanced validation assets for periodic auditing", flush=True)

    trainer = VAETrainer.resume(checkpoint, device='cuda', manifest_sha256=old.fingerprint)
    if trainer.steps != 600 or trainer.config.get('convolution_backend') != 'triton' or trainer.occupancy_level_weights is not None:
        raise ValueError('Unexpected warm start configuration')

    trainer.manifest_sha256 = sets['train'].fingerprint
    trainer.batch_cursor = 0
    trainer.epoch = 0
    trainer.shuffle_seed = 2026
    trainer.batch_size = 1

    train_len = len(sets['train'])
    rows = []
    start = time.perf_counter()

    for target in (600, 900, 1200, 1500, 1900):
        while trainer.steps < target:
            idx = trainer.next_indices(train_len, 1)[0]
            loss = trainer.update([sets['train'][idx]])
            if trainer.steps % 50 == 0:
                print(json.dumps(dict(step=trainer.steps, epoch=trainer.epoch,
                                      asset=sets['train'].records[idx].asset_id,
                                      loss=round(loss['total'], 5))), flush=True)

        # Audit evaluation subset
        cases = []
        for i in eval_indices['validation']:
            asset = sets['validation'][i]
            rec = sets['validation'].records[i]
            teacher = trainer.evaluate([asset])
            decoded = reconstruct_asset(asset, trainer.model, support='predicted')
            free_occ = metrics(asset, decoded)
            case = dict(split='validation', asset=rec.asset_id, group=rec.source_group,
                        teacher_forced_loss=teacher, free_occupancy=free_occ)
            if target == 1900:
                data = dumps(decoded)
                (Path('/kaggle/working') / f'{rec.asset_id}-predicted.t3d').write_bytes(data)
                case['reconstruction_sha256'] = hashlib.sha256(data).hexdigest()
            cases.append(case)

        mean_val_iou = sum(c['free_occupancy']['iou'] for c in cases) / len(cases)
        mean_val_loss = sum(c['teacher_forced_loss']['total'] for c in cases) / len(cases)
        print(json.dumps(dict(step=trainer.steps, mean_val_iou=round(mean_val_iou, 4),
                              mean_val_loss=round(mean_val_loss, 4), eval_count=len(cases))), flush=True)

        row = dict(step=trainer.steps, mean_val_iou=mean_val_iou, mean_val_loss=mean_val_loss, cases=cases)
        if target > 600:
            ckpt_path = Path('/kaggle/working') / f'triton-kenney-corpus-{target}.pt'
            trainer.save(ckpt_path)
            row['checkpoint_sha256'] = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        rows.append(row)

    result = dict(
        scope='Warm-start from triton-mixed-600.pt, trained on 649 Kenney train assets across 4 packs; evaluated on 24 held-out assets from pirate and space packs',
        device=torch.cuda.get_device_name(0),
        train_count=len(sets['train']),
        validation_count=len(sets['validation']),
        old_manifest_sha256=old.fingerprint,
        corpus_manifest_sha256=sets['train'].fingerprint,
        initial_checkpoint_sha256=digest,
        elapsed_seconds=time.perf_counter() - start,
        rows=rows
    )
    out_file = Path('/kaggle/working/triton-kenney-corpus-32.json')
    out_file.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Training complete. Summary saved to {out_file}', flush=True)


if __name__ == '__main__':
    main()
