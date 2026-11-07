"""Build an in-distribution holdout manifest from the 64³ corpus.

The production benchmark assets are always validation. Extra validation assets
are sampled round-robin across source groups so every kit is represented while
training keeps same-kit neighbours (explicit ``in_distribution_holdout`` policy).
"""
import argparse, hashlib, json, os, shutil
from pathlib import Path

BENCHMARK = (
    'kenney-pirate-kit-boat-row-large',
    'kenney-pirate-kit-bottle',
    'kenney-pirate-kit-ship-medium',
    'kenney-pirate-kit-rocks-a',
    'kenney-space-kit-craft-speedera',
    'kenney-space-kit-rocket-basea',
    'kenney-space-kit-monorail-trainfront',
    'kenney-space-kit-barrel',
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='artifacts/kenney-corpus-64/manifest.json')
    parser.add_argument('--output', default='artifacts/kenney-corpus-64-id')
    parser.add_argument('--extra-validation', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    payload = json.loads(manifest.read_text())
    records = payload['assets']
    by_id = {record['asset_id']: record for record in records}
    missing = [asset_id for asset_id in BENCHMARK if asset_id not in by_id]
    if missing:
        raise SystemExit(f'benchmark assets missing from manifest: {missing}')

    extra = max(0, args.extra_validation)
    candidates = [record for record in records if record['asset_id'] not in BENCHMARK]
    groups = {}
    for record in candidates:
        groups.setdefault(record.get('source_group') or record['asset_id'], []).append(record)
    import random
    rng = random.Random(args.seed)
    for group in groups.values():
        group.sort(key=lambda record: record['asset_id'])
        rng.shuffle(group)
    selected = []
    while len(selected) < extra and any(groups.values()):
        for name in sorted(groups):
            if groups[name] and len(selected) < extra:
                selected.append(groups[name].pop())
    validation_ids = set(BENCHMARK) | {record['asset_id'] for record in selected}
    if len(validation_ids) != 8 + len(selected):
        raise SystemExit('benchmark and sampled validation ids overlap')

    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists() and not args.force:
        raise SystemExit(f'{output / "manifest.json"} exists; pass --force to overwrite')
    copied = hardlinked = 0
    for record in records:
        source = manifest.parent / record['path']
        destination = output / record['path']
        if destination.exists():
            destination.unlink()
        try:
            os.link(source, destination)
            hardlinked += 1
        except OSError:
            shutil.copy2(source, destination)
            copied += 1
        if sha256(destination) != record['sha256']:
            raise SystemExit(f'hash mismatch after staging {record["asset_id"]}')

    updated = []
    for record in records:
        updated.append(dict(record, split='validation' if record['asset_id'] in validation_ids else 'train'))
    id_split = dict(
        policy='in_distribution_holdout',
        benchmark=sorted(BENCHMARK),
        extra_validation=sorted(record['asset_id'] for record in selected),
        seed=args.seed,
        train=len(updated) - len(validation_ids),
        validation=len(validation_ids),
        note='Held-out assets share 64³ corpus and source kits with training; not an OOD estimate.',
    )
    result = dict(version=1, split_policy='in_distribution_holdout', id_split=id_split, assets=updated)
    temp = output / 'manifest.json.tmp'
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    temp.replace(output / 'manifest.json')
    print(json.dumps(dict(output=str(output), train=id_split['train'], validation=id_split['validation'],
                          hardlinked=hardlinked, copied=copied,
                          benchmark=list(BENCHMARK), extra=len(selected)), indent=2))


if __name__ == '__main__':
    main()
