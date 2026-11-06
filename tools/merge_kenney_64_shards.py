"""Verify and merge five 64³ corpus shards, removing exact cross-shard duplicates."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile
from t3d.dataset import AssetDataset


def exact_deduplicate(records):
    records = sorted(records, key=lambda r: r['asset_id'])
    parents = list(range(len(records)))
    seen = {}

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, r in enumerate(records):
        for key in (('source', r['source_sha256']), ('prepared', r['sha256'])):
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i

    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)

    keep = set()
    duplicates = []
    for ids in groups.values():
        splits = {records[i]['split'] for i in ids}
        selected = ids[:1] if len(splits) == 1 else []
        keep.update(selected)
        for i in ids:
            if i not in selected:
                duplicates.append(dict(
                    asset_id=records[i]['asset_id'],
                    reason='cross-split exact duplicate' if len(splits) > 1 else 'within-split exact duplicate'
                ))

    return [r for i, r in enumerate(records) if i in keep], duplicates


def main():
    parser = argparse.ArgumentParser(description='Merge 64³ corpus shards')
    parser.add_argument('--shards-dir', type=Path, default=Path('artifacts'), help='Base directory containing downloaded shard folders')
    parser.add_argument('--output', type=Path, default=Path('artifacts/kenney-corpus-64'), help='Output merged dataset directory')
    parser.add_argument('--expected-source', type=str, default=None, help='Expected code bundle SHA-256')
    args = parser.parse_args()

    index_path = Path('artifacts/kenney-raw-pilot/candidate-index.json')
    index_raw = index_path.read_bytes()
    index = json.loads(index_raw)
    candidates = {r['id']: r for r in index['candidates']}

    records = []
    shard_failures = []
    data = {}
    seen_ids = set()

    for shard in range(5):
        root = args.shards_dir / f'kaggle-kenney-corpus-64-shard{shard}'
        report_file = root / f'kenney-64-shard-{shard}-report.json'
        archive_file = root / f'kenney-64-shard-{shard}.zip'

        if not report_file.exists() or not archive_file.exists():
            raise FileNotFoundError(f'Missing shard files for shard {shard} in {root}')

        report = json.loads(report_file.read_text())
        if args.expected_source and report['source_sha256'] != args.expected_source:
            raise ValueError(f'Shard {shard} code bundle mismatch: {report["source_sha256"]} != {args.expected_source}')
        if report['index_sha256'] != hashlib.sha256(index_raw).hexdigest():
            raise ValueError(f'Shard {shard} index hash mismatch')
        if (report['start'], report['stop'], report['stride']) != (shard, 944, 5):
            raise ValueError(f'Shard {shard} range parameters mismatch')

        shard_failures.extend(dict(shard=shard, **r) for r in report['rejected'])

        with zipfile.ZipFile(archive_file) as z:
            manifest = json.loads(z.read('manifest.json'))
            if manifest['version'] != 1 or len(manifest['assets']) != len(report['accepted']):
                raise ValueError(f'Shard {shard} manifest/report asset count mismatch')

            for rec in manifest['assets']:
                name = rec['path']
                asset_id = rec['asset_id']
                if not re.fullmatch(r'kenney-[a-z0-9-]+\.t3d', name) or name != asset_id + '.t3d' or asset_id in seen_ids:
                    raise ValueError(f'Invalid or duplicate asset path {name}')

                seen_ids.add(asset_id)
                source_id = asset_id[len('kenney-'):]
                matches = [r for r in candidates.values() if r['id'].replace('/', '-') == source_id]
                if len(matches) != 1:
                    raise ValueError(f'Unrecognized source ID: {source_id}')
                origin = matches[0]

                if rec['source'] != origin['source_url'] or rec['split'] != origin['split'] or rec['source_group'] != 'kenney-' + origin['group']:
                    raise ValueError(f'Source or split metadata mismatch for {asset_id}')

                metadata = Path('artifacts/kenney-catalog-download') / (origin['id'] + '.json')
                meta = json.loads(metadata.read_text())
                if rec['source_sha256'] != meta['sha256']:
                    raise ValueError(f'Raw source SHA mismatch for {asset_id}')

                blob = z.read(name)
                if hashlib.sha256(blob).hexdigest() != rec['sha256']:
                    raise ValueError(f'Prepared asset SHA mismatch for {name}')

                records.append(rec)
                data[name] = blob

    selected, duplicates = exact_deduplicate(records)
    args.output.mkdir(parents=True, exist_ok=True)

    for r in selected:
        path = args.output / r['path']
        blob = data[r['path']]
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != r['sha256']:
            raise ValueError(f'Existing file differs: {path}')
        path.write_bytes(blob)

    manifest_out = dict(version=1, assets=selected)
    (args.output / 'manifest.json').write_text(json.dumps(manifest_out, indent=2) + '\n')

    audit = dict(
        scope='64³ grid resolution Kenney corpus; exact source/prepared deduplication applied',
        index_sha256=hashlib.sha256(index_raw).hexdigest(),
        candidates=len(index['candidates']),
        accepted=len(selected),
        shard_rejections=shard_failures,
        cross_shard_duplicates=duplicates
    )
    (args.output / 'merge-report.json').write_text(json.dumps(audit, indent=2) + '\n')

    for split in ('train', 'validation'):
        ds = AssetDataset(args.output / 'manifest.json', split=split)
        print(f'{split}: {len(ds)} assets', flush=True)

    print(f'Done. Accepted {len(selected)}, Rejections {len(shard_failures)}, Duplicates {len(duplicates)}')
    print(f'Manifest SHA-256: {hashlib.sha256((args.output / "manifest.json").read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
