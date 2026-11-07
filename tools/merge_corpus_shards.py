"""Merge converted corpus shard manifests into one deduplicated corpus set.

Reuses tools.merge_kenney_shards.exact_deduplicate for transitive duplicate
grouping, hardlinks kept files into the output corpus dir, and verifies every
staged file against its prepared sha.
"""
import argparse, hashlib, json, os, shutil
from pathlib import Path

from tools.merge_kenney_shards import exact_deduplicate


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shards', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    all_records = []
    all_stats = []
    all_rejects = []
    shards_meta = []
    for shard in args.shards:
        manifest = json.loads((shard / 'manifest.json').read_text())
        report = json.loads((shard / 'conversion-report.json').read_text())
        shards_meta.append(dict(dir=str(shard), assets=len(manifest['assets']), start=report.get('start'), stop=report.get('stop'), records_only=report.get('records_only')))
        for record in manifest['assets']:
            all_records.append(dict(record, shard=str(shard)))
        all_stats.extend(report.get('accepted', []))
        all_rejects.extend(report.get('rejected', []))
    records_in = len(all_records)
    selected, duplicates = exact_deduplicate(all_records)
    args.output.mkdir(parents=True, exist_ok=True)
    hardlinked = copied = 0
    for record in selected:
        shard = record.pop('shard')
        source = Path(shard) / record['path']
        if not source.exists():
            raise SystemExit(f"missing shard file for {record['asset_id']}: {source}")
        if sha256(source) != record['sha256']:
            raise SystemExit(f"sha mismatch for {record['asset_id']}: {source}")
        destination = args.output / record['path']
        if destination.exists():
            destination.unlink()
        try:
            os.link(source, destination)
            hardlinked += 1
        except OSError:
            shutil.copy2(source, destination)
            copied += 1
        if sha256(destination) != record['sha256']:
            raise SystemExit(f"hash mismatch after staging {record['asset_id']}")
    manifest = dict(version=1, assets=selected)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'conversion-report.json').write_text(json.dumps(dict(scope='merged corpus shards; exact source/prepared duplicates via tools.merge_kenney_shards.exact_deduplicate', shards=shards_meta, assets_in=records_in, accepted=all_stats, rejected=all_rejects, duplicates=duplicates), indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), shards=len(args.shards), assets_in=records_in, assets_out=len(selected), duplicates=len(duplicates), hardlinked=hardlinked, copied=copied), indent=2))


if __name__ == '__main__':
    main()
