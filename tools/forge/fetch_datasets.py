"""Fetch open-license 3D corpora for FORGE-3D raw->polished training.

Sources:
  polyhaven  521 CC0 PBR models (GLTF 1k with external textures)
  objaverse  filtered Objaverse LVIS subset (Sketchfab CC licenses; per-object attribution)
  thingi     3000 Thingi10K geometry-only NPZ meshes (CC licenses per asset)
  quaternius Quaternius CC0 rigged character packs (Google Drive folder, optional)

All downloads land under --out; a catalog.json / selection.json records provenance.
Runs on the local VM (internet available); Kaggle kernels consume packaged corpora only.
"""
import argparse
import concurrent.futures as futures
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

UA = {'User-Agent': 'forge3d-corpus/1.0 (research)'}
POLYHAVEN_ASSETS = 'https://api.polyhaven.com/assets?t=models'
POLYHAVEN_FILES = 'https://api.polyhaven.com/files/{asset}'
OBJAVERSE_SELECTION_SEED = 42
OBJAVERSE_CATEGORIES = 150
OBJAVERSE_PER_CATEGORY = 22
OBJAVERSE_LIMIT = 3300
THINGI_LIMIT = 3000


def get_json(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def download(url, path, retries=3, timeout=90):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        return path.stat().st_size
    tmp = path.with_suffix(path.suffix + '.part')
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, 'wb') as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.replace(path)
            return path.stat().st_size
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt + 1 == retries:
                raise
            time.sleep(2 * (attempt + 1))


def fetch_polyhaven(out, workers=8):
    root = out / 'polyhaven'
    root.mkdir(parents=True, exist_ok=True)
    models = get_json(POLYHAVEN_ASSETS)
    catalog = []
    tasks = []
    for asset_id, info in sorted(models.items()):
        try:
            files = get_json(POLYHAVEN_FILES.format(asset=asset_id))
            entry = files.get('gltf', {}).get('1k', {}).get('gltf')
            if not entry or not entry.get('url'):
                continue
            asset_dir = root / asset_id
            tasks.append((asset_id, asset_dir, entry))
            catalog.append(dict(asset_id=asset_id, name=info.get('name'),
                                categories=info.get('categories', []),
                                polycount=info.get('polycount'),
                                license='CC0-1.0', source='polyhaven'))
        except Exception as exc:
            print(f'polyhaven metadata failed {asset_id}: {exc}', flush=True)
    print(f'polyhaven: {len(tasks)} downloadable models', flush=True)

    def one(item):
        asset_id, asset_dir, entry = item
        try:
            gltf_path = asset_dir / f'{asset_id}_1k.gltf'
            download(entry['url'], gltf_path)
            for rel, meta in entry.get('include', {}).items():
                download(meta['url'], asset_dir / rel)
            return asset_id, None
        except Exception as exc:
            return asset_id, str(exc)

    failures = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for asset_id, error in pool.map(one, tasks):
            if error:
                failures += 1
                print(f'polyhaven download failed {asset_id}: {error}', flush=True)
    catalog = [row for row in catalog if (root / row['asset_id']).is_dir()]
    (root / 'catalog.json').write_text(json.dumps(dict(source='polyhaven', license='CC0-1.0',
                                                      assets=catalog), indent=2) + '\n')
    print(f'polyhaven: saved {len(catalog)} models ({failures} failures)', flush=True)


def fetch_objaverse(out, workers=8):
    import objaverse
    root = out / 'objaverse'
    root.mkdir(parents=True, exist_ok=True)
    ann = objaverse.load_lvis_annotations()
    ranked = sorted(((len(v), k) for k, v in ann.items()), reverse=True)
    categories = [k for _, k in ranked[:OBJAVERSE_CATEGORIES]]
    rng = random.Random(OBJAVERSE_SELECTION_SEED)
    chosen = {}
    total = 0
    for cat in categories:
        pool = sorted(ann[cat])
        rng.shuffle(pool)
        take = min(OBJAVERSE_PER_CATEGORY, len(pool))
        chosen[cat] = pool[:take]
        total += take
    uids = sorted(u for pool in chosen.values() for u in pool)
    if len(uids) > OBJAVERSE_LIMIT:
        rng.shuffle(uids)
        uids = sorted(uids[:OBJAVERSE_LIMIT])
    selection = dict(source='objaverse', annotations='lvis', seed=OBJAVERSE_SELECTION_SEED,
                     categories=categories, per_category=OBJAVERSE_PER_CATEGORY,
                     chosen=chosen)
    (root / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(f'objaverse: selected {len(uids)} objects across {len(categories)} categories', flush=True)
    paths = {}
    failures = []
    for start in range(0, len(uids), 200):
        chunk = uids[start:start + 200]
        try:
            paths.update(objaverse.load_objects(chunk, download_processes=workers))
        except Exception as exc:
            print(f'objaverse chunk {start} failed: {exc}', flush=True)
            failures.extend(chunk)
        print(f'objaverse: {min(start + 200, len(uids))}/{len(uids)} attempted', flush=True)
    records = []
    for uid, path in sorted(paths.items()):
        path = Path(path)
        if not path.exists():
            continue
        records.append(dict(uid=uid, path=str(path), bytes=path.stat().st_size,
                            source='objaverse', license='Sketchfab per-object (see Objaverse)'))
    (root / 'catalog.json').write_text(json.dumps(dict(source='objaverse',
                                                      assets=records), indent=2) + '\n')
    print(f'objaverse: {len(records)} usable GLBs ({len(failures)} chunk failures)', flush=True)


def fetch_thingi(out, limit=THINGI_LIMIT, workers=8):
    root = out / 'thingi'
    root.mkdir(parents=True, exist_ok=True)
    base = 'https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/main'
    ids_path = root / 'file_ids.txt'
    download(f'{base}/metadata/file_ids.txt', ids_path)
    ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    rng = random.Random(OBJAVERSE_SELECTION_SEED)
    rng.shuffle(ids)
    ids = sorted(ids[:limit])
    print(f'thingi: {len(ids)} of {len(ids_path.read_text().splitlines())} npz meshes selected', flush=True)

    def one(file_id):
        try:
            download(f'{base}/npz/{file_id}.npz', root / 'npz' / f'{file_id}.npz')
            return file_id, None
        except Exception as exc:
            return file_id, str(exc)

    failures = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for file_id, error in pool.map(one, ids):
            if error:
                failures += 1
                print(f'thingi failed {file_id}: {error}', flush=True)
    records = [dict(file_id=p.stem, path=str(p), source='thingi10k',
                    license='Thingi10K per-asset (CC)')
               for p in sorted((root / 'npz').glob('*.npz'))]
    (root / 'catalog.json').write_text(json.dumps(dict(source='thingi10k',
                                                      assets=records), indent=2) + '\n')
    print(f'thingi: saved {len(records)} npz meshes ({failures} failures)', flush=True)


def fetch_quaternius(out):
    root = out / 'quaternius'
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'UltimateModularCharacters'
    if target.exists() and any(target.rglob('*.glb')):
        print('quaternius: already present', flush=True)
        return
    url = 'https://drive.google.com/drive/folders/1USAAquX2JJWuA2m6zol0KUkFe3UkZ8zX'
    try:
        subprocess.run([sys.executable, '-m', 'gdown', '--folder', url, '-O', str(target)],
                       check=True)
        count = len(list(target.rglob('*.glb')))
        print(f'quaternius: downloaded folder ({count} GLBs)', flush=True)
    except Exception as exc:
        print(f'quaternius: optional download failed ({exc}); continuing without it', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='/home/sumanth/forge-data/raw')
    parser.add_argument('sources', nargs='*',
                        default=['polyhaven', 'objaverse', 'thingi', 'quaternius'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--thingi-limit', type=int, default=THINGI_LIMIT)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        started = time.time()
        if source == 'polyhaven':
            fetch_polyhaven(out, args.workers)
        elif source == 'objaverse':
            fetch_objaverse(out, args.workers)
        elif source == 'thingi':
            fetch_thingi(out, args.thingi_limit, args.workers)
        elif source == 'quaternius':
            fetch_quaternius(out)
        else:
            raise SystemExit(f'unknown source {source}')
        print(f'{source}: done in {time.time() - started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
