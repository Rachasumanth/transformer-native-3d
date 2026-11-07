"""Direct, resilient Objaverse GLB fetcher using the object-path index.

Selects the smallest objects from the LVIS selection so downloads stay within
a practical budget, then records a catalog the corpus builder can consume.
"""
import argparse
import concurrent.futures as futures
import gzip
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

UA = {'User-Agent': 'forge3d-corpus/1.0 (research)'}
BASE = 'https://huggingface.co/datasets/allenai/objaverse/resolve/main/'
CACHE = Path.home() / '.objaverse/hf-objaverse-v1'


def request(url, method='GET', timeout=60):
    req = urllib.request.Request(url, headers=UA, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def head_size(url):
    try:
        with request(url, method='HEAD', timeout=30) as response:
            return int(response.headers.get('Content-Length') or 0)
    except Exception:
        return 0


def download(url, path, retries=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        return path.stat().st_size
    tmp = path.with_suffix('.part')
    for attempt in range(retries):
        try:
            with request(url, timeout=120) as response, open(tmp, 'wb') as handle:
                shutil.copyfileobj(response, handle, 1 << 20)
            tmp.replace(path)
            return path.stat().st_size
        except Exception:
            if tmp.exists():
                tmp.unlink()
            if attempt + 1 == retries:
                raise
            time.sleep(3 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', default='/home/sumanth/forge-data/raw/objaverse/selection.json')
    parser.add_argument('--catalog', default='/home/sumanth/forge-data/raw/objaverse/catalog.json')
    parser.add_argument('--limit', type=int, default=1500)
    parser.add_argument('--max-mb', type=float, default=18.0)
    parser.add_argument('--min-kb', type=float, default=60.0)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--head-workers', type=int, default=64)
    args = parser.parse_args()

    selection = json.loads(Path(args.selection).read_text())
    uids = [uid for category in selection['chosen'].values() for uid in category]
    paths = json.load(gzip.open(CACHE / 'object-paths.json.gz'))
    candidates = [(uid, paths[uid]) for uid in uids if uid in paths]
    print(f'candidates {len(candidates)}', flush=True)

    def size_of(item):
        uid, relative = item
        return uid, relative, head_size(BASE + relative)

    sized = []
    with futures.ThreadPoolExecutor(max_workers=args.head_workers) as pool:
        for uid, relative, size in pool.map(size_of, candidates):
            if args.min_kb * 1024 <= size <= args.max_mb * 1024 * 1024:
                sized.append((size, uid, relative))
    sized.sort()
    chosen = sized[:args.limit]
    print(f'sized {len(sized)}, downloading {len(chosen)} '
          f'({sum(row[0] for row in chosen) / 1e9:.2f} GB)', flush=True)

    started = time.time()
    failures = 0

    def fetch(item):
        size, uid, relative = item
        try:
            download(BASE + relative, CACHE / relative)
            return uid, None
        except Exception as exc:
            return uid, str(exc)

    done = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for uid, error in pool.map(fetch, chosen):
            done += 1
            if error:
                failures += 1
                print(f'failed {uid}: {error}', flush=True)
            if done % 50 == 0:
                rate = done / max(time.time() - started, 1e-6)
                print(f'objaverse-direct: {done}/{len(chosen)} ({rate:.2f}/s, '
                      f'{failures} failures)', flush=True)
                write_catalog(args.catalog)

    write_catalog(args.catalog)
    print(f'objaverse-direct: done {done} ({failures} failures) in '
          f'{time.time() - started:.0f}s', flush=True)


def write_catalog(path):
    records = []
    for item in sorted((CACHE / 'glbs').glob('*/*.glb')):
        records.append(dict(uid=item.stem, path=str(item), bytes=item.stat().st_size,
                            source='objaverse',
                            license='Sketchfab per-object (see Objaverse)'))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(source='objaverse', assets=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
