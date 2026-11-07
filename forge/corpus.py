"""Build compact FORGE-3D training records from raw 3D assets.

One npz per asset (~300-500 KB) with raw-input tensors, near-surface SDF
labels at 128^3 support, clean 64^3 occupancy, PBR targets at 64^3 support,
and optional rig targets. Sidecar JSON carries provenance and split data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
import traceback

import numpy as np

from .constants import (GRID, FINE, N_INPUT_POINTS, N_QUERY, N_SURFACE,
                        pack_bits, quantize_normals, quantize_sdf)
from .corrupt import corrupt_mesh, degrade_colors
from .geometry import (closest_face_bary, load_rich, normalize_mesh,
                       occupancy_bits, sample_surface, sanitize_mesh, sdf_at,
                       build_sdf_queries)
from .rig import load_rig, sample_rig

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_SOURCE_MB = float(os.environ.get('FORGE_MAX_SOURCE_MB', '80'))
SOURCE_LICENSES = {
    'kenney': 'CC0-1.0',
    'polyhaven': 'CC0-1.0',
    'objaverse': 'Sketchfab per-object',
    'thingi10k': 'Thingi10K per-asset',
    'quaternius': 'CC0-1.0',
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_id(path, source):
    stem = Path(path).stem
    digest = hashlib.sha1(f'{source}:{path}'.encode()).hexdigest()[:12]
    return f'{source}-{stem}-{digest}'


def _rng_for(asset_id):
    seed = int.from_bytes(hashlib.sha256(asset_id.encode()).digest()[:8], 'little')
    return np.random.default_rng(seed)


def _bary_to_srgb(rich, mesh, points):
    face_ids, bary = closest_face_bary(mesh, points)
    return rich.sample(face_ids, bary)


def build_record(path, source, *, asset_id=None, license=None, n_query=N_QUERY,
                 keep_raw=False):
    """Return (arrays dict, meta dict). Raises on unusable assets."""
    path = Path(path)
    asset_id = asset_id or _asset_id(path, source)
    rng = _rng_for(asset_id)
    rich_mesh, rich = load_rich(path)
    if not len(rich_mesh.faces):
        raise ValueError('no faces')
    textured = bool(getattr(rich, 'has_texture', False))
    colored = bool(np.any(np.abs(rich.base_factor[:, :3] - 1.0) > 0.01))
    has_material = textured or colored or rich.vertex_colors is not None
    clean = sanitize_mesh(rich_mesh)
    clean, center, scale = normalize_mesh(clean)
    rich_vertices = (np.asarray(rich_mesh.vertices, dtype=np.float64) - center) * scale
    rich_mesh = rich_mesh.copy()
    rich_mesh.vertices = rich_vertices
    if np.asarray(rich_mesh.faces).max() >= len(rich_vertices):
        raise ValueError('rich face index out of range')

    coords, query_normals = build_sdf_queries(clean, n_surface=N_SURFACE, fine=FINE, rng=rng)
    if len(coords) != n_query:
        raise ValueError(f'expected {n_query} SDF queries, got {len(coords)}')
    sdf = sdf_at(clean, coords)
    sdf = np.clip(sdf, -4.5 / FINE, 4.5 / FINE)

    surface_count = min(N_SURFACE, len(coords))
    pbr_values = _bary_to_srgb(rich, rich_mesh, coords[:surface_count])
    pbr_values = np.clip(pbr_values, 0.0, 1.0)
    if not has_material:
        pbr_values[:] = 0.0
    has_tex = int(has_material)

    raw_mesh = None
    recipe = None
    for attempt in range(4):
        candidate, candidate_recipe = corrupt_mesh(clean, rng)
        candidate_bits = occupancy_bits(candidate, GRID)
        fill = np.unpackbits(candidate_bits).mean()
        if fill > 0.0001 and fill < 0.85:
            raw_mesh, recipe = candidate, candidate_recipe
            break
        recipe = candidate_recipe
    if raw_mesh is None:
        raise ValueError('corruption produced degenerate occupancy')

    raw_points, raw_normals, _ = sample_surface(raw_mesh, N_INPUT_POINTS)
    raw_points = np.clip(raw_points, -0.5, 0.5)
    if has_material:
        raw_colors = _bary_to_srgb(rich, rich_mesh, raw_points)[:, :3]
        raw_colors = degrade_colors(raw_colors, rng)
        raw_colors = np.rint(np.clip(raw_colors, 0, 1) * 255).astype(np.uint8)
    else:
        raw_colors = np.zeros((len(raw_points), 3), dtype=np.uint8)

    if has_material:
        pbr_hint = degrade_colors(pbr_values[:, :3].copy(), rng)
        pbr_hint = np.rint(np.clip(pbr_hint, 0, 1) * 255).astype(np.uint8)
    else:
        pbr_hint = np.zeros((surface_count, 3), dtype=np.uint8)

    rig = load_rig(path)
    arrays = {
        'occ64': occupancy_bits(raw_mesh, GRID),
        'occ_target64': occupancy_bits(clean, GRID),
        'pts': np.concatenate([raw_points, raw_normals], axis=1).astype(np.float16),
        'cols': raw_colors,
        'has_tex': np.array([has_tex], dtype=np.uint8),
        'sdf': quantize_sdf(sdf, FINE),
        'sdf_coords': coords.astype(np.float16),
        'sdf_normals': quantize_normals(query_normals),
        'pbr': np.rint(pbr_values * 255).astype(np.uint8),
        'pbr_raw': pbr_hint,
    }
    if keep_raw:
        arrays['raw_vertices'] = np.asarray(raw_mesh.vertices, dtype=np.float16)
        arrays['raw_faces'] = np.asarray(raw_mesh.faces, dtype=np.int32)
    meta = {
        'asset_id': asset_id,
        'source': source,
        'license': license or SOURCE_LICENSES.get(source, 'unknown'),
        'source_path': str(path),
        'source_sha256': _sha256(path),
        'center': [float(x) for x in center],
        'scale': float(scale),
        'has_pbr': bool(has_tex),
        'has_rig': False,
        'faces_clean': int(len(clean.faces)),
        'faces_raw': int(len(raw_mesh.faces)),
        'n_query': int(len(coords)),
        'recipe': recipe,
        'has_raw': bool(keep_raw),
    }
    if rig is not None:
        joint_positions = (np.asarray(rig.joints, dtype=np.float64) - center) * scale
        rig_vertices = (np.asarray(rig.vertices, dtype=np.float64) - center) * scale
        from .rig import RigData
        normalized_rig = RigData(joint_positions, rig.parents, rig_vertices,
                                 rig.joint_ids, rig.weights)
        weights = sample_rig(normalized_rig, coords[:surface_count])
        if weights is not None and weights.shape[1] <= 24:
            arrays['rig_joints'] = joint_positions.astype(np.float16)
            arrays['rig_parents'] = np.asarray(rig.parents, dtype=np.int16)
            arrays['rig_skin'] = np.rint(np.clip(weights, 0, 1) * 255).astype(np.uint8)
            meta['has_rig'] = True
            meta['joint_count'] = int(len(joint_positions))
    return arrays, meta


def _task_list(sources, limit=0):
    tasks = []
    if 'kenney' in sources:
        root = REPO_ROOT / 'artifacts/kenney-catalog-download'
        for path in sorted(root.glob('*/*.glb')):
            tasks.append((path, 'kenney', None))
    if 'polyhaven' in sources:
        root = Path('/home/sumanth/forge-data/raw/polyhaven')
        for path in sorted(root.glob('*/*_1k.gltf')):
            tasks.append((path, 'polyhaven', None))
    if 'objaverse' in sources:
        catalog = Path('/home/sumanth/forge-data/raw/objaverse/catalog.json')
        if catalog.exists():
            for row in json.loads(catalog.read_text())['assets']:
                tasks.append((Path(row['path']), 'objaverse', row['uid']))
        else:
            cache = Path.home() / '.objaverse/hf-objaverse-v1/glbs'
            for path in sorted(cache.glob('*/*.glb')):
                tasks.append((path, 'objaverse', path.stem))
    if 'thingi' in sources:
        catalog = Path('/home/sumanth/forge-data/raw/thingi/catalog.json')
        if catalog.exists():
            for row in json.loads(catalog.read_text())['assets']:
                tasks.append((Path(row['path']), 'thingi10k', row['file_id']))
    if 'quaternius' in sources:
        root = Path('/home/sumanth/forge-data/raw/quaternius')
        for path in sorted(root.rglob('*.glb')):
            tasks.append((path, 'quaternius', None))
    if limit:
        tasks = tasks[:limit]
    return tasks


def _split_for(asset_id, source):
    digest = hashlib.sha256(asset_id.encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 3:
        return 'test'
    if bucket < 8:
        return 'validation'
    return 'train'


def _source_size(path):
    path = Path(path)
    if path.suffix.lower() == '.gltf':
        return sum(item.stat().st_size for item in path.parent.iterdir() if item.is_file())
    return path.stat().st_size


def build_asset(task):
    path, source, source_id = task
    try:
        if _source_size(path) > MAX_SOURCE_MB * 1024 * 1024:
            raise ValueError(f'source larger than {MAX_SOURCE_MB:.0f} MB')
        asset_id = _asset_id(path, source) if source_id is None else f'{source}-{source_id}'
        split = _split_for(asset_id, source)
        arrays, meta = build_record(path, source, asset_id=asset_id,
                                    keep_raw=split == 'test')
        meta['split'] = split
        meta['path'] = f'assets/{asset_id}.npz'
        digest = hashlib.sha256()
        for key in sorted(arrays):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(arrays[key]).tobytes())
        meta['sha256'] = digest.hexdigest()
        return asset_id, arrays, meta, None
    except Exception as exc:
        return None, None, None, f'{Path(path)}: {type(exc).__name__}: {exc}'



def build_corpus(tasks, out_dir, *, workers=6, limit=0, progress_every=25):
    import multiprocessing as mp
    out_dir = Path(out_dir)
    assets_dir = out_dir / 'assets'
    assets_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    skipped_existing = 0
    for path, source, source_id in tasks:
        asset_id = _asset_id(path, source) if source_id is None else f'{source}-{source_id}'
        target = assets_dir / f'{asset_id}.npz'
        if target.exists() and target.with_suffix('.json').exists():
            skipped_existing += 1
            continue
        pending.append((path, source, source_id))
    if limit:
        pending = pending[:limit]
    print(f'corpus: {len(pending)} pending, {skipped_existing} already built', flush=True)
    started = time.time()
    failures = []
    done = 0
    records = []
    context = mp.get_context('fork')

    def init():
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        try:
            import resource
            raw = os.environ.get('FORGE_WORKER_MEM_GB', '').strip()
            if raw:
                limit = int(float(raw) * 1024 ** 3)
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except Exception:
            pass

    prepared = []
    for path, source, source_id in pending:
        asset_id = _asset_id(path, source) if source_id is None else f'{source}-{source_id}'
        prepared.append((path, source, source_id, assets_dir / f'{asset_id}.npz'))
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool

    attempts = 6
    failed_targets = set()
    for attempt in range(attempts):
        remaining = [row for row in prepared
                     if str(row[3]) not in failed_targets
                     and not (row[3].exists() and row[3].with_suffix('.json').exists())]
        if not remaining:
            break
        print(f'corpus: attempt {attempt + 1} with {len(remaining)} pending', flush=True)
        broken = False
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                     initializer=init) as executor:
                futures = {executor.submit(_execute_task, row): row for row in remaining}
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        target, _, error = future.result()
                    except BrokenProcessPool:
                        broken = True
                        break
                    except Exception as exc:
                        target, error = row[3], f'{row[3]}: {type(exc).__name__}: {exc}'
                    done += 1
                    if error:
                        failures.append(error)
                        failed_targets.add(str(target))
                    if done % progress_every == 0:
                        rate = done / max(time.time() - started, 1e-6)
                        print(f'corpus: {done}/{len(remaining)} ({rate:.2f}/s, '
                              f'{len(failures)} failures)', flush=True)
        except (BrokenProcessPool, Exception) as exc:  # executor died
            print(f'corpus: pool restart {type(exc).__name__}: {exc}', flush=True)
            broken = True
        if broken or not [row for row in remaining
                          if not (row[3].exists() and row[3].with_suffix('.json').exists())]:
            if broken:
                print('corpus: broken pool, retrying remaining', flush=True)
            continue
    manifest = rebuild_manifest(out_dir)
    manifest['failures'] = failures
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'corpus: built {len(manifest["assets"])} assets, {len(failures)} failures '
          f'in {time.time() - started:.1f}s', flush=True)
    return manifest


def _execute_task(prepared):
    import signal

    path, source, source_id, out = prepared
    task = (path, source, source_id)

    def timeout(signum, frame):
        raise TimeoutError('asset build exceeded 300s')

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 300.0)
    try:
        asset_id, arrays, meta, error = build_asset(task)
        if error:
            return out, None, error
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, **arrays)
            (out.with_suffix('.json')).write_text(json.dumps(meta, indent=2) + '\n')
            return out, meta, None
        except Exception as exc:
            return out, None, f'{out}: {type(exc).__name__}: {exc}'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)


def rebuild_manifest(out_dir):
    out_dir = Path(out_dir)
    records = []
    for path in sorted((out_dir / 'assets').glob('*.json')):
        records.append(json.loads(path.read_text()))
    manifest = {
        'version': 1,
        'format': 'forge3d-npz-v1',
        'grid': GRID,
        'fine': FINE,
        'sdf_clip_voxels': 4.0,
        'assets': records,
        'failures': [],
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def pack_shards(out_dir, shard_size=250):
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / 'manifest.json').read_text())
    shard_dir = out_dir / 'shards'
    shard_dir.mkdir(exist_ok=True)
    assets = manifest['assets']
    shards = []
    for start in range(0, len(assets), shard_size):
        chunk = assets[start:start + shard_size]
        shard_path = shard_dir / f'shard-{start // shard_size:04d}.tar'
        with tarfile.open(shard_path, 'w') as archive:
            for row in chunk:
                npz = out_dir / row['path']
                sidecar = npz.with_suffix('.json')
                archive.add(npz, arcname=row['path'])
                archive.add(sidecar, arcname=str(Path(row['path']).with_suffix('.json')))
            index = json.dumps({'assets': chunk}, indent=2).encode()
            info = tarfile.TarInfo('index.json')
            info.size = len(index)
            import io
            archive.addfile(info, io.BytesIO(index))
        shards.append(dict(path=str(shard_path.relative_to(out_dir)),
                           bytes=shard_path.stat().st_size, assets=len(chunk)))
        print(f'packed {shard_path.name}: {len(chunk)} assets, '
              f'{shard_path.stat().st_size / 1e6:.1f} MB', flush=True)
    manifest['shards'] = shards
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='/home/sumanth/forge-data/corpus-v1')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--sources', nargs='*',
                        default=['kenney', 'polyhaven', 'objaverse', 'thingi'])
    parser.add_argument('--pack', action='store_true')
    parser.add_argument('--shard-size', type=int, default=250)
    args = parser.parse_args()
    tasks = _task_list(args.sources, args.limit)
    print(f'corpus: {len(tasks)} source assets selected', flush=True)
    build_corpus(tasks, args.out, workers=args.workers, limit=args.limit)
    if args.pack:
        pack_shards(args.out, args.shard_size)


if __name__ == '__main__':
    main()
