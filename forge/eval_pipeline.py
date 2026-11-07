"""Evaluate the FORGE-3D pipeline on the held-out test split.

Reports chamfer distance and F-score of polished output against clean ground
truth, plus the unpolished raw-input baseline for comparison.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .geometry import (chamfer_distance, fscore, load_rich, normalize_mesh,
                       sample_points, sanitize_mesh)
from .pipeline import Polisher


def _metrics(mesh, truth, samples=20000):
    predicted = sample_points(mesh, samples)
    target = sample_points(truth, samples)
    chamfer, _ = chamfer_distance(predicted, target)
    return {
        'chamfer': float(chamfer),
        'fscore_005': fscore(predicted, target, 0.005),
        'fscore_010': fscore(predicted, target, 0.010),
        'fscore_020': fscore(predicted, target, 0.020),
    }


def evaluate(polisher, manifest_path, *, limit=0, out_dir='', quiet=False):
    manifest = json.loads(Path(manifest_path).read_text())
    rows = [row for row in manifest['assets']
            if row['split'] == 'test' and row.get('has_raw', False)]
    if limit:
        rows = rows[:limit]
    results, failures = [], []
    started = time.time()
    root = Path(manifest_path).parent
    for index, row in enumerate(rows):
        try:
            data = np.load(root / row['path'])
            vertices = np.asarray(data['raw_vertices'], dtype=np.float64)
            faces = np.asarray(data['raw_faces'], dtype=np.int64)
            import trimesh
            raw = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            rich_mesh, rich = load_rich(row['source_path'])
            source = sanitize_mesh(rich_mesh)
            source, _, _ = normalize_mesh(source)
            polished, pbr, stats = polisher.polish_mesh(
                raw, rich, rich_mesh=rich_mesh, normalization=(row['center'], row['scale']))
            polished_stats = _metrics(polished, source)
            raw_stats = _metrics(raw, source)
            entry = {'asset_id': row['asset_id'], 'source': row['source'],
                     'polished': polished_stats, 'raw': raw_stats,
                     'stats': stats}
            results.append(entry)
            if not quiet:
                print(index + 1, row['asset_id'], 'polished CD',
                      round(polished_stats['chamfer'], 6), 'raw CD',
                      round(raw_stats['chamfer'], 6), flush=True)
        except Exception as exc:
            failures.append(f"{row['asset_id']}: {type(exc).__name__}: {exc}")
            if not quiet:
                print(index + 1, row['asset_id'], 'FAILED', failures[-1], flush=True)

    def summarize(key):
        if not results:
            return {}
        return {
            'chamfer': float(np.mean([entry[key]['chamfer'] for entry in results])),
            'fscore_005': float(np.mean([entry[key]['fscore_005'] for entry in results])),
            'fscore_010': float(np.mean([entry[key]['fscore_010'] for entry in results])),
            'fscore_020': float(np.mean([entry[key]['fscore_020'] for entry in results])),
        }

    report = {
        'assets': len(results),
        'failures': failures,
        'polished': summarize('polished'),
        'raw': summarize('raw'),
        'seconds': round(time.time() - started, 1),
        'results': results,
    }
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'eval-report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--gct', required=True)
    parser.add_argument('--mat', default='')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out', default='')
    parser.add_argument('--device', default='')
    parser.add_argument('--target-faces', type=int, default=80000)
    args = parser.parse_args()
    polisher = Polisher(args.gct, args.mat or None, args.device or None,
                        target_faces=args.target_faces)
    report = evaluate(polisher, args.manifest, limit=args.limit, out_dir=args.out)
    print(json.dumps({key: report[key] for key in ('assets', 'failures', 'polished',
                                                   'raw', 'seconds')}, indent=2))


if __name__ == '__main__':
    main()
