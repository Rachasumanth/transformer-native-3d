"""Independently verify native audit identities and source reconstruction metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from t3d.container import load
from t3d.extraction import voxels_to_mesh
from t3d.import_glb import load_glb
from t3d.metrics import surface_comparison
from t3d.native_voxelize import UPSTREAM_COMMIT
from t3d.normalization import normalize_asset


def audit(outputs, sources, expected_source_sha256):
    outputs, sources = Path(outputs), Path(sources)
    recorded = json.loads((outputs / 'native-converter-report.json').read_text())
    if recorded['source_sha256'] != expected_source_sha256:
        raise ValueError('Notebook source bundle identity mismatch')
    if recorded['upstream_commit'] != UPSTREAM_COMMIT:
        raise ValueError('Upstream converter revision mismatch')
    names = ('Avocado', 'WaterBottle', 'Lantern')
    if set(recorded['assets']) != set(names):
        raise ValueError('Incomplete or unexpected native asset set')
    # Validate every identity before spending time on any geometric measurements.
    inputs = []
    for name in names:
        item = recorded['assets'][name]
        if set(item['resolutions']) != {'16', '32'}:
            raise ValueError(f'Incomplete or unexpected resolution set for {name}')
        source = load_glb(sources / f'{name}.glb')
        if source.source_sha256 != item['source_sha256']:
            raise ValueError(f'Source asset identity mismatch: {name}')
        reference, _, _ = normalize_asset(source.vertices)
        reference = np.clip(reference, -.5, .5)
        for size in (16, 32):
            path = outputs / f'native-{name}-{size}.t3d'
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item['resolutions'][str(size)]['output_sha256']:
                raise ValueError(f'Native output identity mismatch: {path.name}')
            asset = load(path).asset
            if tuple(asset.grid_size) != (size,)*3 or not np.array_equal(asset.aabb, [[-.5]*3, [.5]*3]):
                raise ValueError(f'Native output grid or bounds mismatch: {path.name}')
            inputs.append((name, size, source, reference, asset, digest))
    report = dict(source_sha256=expected_source_sha256, upstream_commit=UPSTREAM_COMMIT,
                  scope='three source assets at two resolutions; sampled surface metrics, not research benchmark',
                  cases=[])
    for name, size, source, reference, asset, digest in inputs:
        vertices, faces = voxels_to_mesh(asset)
        measurements = [surface_comparison(reference, source.faces, vertices, faces,
                                          samples=256, seed=seed) for seed in (0, 1, 2)]
        report['cases'].append(dict(asset=name, grid_size=size, source_sha256=source.source_sha256,
                                   output_sha256=digest, measurements=measurements))
        print(name, size, [(m.get('fscore'), m.get('nonmanifold_edges')) for m in measurements], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.outputs, args.sources, args.source_sha256)
    args.report.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
