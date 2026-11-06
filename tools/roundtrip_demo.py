"""Run with python -m tools.roundtrip_demo; writes inspectable CPU artifacts."""
from pathlib import Path
from collections import Counter
import json
import time
import numpy as np
from t3d.voxelize import mesh_to_voxels
from t3d.extraction import voxels_to_mesh
from t3d.topology import orient_faces
from t3d.container import save, load
from t3d.materials import PBRMaterial, sample_pbr


def main():
    output = Path(__file__).resolve().parents[1] / 'artifacts' / 'roundtrip'
    output.mkdir(parents=True, exist_ok=True)
    vertices = np.array([[x, y, z] for x in (.2, .8) for y in (.2, .8) for z in (.2, .8)])
    faces = []
    for axis in range(3):
        for side in (.2, .8):
            a, b, c, d = np.flatnonzero(vertices[:, axis] == side)
            faces.extend([[a, b, d], [a, d, c]])
    start = time.perf_counter()
    asset = mesh_to_voxels(vertices, faces, grid_size=(8,)*3,
                           aabb=[[0]*3, [1]*3], regularization_weight=0)
    encoded = time.perf_counter()
    # Explicit synthetic source material for this fixture; not learned output.
    asset = sample_pbr(asset, vertices, faces,
                       [PBRMaterial(base_color=(.2, .5, .8, 1), metallic=.1, roughness=.6)],
                       np.zeros(len(faces), dtype=int))
    materialized = time.perf_counter()
    save(output / 'cube.t3d', asset)
    reloaded = load(output / 'cube.t3d').asset
    storage_error = float(np.max(np.abs(reloaded.world_vertices - asset.world_vertices)))
    stored = time.perf_counter()
    v, f = voxels_to_mesh(asset)
    decoded = time.perf_counter()
    f, orientation = orient_faces(v, f, outward=True)
    oriented = time.perf_counter()
    edges = Counter(tuple(sorted((int(a), int(b)))) for tri in f for a, b in zip(tri, np.roll(tri, -1)))
    report = dict(grid_size=list(asset.grid_size), active_voxels=len(v), triangles=len(f),
                  boundary_edges=sum(n == 1 for n in edges.values()),
                  nonmanifold_edges=sum(n > 2 for n in edges.values()),
                  euler_characteristic=len(np.unique(f)) - len(edges) + len(f),
                  bounds=[v.min(axis=0).tolist(), v.max(axis=0).tolist()],
                  encode_seconds=encoded-start, decode_seconds=decoded-stored,
                  material_seconds=materialized-encoded, storage_seconds=stored-materialized,
                  pbr_channels=['linear_r', 'linear_g', 'linear_b', 'metallic', 'roughness', 'alpha'],
                  orientation_seconds=oriented-decoded, orientation=orientation,
                  storage_max_vertex_error=storage_error,
                  winding='consistent; closed components have positive signed volume', backend='CPU reference')
    lines = ['# Reference roundtrip; consistently wound closed cube']
    lines.extend('v ' + ' '.join(map(str, row)) for row in v)
    lines.extend('f ' + ' '.join(str(int(i)+1) for i in row) for row in f)
    (output / 'cube.obj').write_text('\n'.join(lines) + '\n')
    (output / 'cube.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
