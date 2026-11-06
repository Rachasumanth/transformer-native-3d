"""Compare source and exported PBR values at matched surface points.

This is an attribute-space audit after UV atlas repacking. It does not model
lighting, camera response, or image-space SSIM/LPIPS.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from t3d.import_rig import load_rig_glb
from t3d.materials import bilinear, closest_surface
from t3d.sampling import _mesh


CHANNELS = ('base_r', 'base_g', 'base_b', 'metallic', 'roughness', 'alpha')


def _sample_materials(points, face_ids, bary, materials, face_materials, corner_uv):
    result = np.empty((len(points), 6), dtype=float)
    query_uv = np.einsum('ni,nij->nj', bary, corner_uv[face_ids])
    for material_id, material in enumerate(materials):
        selected = np.flatnonzero(face_materials[face_ids] == material_id)
        if not len(selected):
            continue
        color = np.tile(material.base_color, (len(selected), 1))
        metallic = np.full(len(selected), material.metallic)
        roughness = np.full(len(selected), material.roughness)
        uv = query_uv[selected]
        if material.base_texture is not None:
            texture = bilinear(material.base_texture, uv, wrap=material.wrap, srgb=True)
            color[:, :3] *= texture[:, :3]
            if texture.shape[1] == 4:
                color[:, 3] *= texture[:, 3]
        if material.metallic_roughness_texture is not None:
            texture = bilinear(material.metallic_roughness_texture, uv, wrap=material.wrap)
            metallic *= texture[:, 2]
            roughness *= texture[:, 1]
        if material.alpha_mode == 'OPAQUE':
            color[:, 3] = 1
        elif material.alpha_mode == 'MASK':
            color[:, 3] = color[:, 3] >= material.alpha_cutoff
        result[selected] = np.column_stack((color[:, :3], metallic, roughness, color[:, 3]))
    return result


def _metrics(reference, prediction):
    error = prediction - reference
    absolute = np.abs(error)
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {
        'mae': float(absolute.mean()),
        'rmse': rmse,
        'p95_absolute_error': float(np.quantile(absolute, .95)),
        'psnr_db_peak_1': None if rmse == 0 else float(-20 * np.log10(rmse)),
    }


def audit(source_path, output_path, *, samples=4096, seed=0):
    if type(samples) is not int or samples <= 0:
        raise ValueError('samples must be a positive integer')
    source, _ = load_rig_glb(source_path, include_pbr=True).normalized()
    output = load_rig_glb(output_path, include_pbr=True)
    _, faces, _, areas = _mesh(output.vertices, output.faces)
    rng = np.random.default_rng(seed)
    owners = rng.choice(len(faces), size=samples, p=areas / areas.sum())
    random = rng.random((samples, 2))
    root = np.sqrt(random[:, 0])
    bary = np.column_stack((1 - root, root * (1 - random[:, 1]), root * random[:, 1]))
    points = np.einsum('ni,nij->nj', bary, output.vertices[faces[owners]])

    source_ids, source_bary = closest_surface(points, source.vertices, source.faces, method='bvh')
    source_points = np.einsum('ni,nij->nj', source_bary, source.vertices[source.faces[source_ids]])
    source_values = _sample_materials(points, source_ids, source_bary, source.pbr_materials,
                                      source.face_materials, source.corner_uv)
    output_values = _sample_materials(points, owners, bary, output.pbr_materials,
                                      output.face_materials, output.corner_uv)
    return {
        'source_sha256': hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(Path(output_path).read_bytes()).hexdigest(),
        'samples': samples,
        'seed': seed,
        'source_projection_distance_mean': float(np.linalg.norm(points - source_points, axis=1).mean()),
        'source_projection_distance_p95': float(np.quantile(np.linalg.norm(points - source_points, axis=1), .95)),
        'linear_pbr': {name: _metrics(source_values[:, i], output_values[:, i])
                       for i, name in enumerate(CHANNELS)},
        'scope': 'Area-weighted output surface samples compared with source PBR at nearest source-surface points; attribute-space only, not rendered material fidelity.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_glb', type=Path)
    parser.add_argument('output_glb', type=Path)
    parser.add_argument('--samples', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    report = audit(args.source_glb, args.output_glb, samples=args.samples, seed=args.seed)
    rendered = json.dumps(report, indent=2) + '\n'
    if args.json:
        args.json.write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
