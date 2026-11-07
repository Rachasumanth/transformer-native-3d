"""End-to-end FORGE-3D pipeline: raw mesh -> polished, textured GLB.

Usage:
  python -m forge.pipeline raw.glb out.glb --gct path/to/forge-gct-best.pt \
      --mat path/to/forge-mat-best.pt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .constants import FINE, GRID, N_INPUT_POINTS, N_SURFACE, unpack_bits
from .geometry import (closest_face_bary, load_rich, normalize_mesh, occupancy_bits,
                       postprocess_mesh, sample_surface, sdf_to_mesh)
from .model_gct import GeometryCompletionTransformer
from .model_mat import MaterialTransformer

from t3d.glb import pack_static_glb, validate_glb_semantics


def _clean_state(state):
    if 'model' in state and isinstance(state['model'], dict):
        state = state['model']
    return {key.replace('module.', '', 1): value for key, value in state.items()}


def load_gct(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config') or {}
    model = GeometryCompletionTransformer(width=config.get('width', 384),
                                          heads=config.get('heads', 8),
                                          depth=config.get('depth', 8))
    model.load_state_dict(_clean_state(checkpoint))
    model.to(device).eval()
    return model


def load_mat(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config') or {}
    model = MaterialTransformer(width=config.get('width', 256),
                                heads=config.get('heads', 8),
                                depth=config.get('depth', 6))
    model.load_state_dict(_clean_state(checkpoint))
    model.to(device).eval()
    return model


class Polisher:
    def __init__(self, gct_path, mat_path=None, device=None, target_faces=80000,
                 smooth_iters=2):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.gct = load_gct(gct_path, self.device)
        self.mat = load_mat(mat_path, self.device) if mat_path else None
        self.target_faces = target_faces
        self.smooth_iters = smooth_iters
        self._grid = None

    def _queries(self):
        if self._grid is None:
            axis = -0.5 + (np.arange(FINE, dtype=np.float32) + 0.5) / FINE
            gx, gy, gz = np.meshgrid(axis, axis, axis, indexing='ij')
            self._grid = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
        return self._grid

    @torch.no_grad()
    def _predict_sdf(self, occ, points, chunk=262144):
        feature_field, _ = self.gct.encode(occ, points)
        queries = self._queries()
        values = np.empty(len(queries), dtype=np.float32)
        for start in range(0, len(queries), chunk):
            block = torch.from_numpy(queries[start:start + chunk]).to(self.device)
            block = block.unsqueeze(0)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.device.type == 'cuda'):
                predicted = self.gct.sdf(feature_field, block)
            values[start:start + chunk] = predicted[0].float().cpu().numpy()
        return values.reshape(FINE, FINE, FINE)

    @torch.no_grad()
    def _predict_pbr(self, occ, coords, normals, hint, has_tex, chunk=15000):
        if self.mat is None:
            return np.tile(np.array([0.8, 0.8, 0.8, 0.0, 0.6, 1.0], dtype=np.float32),
                           (len(coords), 1))
        output = np.empty((len(coords), 6), dtype=np.float32)
        flag = torch.tensor([float(has_tex)], dtype=torch.float32, device=self.device)
        for start in range(0, len(coords), chunk):
            block = slice(start, start + chunk)
            coords_t = torch.from_numpy(coords[block]).to(self.device).unsqueeze(0)
            normals_t = torch.from_numpy(normals[block]).to(self.device).unsqueeze(0)
            hint_t = torch.from_numpy(hint[block]).to(self.device).unsqueeze(0)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.device.type == 'cuda'):
                predicted = self.mat(occ, coords_t, normals_t, hint_t, flag)
            output[block] = predicted[0].float().cpu().numpy()
        return np.clip(output, 0.0, 1.0)

    @torch.no_grad()
    def polish_mesh(self, raw_mesh, rich=None, rich_mesh=None, normalization=None):
        raw_mesh = raw_mesh.copy()
        if not len(raw_mesh.faces) or not len(raw_mesh.vertices):
            raise ValueError('input mesh has no faces')
        if normalization is not None:
            center = np.asarray(normalization[0], dtype=np.float64)
            scale = float(normalization[1])
            normalized = raw_mesh.copy()
            normalized.vertices = (np.asarray(raw_mesh.vertices, dtype=np.float64) - center) * scale
        else:
            normalized, center, scale = normalize_mesh(raw_mesh)
        occupancy = unpack_bits(occupancy_bits(normalized, GRID), GRID)
        occ = torch.from_numpy(occupancy)[None, None].to(self.device)

        points, normals, _ = sample_surface(normalized, N_INPUT_POINTS)
        pts = np.concatenate([np.clip(points, -0.5, 0.5), normals], axis=1).astype(np.float32)
        pts_t = torch.from_numpy(pts)[None].to(self.device)

        with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                            enabled=self.device.type == 'cuda'):
            feature_field, _ = self.gct.encode(occ, pts_t)
        queries = self._queries()
        chunk = 262144
        values = np.empty(len(queries), dtype=np.float32)
        for start in range(0, len(queries), chunk):
            block = torch.from_numpy(queries[start:start + chunk]).to(self.device)[None]
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.device.type == 'cuda'):
                predicted = self.gct.sdf(feature_field, block)
            values[start:start + chunk] = predicted[0].float().cpu().numpy()
        grid = values.reshape(FINE, FINE, FINE)

        mesh = sdf_to_mesh(grid, FINE)
        mesh = postprocess_mesh(mesh, target_faces=self.target_faces,
                                smooth_iters=self.smooth_iters)
        result = mesh.copy()

        pbr = None
        has_tex = 0
        if self.mat is not None:
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            vertex_normals = _safe_normals(mesh).astype(np.float64)
            if rich is not None:
                source = (rich_mesh if rich_mesh is not None else raw_mesh).copy()
                source.vertices = (np.asarray(source.vertices, dtype=np.float64) - center) * scale
                hint = _sample_rich(rich, source, vertices)[:, :3]
                has_tex = int(bool(getattr(rich, 'has_texture', False)))
                if not has_tex:
                    hint[:] = 0.0
            else:
                hint = np.zeros((len(vertices), 3), dtype=np.float64)
            pbr = self._predict_pbr(occ, vertices.astype(np.float32),
                                    vertex_normals.astype(np.float32),
                                    hint.astype(np.float32), has_tex)
            result.vertices = vertices / scale + center
        else:
            result.vertices = np.asarray(mesh.vertices, dtype=np.float64) / scale + center
        stats = {
            'vertices': int(len(result.vertices)),
            'faces': int(len(result.faces)),
            'center': [float(x) for x in center],
            'scale': float(scale),
            'target_faces': self.target_faces,
            'material': bool(self.mat is not None),
            'has_tex': bool(has_tex),
        }
        return result, pbr, stats


def _safe_normals(mesh):
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    bad = (~np.isfinite(normals).all(axis=1)) | (np.linalg.norm(normals, axis=1) < 1e-8)
    if bad.any():
        normals[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return normals


def _sample_rich(rich, mesh, points):
    face_ids, bary = closest_face_bary(mesh, points)
    return rich.sample(face_ids, bary)


def _source_up(path):
    suffix = Path(path).suffix.lower()
    return 'y' if suffix in ('.glb', '.gltf') else 'z'


def process_file(polisher, path, out_path, *, source_up=None, target_faces=None):
    path, out_path = Path(path), Path(out_path)
    started = time.time()
    rich_mesh, rich = load_rich(path)
    mesh, pbr, stats = polisher.polish_mesh(rich_mesh, rich)
    up = source_up or _source_up(path)
    data = pack_static_glb(np.asarray(mesh.vertices, dtype=np.float32),
                           np.asarray(mesh.faces, dtype=np.int32),
                           normals=_safe_normals(mesh),
                           pbr=pbr, source_up=up)
    validate_glb_semantics(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    stats.update(source=str(path), output=str(out_path), source_up=up,
                 seconds=round(time.time() - started, 2), bytes=len(data))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--gct', required=True)
    parser.add_argument('--mat', default='')
    parser.add_argument('--device', default='')
    parser.add_argument('--target-faces', type=int, default=80000)
    parser.add_argument('--source-up', choices=['y', 'z'], default=None)
    args = parser.parse_args()
    polisher = Polisher(args.gct, args.mat or None, args.device or None,
                        target_faces=args.target_faces)
    stats = process_file(polisher, args.input, args.output, source_up=args.source_up)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
