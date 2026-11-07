"""Datasets and collation for FORGE-3D training."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import (GRID, FINE, dequantize_normals, dequantize_sdf,
                        unpack_bits)


def srgb_to_linear(values):
    values = np.asarray(values, dtype=np.float32)
    return np.where(values <= 0.04045, values / 12.92,
                    ((values + 0.055) / 1.055) ** 2.4).astype(np.float32)


def collate(items):
    keys = items[0].keys()
    batch = {}
    for key in keys:
        values = [item[key] for item in items]
        first = values[0]
        if isinstance(first, str):
            batch[key] = values
        else:
            batch[key] = torch.from_numpy(np.stack(values))
    return batch


class ForgeDataset(Dataset):
    """Loads compact FORGE npz records for one split and task."""

    def __init__(self, manifest, split, *, task='gct', limit=0, ids=None,
                 max_assets=0, augment=False):
        self.manifest = Path(manifest).resolve()
        document = json.loads(self.manifest.read_text())
        if document.get('format') != 'forge3d-npz-v1':
            raise ValueError('unsupported corpus manifest')
        assets = [row for row in document['assets'] if row['split'] == split]
        if task == 'mat':
            assets = [row for row in assets if row.get('has_pbr')]
        if ids is not None:
            wanted = set(ids)
            assets = [row for row in assets if row['asset_id'] in wanted]
        if max_assets:
            assets = assets[:max_assets]
        if limit:
            assets = assets[:limit]
        if not assets:
            raise ValueError(f'no assets for split={split} task={task}')
        self.assets = assets
        self.root = self.manifest.parent
        self.task = task
        self.augment = augment

    def __len__(self):
        return len(self.assets)

    def __getitem__(self, index):
        row = self.assets[index]
        data = np.load(self.root / row['path'])
        if self.task == 'gct':
            occ = unpack_bits(data['occ64'])[None]
            target = unpack_bits(data['occ_target64'])[None]
            points = np.asarray(data['pts'], dtype=np.float32)
            if self.augment:
                occ, points = _augment_gct(occ, points)
            return {
                'asset_id': row['asset_id'],
                'occ': occ.astype(np.float32),
                'target_occ': target.astype(np.float32),
                'pts': points,
                'sdf_coords': np.asarray(data['sdf_coords'], dtype=np.float32),
                'sdf': dequantize_sdf(np.asarray(data['sdf'])),
            }
        if self.task == 'mat':
            count = len(data['pbr'])
            coords = np.asarray(data['sdf_coords'][:count], dtype=np.float32)
            normals = dequantize_normals(np.asarray(data['sdf_normals'][:count]))
            pbr = np.asarray(data['pbr'], dtype=np.float32) / 255.0
            hint = np.asarray(data['pbr_raw'], dtype=np.float32) / 255.0
            color_linear = srgb_to_linear(pbr[:, :3])
            target = np.concatenate([color_linear, pbr[:, 3:6]], axis=1)
            return {
                'asset_id': row['asset_id'],
                'occ': unpack_bits(data['occ64'])[None].astype(np.float32),
                'coords': coords,
                'normals': normals.astype(np.float32),
                'hint': hint,
                'pbr': target.astype(np.float32),
                'has_tex': np.float32(row.get('has_pbr', 0)),
            }
        raise ValueError(f'unknown task {self.task}')


def _augment_gct(occ, points):
    rng = np.random.default_rng()
    count = len(points)
    keep = rng.random(count) > rng.uniform(0.02, 0.15)
    if keep.sum() >= 256:
        points = points[keep]
    if len(points) < count:
        extra = rng.integers(0, len(points), count - len(points))
        points = np.concatenate([points, points[extra]], axis=0)
    elif len(points) > count:
        points = points[:count]
    points = points.copy()
    points[:, :3] += rng.normal(0.0, 0.0015, size=(len(points), 3)).astype(np.float32)
    points[:, :3] = np.clip(points[:, :3], -0.5, 0.5)
    occ = occ.copy()
    if rng.random() < 0.5:
        size = int(rng.integers(4, 12))
        start = [int(rng.integers(0, GRID - size)) for _ in range(3)]
        occ[:, start[0]:start[0] + size, start[1]:start[1] + size,
            start[2]:start[2] + size] = 0.0
    return occ, points


class ShardDataset(Dataset):
    """Random-access dataset over an extracted shard directory."""

    def __init__(self, root, records, task='gct'):
        self.root = Path(root)
        self.records = list(records)
        self.task = task

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        data = np.load(self.root / row['path'])
        if self.task == 'gct':
            return {
                'asset_id': row['asset_id'],
                'occ': unpack_bits(data['occ64'])[None].astype(np.float32),
                'target_occ': unpack_bits(data['occ_target64'])[None].astype(np.float32),
                'pts': np.asarray(data['pts'], dtype=np.float32),
                'sdf_coords': np.asarray(data['sdf_coords'], dtype=np.float32),
                'sdf': dequantize_sdf(np.asarray(data['sdf'])),
            }
        raise ValueError(self.task)


def collate_mat(items):
    return collate(items)


def load_manifest(path):
    return json.loads(Path(path).read_text())
