"""Versioned, provenance-bearing manifests for prepared .t3d training assets."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from .container import loads


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    path: str
    sha256: str
    split: str
    source: str
    license: str
    source_sha256: str | None = None
    source_group: str | None = None


class AssetDataset:
    def __init__(self, manifest, *, split, require_pbr=False):
        self.manifest = Path(manifest).resolve()
        document = json.loads(self.manifest.read_text())
        if document.get('version') != 1 or not isinstance(document.get('assets'), list):
            raise ValueError('Unsupported dataset manifest')
        if split not in ('train', 'validation', 'test'):
            raise ValueError('Invalid dataset split')
        self.require_pbr = require_pbr
        self.fingerprint = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        records, ids, hashes, paths = [], set(), set(), set()
        source_splits, group_splits = {}, {}
        for value in document['assets']:
            record = AssetRecord(**value)
            if (not all(isinstance(getattr(record, key), str) and getattr(record, key).strip()
                        for key in ('asset_id', 'path', 'sha256', 'split', 'source', 'license'))
                    or record.split not in ('train', 'validation', 'test')
                    or not re.fullmatch('[0-9a-f]{64}', record.sha256)):
                raise ValueError('Invalid record metadata')
            if (record.source_sha256 is not None and
                    (not isinstance(record.source_sha256, str) or
                     not re.fullmatch('[0-9a-f]{64}', record.source_sha256))):
                raise ValueError('Invalid source content hash')
            if (record.source_group is not None and
                    (not isinstance(record.source_group, str) or not record.source_group.strip())):
                raise ValueError('Invalid source group')
            for identity, seen in ((record.source_sha256, source_splits),
                                   (record.source_group, group_splits)):
                if identity is not None:
                    if identity in seen and seen[identity] != record.split:
                        raise ValueError('Source identity crosses dataset splits')
                    seen[identity] = record.split
            path = (self.manifest.parent/record.path).resolve()
            if Path(record.path).is_absolute() or not path.is_relative_to(self.manifest.parent):
                raise ValueError('Asset path must stay within the manifest directory')
            if record.asset_id in ids or record.sha256 in hashes or path in paths:
                raise ValueError('Duplicate asset ID, content or path; split leakage is forbidden')
            ids.add(record.asset_id); hashes.add(record.sha256); paths.add(path)
            if record.split == split:
                records.append(record)
        self.records = tuple(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        data = (self.manifest.parent/record.path).read_bytes()
        if hashlib.sha256(data).hexdigest() != record.sha256:
            raise ValueError(f'Content hash mismatch for {record.asset_id}')
        asset = loads(data).asset
        if not len(asset.coords) or (self.require_pbr and asset.pbr is None):
            raise ValueError('Training assets need active voxels and requested PBR attributes')
        return asset


def batch_arrays(assets, mode):
    """Build batch-indexed sparse arrays; never mix voxels across assets."""
    if not assets or mode not in ('shape', 'material'):
        raise ValueError('Batch needs assets and shape/material mode')
    coords, features = [], []
    for i, asset in enumerate(assets):
        if not len(asset.coords):
            raise ValueError('Empty training asset')
        coords.append(np.column_stack((np.full(len(asset.coords), i, dtype=np.int64), asset.coords)))
        if mode == 'shape':
            split = asset.split_weight if asset.split_weight is not None else np.zeros((len(asset.coords), 1))
            features.append(np.column_stack((asset.dual_vertices, asset.intersected_flag, split)))
        else:
            if asset.pbr is None:
                raise ValueError('Material training requires PBR on every asset')
            features.append(asset.pbr)
    return np.concatenate(coords).astype(np.int64), np.concatenate(features).astype(np.float32)
