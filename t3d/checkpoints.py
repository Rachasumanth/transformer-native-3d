"""Manifest-backed checkpoint discovery, verification, and release diagnostics.

The package does not ship model binaries. This module provides a stable logical
name -> verified file mapping so inference, benchmarks, and release checks do not
hard-code experiment directory names.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

MANIFEST_NAME = 'checkpoints_manifest.json'

# Preferred order per logical role. The first existing file wins. Explicit
# paths always override discovery.
CANDIDATES = {
    'occupancy_64': (
        'occupancy-dit-64-20000.pt',
        'occupancy-dit-64-10000.pt',
        'occupancy-dit-64-4000.pt',
        'occupancy-dit-64-3000.pt',
        'occupancy-dit-64-2000.pt',
        'occupancy-dit-64-1500.pt',
        'occupancy-dit-64-1000.pt',
        'occupancy-dit-64-500.pt',
    ),
    'occupancy_32': ('occupancy-dit-600.pt',),
    'geometry_64': (
        'geometry-dit-64-20000.pt',
        'geometry-dit-64-10000.pt',
        'geometry-dit-64-4000.pt',
        'geometry-dit-64-3000.pt',
        'geometry-dit-64-2000.pt',
        'geometry-dit-64-1000.pt',
        'geometry-dit-64-4000-v5.pt',
        'geometry-dit-64-600.pt',
    ),
    'geometry_32': ('geometry-dit-600.pt',),
    'shape_vae_64': ('triton-kenney-corpus-64-1200.pt',),
    'shape_vae_32': ('triton-kenney-corpus-1900.pt',),
    'material_flow': ('material-flow-dit-300.pt', 'material-flow-dit-600.pt'),
    'material_vae': ('material-multiasset-300.pt',),
    'rig_flow': (
        'rig-flow-dit-2500.pt',
        'rig-flow-dit-2000.pt',
        'rig-flow-dit-1500.pt',
        'rig-flow-dit-1000.pt',
        'rig-flow-dit-600.pt',
        'rig-flow-dit-250.pt',
    ),
    'skin_ae': ('skin-pose-pilot.pt',),
    'unified_64': (
        'unified-dit-64-20000.pt',
        'unified-dit-64-15000.pt',
        'unified-dit-64-12000.pt',
        'unified-dit-64-10000.pt',
        'unified-dit-64-5000.pt',
        'unified-dit-64-2500.pt',
        'unified-dit-64-5.pt',
    ),
}

# Search order matters: newer verified experiment directories take precedence.
_SEARCH_DIRS = (
    'kaggle-unified-dit-v1',
    'kaggle-kenney-64-unified-dit',
    'kaggle-kenney-64-expand-train',
    'kaggle-kenney-64-occupancy-dit-v4',
    'kaggle-kenney-64-occupancy-dit',
    'kaggle-kenney-64-geometry-dit-v6',
    'kaggle-kenney-64-geometry-dit',
    'kaggle-kenney-corpus-64-training',
    'kaggle-material-flow-dit-v1',
    'kaggle-material-multiasset-32-v1',
    'kaggle-rig-flow-dit-v4',
    'kaggle-rig-flow-dit-v3',
    'kaggle-rig-flow-dit-v2',
    'kaggle-skin-family-bind-select-v8/loss-1.0',
    'kaggle-skin-pose-v1',
    'kaggle-kenney-corpus-training-v35',
    'kaggle-kenney-occupancy-dit-v37',
    'kaggle-kenney-geometry-dit-v38',
)


class CheckpointError(FileNotFoundError):
    """Raised when a required checkpoint cannot be found or verified."""


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def artifacts_root(root: Path | str | None = None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get('T3D_ARTIFACTS')
    return Path(env) if env else repo_root() / 'artifacts'


def load_manifest(path: Path | str | None = None) -> dict:
    manifest = Path(path) if path is not None else repo_root() / MANIFEST_NAME
    if not manifest.exists():
        return {'checkpoints': []}
    return json.loads(manifest.read_text())


def manifest_entries(path: Path | str | None = None) -> dict[str, dict]:
    return {entry['name']: entry for entry in load_manifest(path).get('checkpoints', [])}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def candidates_for(kind: str, resolution: int | None = None) -> tuple[str, ...]:
    if kind in ('occupancy', 'geometry', 'shape_vae'):
        if resolution not in (32, 64):
            raise ValueError(f'resolution must be 32 or 64 for {kind!r}')
        kind = f'{kind}_{resolution}'
    try:
        return CANDIDATES[kind]
    except KeyError as exc:
        raise KeyError(f'Unknown checkpoint role {kind!r}; known roles: {sorted(CANDIDATES)}') from exc


def resolve_checkpoint(
    kind: str,
    *,
    resolution: int | None = None,
    path: Path | str | None = None,
    root: Path | str | None = None,
    verify: bool | None = None,
    manifest_path: Path | str | None = None,
) -> Path:
    """Resolve a logical checkpoint role to an existing, optionally verified file.

    ``verify`` defaults to the ``T3D_VERIFY_CHECKPOINTS`` environment variable
    (enabled unless set to 0/false). Unknown files are accepted when explicitly
    passed; discovered files are only accepted from the manifest.
    """
    if verify is None:
        verify = os.environ.get('T3D_VERIFY_CHECKPOINTS', '1').lower() not in ('0', 'false', 'no')
    entries = manifest_entries(manifest_path)

    if path is not None:
        chosen = Path(path)
        if not chosen.exists():
            raise CheckpointError(f'Checkpoint for {kind!r} not found at explicit path: {chosen}')
        expected = entries.get(chosen.name)
        if verify and expected is not None:
            actual = sha256_file(chosen)
            if actual != expected['sha256']:
                raise CheckpointError(
                    f'Checkpoint SHA-256 mismatch for {chosen}: {actual} != manifest {expected["sha256"]}'
                )
        return chosen

    names = candidates_for(kind, resolution)
    base = artifacts_root(root)
    searched: list[Path] = []
    for name in names:
        for directory in _SEARCH_DIRS:
            candidate = base / directory / name
            searched.append(candidate)
            if candidate.is_file():
                expected = entries.get(name)
                if expected is None:
                    continue
                if verify:
                    actual = sha256_file(candidate)
                    if actual != expected['sha256']:
                        raise CheckpointError(
                            f'Checkpoint SHA-256 mismatch for {candidate}: {actual} != manifest {expected["sha256"]}'
                        )
                return candidate
        # Fall back to a bounded recursive search for renamed experiment dirs.
        for candidate in base.glob(f'*/{name}'):
            if candidate.is_file() and name in entries:
                searched.append(candidate)
                if verify and sha256_file(candidate) != entries[name]['sha256']:
                    raise CheckpointError(f'Checkpoint SHA-256 mismatch for {candidate}')
                return candidate

    raise CheckpointError(
        f'No verified checkpoint found for role {kind!r} (files tried: {", ".join(names)}). '
        f'Searched under {base}. Set T3D_ARTIFACTS or pass an explicit path.'
    )


def resolve_optional(kind: str, **kwargs) -> Path | None:
    try:
        return resolve_checkpoint(kind, **kwargs)
    except CheckpointError:
        return None


def release_report(root: Path | str | None = None, manifest_path: Path | str | None = None) -> dict:
    """Return a machine-readable inventory used by doctor/release checks."""
    entries = manifest_entries(manifest_path)
    roles = {}
    for kind in CANDIDATES:
        resolution = None
        if kind.endswith('_64'):
            resolution = 64
        elif kind.endswith('_32'):
            resolution = 32
        try:
            found = resolve_checkpoint(kind, resolution=resolution, root=root, manifest_path=manifest_path, verify=False)
            roles[kind] = {'path': str(found), 'present': True, 'verified': found.name in entries}
        except CheckpointError:
            roles[kind] = {'path': None, 'present': False, 'verified': False}
    present = sum(1 for role in roles.values() if role['present'])
    return {
        'artifacts_root': str(artifacts_root(root)),
        'manifest_entries': len(entries),
        'roles': roles,
        'present_roles': present,
        'total_roles': len(roles),
    }
