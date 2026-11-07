"""Deployment diagnostics for the T3D runtime.

``doctor`` performs cheap, offline checks that catch the most common production
failures before a process starts serving traffic: unsupported Python, missing
dependencies, missing or corrupt checkpoints, an unusable output directory, and
an unavailable accelerator when one is required.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from .checkpoints import CheckpointError, release_report, repo_root, resolve_checkpoint, sha256_file

REQUIRED_64 = ('shape_vae_64', 'rig_flow', 'material_flow')
OPTIONAL = ('material_vae', 'geometry_64', 'occupancy_64', 'skin_ae', 'unified_64')


def _check(name: str, ok: bool, detail: str, *, required: bool = True) -> dict:
    return {'name': name, 'ok': bool(ok), 'required': required, 'detail': detail}


def _dependency_versions() -> dict:
    versions = {'python': sys.version.split()[0]}
    for module in ('torch', 'numpy', 'trimesh', 'transformers'):
        try:
            loaded = __import__(module)
            versions[module] = getattr(loaded, '__version__', 'unknown')
        except Exception as exc:  # pragma: no cover - environment dependent
            versions[module] = f'unavailable: {exc.__class__.__name__}'
    return versions


def run_doctor(*, artifacts: Path | str | None = None, require_gpu: bool = False,
               check_output: Path | str | None = None, verify: bool = True,
               asset: Path | str | None = None, max_tris: int = 15000,
               require_rig: bool = False) -> dict:
    checks: list[dict] = []
    warnings: list[str] = []

    if asset is not None:
        from .asset_validator import validate_production_asset
        asset_path = Path(asset)
        if not asset_path.exists():
            checks.append(_check('asset_exists', False, f'Asset file not found: {asset_path}'))
            return {
                'ok': False,
                'repo_root': str(repo_root()),
                'checks': checks,
                'warnings': warnings,
            }
        asset_bytes = asset_path.read_bytes()
        asset_report = validate_production_asset(asset_bytes, max_tris=max_tris, require_rig=require_rig)
        for gate_key, gate in asset_report['gates'].items():
            checks.append(_check(gate['name'], gate['ok'], gate['detail'], required=True))
        return {
            'ok': asset_report['ok'],
            'repo_root': str(repo_root()),
            'asset': str(asset_path),
            'asset_report': asset_report,
            'checks': checks,
            'warnings': warnings,
        }

    major, minor = sys.version_info[:2]
    checks.append(_check(
        'python', (major, minor) >= (3, 10),
        f'Python {major}.{minor} (requires >=3.10; 3.12 recommended)',
    ))

    versions = _dependency_versions()
    for module in ('torch', 'numpy', 'trimesh', 'transformers'):
        value = versions.get(module, '')
        checks.append(_check(f'dependency:{module}', not str(value).startswith('unavailable'), str(value)))

    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        detail = torch.cuda.get_device_name(0) if cuda else 'CPU fallback active'
        checks.append(_check('accelerator', cuda or not require_gpu, detail, required=require_gpu))
        if not cuda:
            warnings.append('CUDA is unavailable; generation will use the slower CPU backend')
    except Exception as exc:  # pragma: no cover
        checks.append(_check('accelerator', False, f'torch unavailable: {exc}', required=require_gpu))

    root = Path(artifacts) if artifacts is not None else None
    try:
        report = release_report(root=root)
    except Exception as exc:
        report = {'roles': {}, 'present_roles': 0, 'total_roles': 0}
        checks.append(_check('checkpoint_manifest', False, f'cannot read checkpoint registry: {exc}'))
    else:
        checks.append(_check(
            'checkpoint_manifest', report.get('manifest_entries', 0) > 0,
            f"{report.get('present_roles', 0)}/{report.get('total_roles', 0)} roles discoverable",
        ))

    from .checkpoints import manifest_entries
    entries = manifest_entries()
    for role in (*REQUIRED_64, 'unified_64'):
        found = report.get('roles', {}).get(role, {})
        path = found.get('path')
        if not path:
            if role == 'unified_64':
                continue
            checks.append(_check(f'checkpoint:{role}', False, 'not found'))
            continue
        expected = entries.get(Path(path).name)
        if expected is None:
            checks.append(_check(f'checkpoint:{role}', False, f'{path} is not listed in the manifest'))
            continue
        if verify:
            actual = sha256_file(Path(path))
            checks.append(_check(
                f'checkpoint:{role}', actual == expected['sha256'],
                f'{Path(path).name} sha256={"ok" if actual == expected["sha256"] else actual}',
            ))
        else:
            checks.append(_check(f'checkpoint:{role}', True, f'{Path(path).name} (hash not checked)'))
    for role in OPTIONAL:
        found = report.get('roles', {}).get(role, {})
        if not found.get('path'):
            warnings.append(f'optional checkpoint {role!r} is not available')

    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        checks.append(_check('service_dependencies', True, 'fastapi and uvicorn importable'))
    except Exception as exc:
        checks.append(_check('service_dependencies', False, f'service extras missing: {exc}', required=False))
        warnings.append('Install the service extras with: pip install ".[service]"')

    target = Path(check_output) if check_output is not None else Path(tempfile.gettempdir())
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / '.t3d-doctor-write-test'
        probe.write_text('ok')
        probe.unlink()
        checks.append(_check('output_writable', True, f'{target} is writable'))
    except Exception as exc:
        checks.append(_check('output_writable', False, f'{target} is not writable: {exc}'))

    free = shutil.disk_usage(target).free if target.exists() else 0
    if free and free < 1 << 30:
        warnings.append(f'low disk space at {target}: {free / (1 << 30):.2f} GiB free')

    required_ok = all(item['ok'] for item in checks if item['required'])
    return {
        'ok': required_ok,
        'repo_root': str(repo_root()),
        'artifacts_root': str(Path(artifacts) if artifacts is not None else report.get('artifacts_root', '')),
        'versions': versions,
        'checks': checks,
        'warnings': warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='t3d doctor', description='Check T3D deployment prerequisites or validate production 3D assets')
    parser.add_argument('--artifacts', type=Path, default=None, help='Artifacts root override (default: T3D_ARTIFACTS or repo artifacts/)')
    parser.add_argument('--output', type=Path, default=None, help='Directory to probe for write access')
    parser.add_argument('--require-gpu', action='store_true', help='Fail when CUDA is unavailable')
    parser.add_argument('--no-verify', action='store_true', help='Skip checkpoint SHA-256 verification')
    parser.add_argument('--asset', type=Path, default=None, help='Path to GLB asset to validate against the 10 production gates')
    parser.add_argument('--max-tris', type=int, default=15000, help='Maximum triangles for asset validation (default: 15000)')
    parser.add_argument('--require-rig', action='store_true', help='Enforce skinning and skeleton validation (fail if static mesh)')
    parser.add_argument('--json', action='store_true', help='Emit machine-readable JSON')
    args = parser.parse_args(argv)

    if args.asset is not None:
        report = run_doctor(asset=args.asset, max_tris=args.max_tris, require_rig=args.require_rig)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"Asset verification: {args.asset}")
            for item in report['checks']:
                status = 'PASS' if item['ok'] else 'FAIL'
                print(f'[{status}] {item["name"]}: {item["detail"]}')
            for warning in report['warnings']:
                print(f'[WARN] {warning}')
            print('PASS' if report['ok'] else 'FAIL')
        return 0 if report['ok'] else 1

    report = run_doctor(artifacts=args.artifacts, require_gpu=args.require_gpu,
                        check_output=args.output, verify=not args.no_verify)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for item in report['checks']:
            status = 'PASS' if item['ok'] else ('FAIL' if item['required'] else 'WARN')
            print(f'[{status}] {item["name"]}: {item["detail"]}')
        for warning in report['warnings']:
            print(f'[WARN] {warning}')
        print('READY' if report['ok'] else 'NOT READY')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())

