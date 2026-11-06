"""Unified command-line interface for the Transformer-Native 3D (T3D) format."""
import argparse
import sys
from pathlib import Path


def cmd_generate(args):
    from .generate import main as gen_main
    sys.argv = [sys.argv[0]] + args
    gen_main()


def cmd_export(args):
    from .export import main as exp_main
    sys.argv = [sys.argv[0]] + args
    exp_main()


def cmd_migrate(args):
    from .container import migrate_container
    parser = argparse.ArgumentParser(description='Migrate legacy .t3d containers to production v2 sealed format')
    parser.add_argument('input', type=Path, help='Input .t3d file path')
    parser.add_argument('--output', type=Path, default=None, help='Output path (defaults to overwriting input)')
    parsed = parser.parse_args(args)

    data = parsed.input.read_bytes()
    migrated = migrate_container(data)
    out_path = parsed.output or parsed.input
    out_path.write_bytes(migrated)
    print(f"Successfully migrated {parsed.input} -> {out_path} ({len(migrated)} bytes, sealed with SHA-256)")


def cmd_info(args):
    from .container import load
    parser = argparse.ArgumentParser(description='Inspect a .t3d container file')
    parser.add_argument('input', type=Path, help='Input .t3d file path')
    parsed = parser.parse_args(args)

    container = load(parsed.input)
    asset = container.asset
    print(f"File: {parsed.input}")
    print(f"Grid Size: {asset.grid_size}")
    print(f"Bounding Box: [{asset.aabb[0].tolist()} to {asset.aabb[1].tolist()}]")
    print(f"Active Voxels: {len(asset.coords)}")
    print(f"Serialization Curve: {container.curve}")
    print(f"BPT Tokens: {len(container.bpt_tokens)}")
    print(f"Attributes:")
    print(f"  - Dual Vertices: {asset.dual_vertices.shape}")
    print(f"  - Intersected Flags: {asset.intersected_flag.shape}")
    print(f"  - Split Weight: {asset.split_weight is not None}")
    print(f"  - PBR Material: {asset.pbr is not None} ({None if asset.pbr is None else asset.pbr.shape})")
    print(f"  - Skeleton Field: {asset.skeleton_field is not None}")
    print(f"  - Skin Embedding: {asset.skin_embedding is not None}")


def main():
    parser = argparse.ArgumentParser(
        prog='t3d',
        description='Transformer-Native 3D (.t3d) Model Format CLI'
    )
    subparsers = parser.add_subparsers(dest='command', help='Sub-command to execute')

    # Subparsers with REMAINDER to pass through flags
    subparsers.add_parser('generate', help='Generate 3D models from text prompts using flow DiT models')
    subparsers.add_parser('export', help='Export raw T3D assets to production GLB via CUDA baking')
    subparsers.add_parser('migrate', help='Migrate legacy container to production sealed format')
    subparsers.add_parser('info', help='Inspect container metadata and voxel attributes')

    parsed, unknown = parser.parse_known_args()
    if parsed.command is None:
        parser.print_help()
        sys.exit(0)

    if parsed.command == 'generate':
        cmd_generate(unknown)
    elif parsed.command == 'export':
        cmd_export(unknown)
    elif parsed.command == 'migrate':
        cmd_migrate(unknown)
    elif parsed.command == 'info':
        cmd_info(unknown)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
