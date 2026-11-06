"""Build multi-asset textured PBR training and validation dataset with quarter-turn augmentations."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from t3d.augmentation import quarter_turn
from t3d.container import save
from t3d.import_glb import load_glb
from t3d.materials import sample_pbr
from t3d.normalization import normalize_asset
from t3d.voxelize import mesh_to_voxels


MODELS = [
    # Train split (5 diverse objects x 4 rotations = 20 assets)
    dict(id='avocado', path='artifacts/native-inputs/Avocado.glb', split='train', group='avocado', license='CC0-1.0; Microsoft'),
    dict(id='waterbottle', path='artifacts/native-inputs/WaterBottle.glb', split='train', group='waterbottle', license='CC0-1.0; Microsoft'),
    dict(id='fish', path='artifacts/bpt-candidates/BarramundiFish/BarramundiFish.glb', split='train', group='fish', license='CC-BY-4.0; PolyHaven/Khronos'),
    dict(id='duck', path='artifacts/bpt-candidates/Duck/Duck.glb', split='train', group='duck', license='CC0-1.0; Sony/Khronos'),
    dict(id='box', path='artifacts/bpt-candidates/Box/Box.glb', split='train', group='box', license='CC0-1.0; Khronos'),

    # Validation split (2 completely held-out objects x 4 rotations = 8 assets)
    dict(id='lantern', path='artifacts/native-inputs/Lantern.glb', split='validation', group='lantern', license='CC0-1.0; Microsoft'),
    dict(id='boombox', path='artifacts/bpt-candidates/BoomBox/BoomBox.glb', split='validation', group='boombox', license='CC0-1.0; Microsoft')
]


def prepare_rotated_asset(source_path, turns, grid_size=32):
    imported = load_glb(source_path)
    norm_v, _, _ = normalize_asset(imported.vertices)
    norm_v = np.clip(np.asarray(norm_v), -0.5, 0.5)

    rot_v = np.asarray(quarter_turn(norm_v, turns, axis='y'))
    asset = mesh_to_voxels(rot_v, imported.faces, grid_size=(grid_size, grid_size, grid_size), aabb=[[-0.5]*3, [0.5]*3], reject_multiple_crossings=False)
    asset = sample_pbr(asset, rot_v, imported.faces, imported.materials, imported.face_materials, corner_uv=imported.corner_uv)
    return asset, imported.source_sha256


def main():
    out_dir = Path('artifacts/material-multiasset-32')
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    reports = []

    for entry in MODELS:
        src = Path(entry['path'])
        if not src.exists():
            raise FileNotFoundError(f'Model source file not found: {src}')

        for turns in range(4):
            asset_id = f"{entry['id']}-rot{turns*90}"
            out_path = out_dir / f"{asset_id}.t3d"

            asset, src_sha = prepare_rotated_asset(src, turns, grid_size=32)
            if asset.pbr is None or len(asset.pbr) == 0:
                raise ValueError(f'PBR extraction failed for {asset_id}')

            save(out_path, asset)
            content = out_path.read_bytes()
            sha256 = hashlib.sha256(content).hexdigest()

            rec = dict(
                asset_id=asset_id,
                path=out_path.name,
                sha256=sha256,
                split=entry['split'],
                source=f"local:{entry['path']}#rot={turns*90}",
                license=entry['license'],
                source_sha256=src_sha,
                source_group=f"multiasset-{entry['group']}"
            )
            records.append(rec)

            rep = dict(
                asset_id=asset_id,
                group=entry['group'],
                turns=turns,
                degrees=turns * 90,
                split=entry['split'],
                active_voxels=len(asset.coords),
                pbr_min=asset.pbr.min(axis=0).tolist(),
                pbr_max=asset.pbr.max(axis=0).tolist()
            )
            reports.append(rep)
            print(f"Prepared {asset_id}: {len(asset.coords)} voxels, split={entry['split']}", flush=True)

    manifest = dict(version=1, assets=records)
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    summary = dict(
        scope='Multi-asset PBR material corpus with 4-turn axial augmentations; 5 train objects, 2 held-out validation objects',
        total_assets=len(records),
        train_count=sum(1 for r in records if r['split'] == 'train'),
        validation_count=sum(1 for r in records if r['split'] == 'validation'),
        reports=reports
    )
    (out_dir / 'conversion-report.json').write_text(json.dumps(summary, indent=2) + '\n')

    archive = out_dir.parent / (out_dir.name + '.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in [out_dir / 'manifest.json', out_dir / 'conversion-report.json', *(out_dir / r['path'] for r in records)]:
            z.write(path, arcname=path.name)

    print(f"Complete: {len(records)} assets ({summary['train_count']} train, {summary['validation_count']} val) -> {archive}")


if __name__ == '__main__':
    main()
