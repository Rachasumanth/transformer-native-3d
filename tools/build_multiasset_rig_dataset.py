"""Build multi-asset rigged training and validation dataset with quarter-turn augmentations."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import torch

from t3d.articulation import Skeleton, attach_articulation, transfer_skin_weights
from t3d.augmentation import quarter_turn
from t3d.container import dumps
from t3d.dataset import AssetDataset
from t3d.import_rig import load_rig_glb
from t3d.materials import sample_pbr
from t3d.skin_ae import SkinAE
from t3d.voxelize import mesh_to_voxels


RIG_MODELS = [
    # Train split: 5 diverse rigged models x 4 rotations = 20 candidate assets
    dict(id='corgi', path='artifacts/rigged-inputs/Corgi/Corgi.glb', split='train', group='corgi',
         prompt='a 3D model of a corgi dog', license='CC0-1.0; gobkit'),
    dict(id='bat', path='artifacts/rigged-inputs/Bat/Bat.glb', split='train', group='bat',
         prompt='a 3D model of a flying bat', license='CC0-1.0; gobkit'),
    dict(id='shark', path='artifacts/rigged-inputs/Shark/Shark.glb', split='train', group='shark',
         prompt='a 3D model of a shark fish', license='CC0-1.0; gobkit'),
    dict(id='cesiumman', path='artifacts/rigged-inputs/CesiumMan/CesiumMan.glb', split='train', group='cesiumman',
         prompt='a 3D model of a humanoid walking figure', license='CC-BY-4.0; Khronos/Cesium'),
    dict(id='riggedsimple', path='artifacts/rigged-inputs/RiggedSimple/RiggedSimple.glb', split='train', group='riggedsimple',
         prompt='a 3D model of a simple articulated cylinder limb', license='CC0-1.0; Khronos'),

    # Held-out Validation split: 2 completely held-out models x 4 rotations = 8 candidate assets
    dict(id='fox', path='artifacts/rigged-inputs/Fox/Fox.glb', split='validation', group='fox',
         prompt='a 3D model of a red fox animal', license='CC0-1.0; Khronos'),
    dict(id='riggedfigure', path='artifacts/rigged-candidates/RiggedFigure/RiggedFigure.glb', split='validation', group='riggedfigure',
         prompt='a 3D model of an articulated humanoid figure', license='CC-BY-4.0; Khronos/Cesium')
]


def prepare_rotated_rig_asset(source_path, turns, skinae_model, grid_size=32):
    rig, _ = load_rig_glb(source_path, include_pbr=True).normalized()
    raw_v = np.clip(rig.vertices, -0.5, 0.5)

    rot_v = np.asarray(quarter_turn(raw_v, turns, axis='y'))
    rot_joints = np.asarray(quarter_turn(rig.skeleton.joints, turns, axis='y'))
    rot_skeleton = Skeleton(rot_joints, rig.skeleton.parents)

    geom = mesh_to_voxels(rot_v, rig.faces, grid_size=(grid_size, grid_size, grid_size),
                          aabb=[[-0.5]*3, [0.5]*3], reject_multiple_crossings=False)
    centers = geom.aabb[0] + (geom.coords.astype(float) + 0.5) * geom.voxel_size
    transferred = transfer_skin_weights(centers, rot_v, rig.faces, rig.weights)

    with torch.no_grad():
        j_tensor = torch.as_tensor(rot_skeleton.joints, dtype=torch.float32)
        p_tensor = torch.as_tensor(rot_skeleton.parents, dtype=torch.int64)
        je = skinae_model.encode_joints(j_tensor, p_tensor)

    asset = attach_articulation(geom, rot_skeleton, voxel_weights=transferred,
                                joint_embeddings=je.cpu().numpy())

    if len(rig.pbr_materials):
        asset = sample_pbr(asset, rot_v, rig.faces, rig.pbr_materials, rig.face_materials, corner_uv=rig.corner_uv)
    else:
        from dataclasses import replace
        default_pbr = np.tile([0.5, 0.5, 0.5, 0.0, 0.5, 1.0], (len(asset.coords), 1)).astype(np.float32)
        asset = replace(asset, pbr=default_pbr)

    source_sha = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    return asset, rot_skeleton, source_sha


def main():
    out_dir = Path('artifacts/rig-multiasset-32')
    out_dir.mkdir(parents=True, exist_ok=True)

    skinae_ckpt_path = Path('artifacts/kaggle-skin-family-bind-select-v8/loss-1.0/skin-pose-pilot.pt')
    if not skinae_ckpt_path.exists():
        raise FileNotFoundError(f"SkinAE checkpoint not found: {skinae_ckpt_path}")
    skinae_ckpt = torch.load(skinae_ckpt_path, map_location='cpu', weights_only=True)
    skinae = SkinAE(**skinae_ckpt['config'])
    skinae.load_state_dict(skinae_ckpt['model'])
    skinae.freeze()
    skinae_sha = hashlib.sha256(skinae_ckpt_path.read_bytes()).hexdigest()

    records = []
    reports = []
    duplicates = []
    seen_hashes = {}
    prompts_map = {}

    for entry in RIG_MODELS:
        src = Path(entry['path'])
        if not src.exists():
            raise FileNotFoundError(f"Model source file not found: {src}")

        for turns in range(4):
            asset_id = f"{entry['id']}-rot{turns*90}"
            out_path = out_dir / f"{asset_id}.t3d"

            asset, rot_skeleton, src_sha = prepare_rotated_rig_asset(src, turns, skinae, grid_size=32)
            if asset.skeleton_field is None or asset.skin_embedding is None:
                raise ValueError(f"Rig field extraction failed for {asset_id}")

            raw_bytes = dumps(asset)
            asset_sha = hashlib.sha256(raw_bytes).hexdigest()

            if asset_sha in seen_hashes:
                duplicates.append({
                    'asset_id': asset_id,
                    'duplicate_of': seen_hashes[asset_sha],
                    'sha256': asset_sha,
                    'split': entry['split']
                })
                print(f"Skipping exact duplicate: {asset_id} == {seen_hashes[asset_sha]}")
                continue

            seen_hashes[asset_sha] = asset_id
            out_path.write_bytes(raw_bytes)

            prompt_text = entry['prompt']
            if turns > 0:
                prompt_text = f"{prompt_text}, turned {turns*90} degrees"
            prompts_map[asset_id] = prompt_text

            rec = dict(
                asset_id=asset_id,
                path=f"{asset_id}.t3d",
                sha256=asset_sha,
                split=entry['split'],
                source=f"glTF-Rigged / {entry['id']}",
                license=entry['license'],
                source_sha256=src_sha,
                source_group=entry['group']
            )
            records.append(rec)

            reports.append(dict(
                asset_id=asset_id,
                split=entry['split'],
                group=entry['group'],
                rotation_degrees=turns * 90,
                voxels=len(asset.coords),
                joint_count=len(rot_skeleton.joints),
                skeleton_field_shape=list(asset.skeleton_field.shape),
                skin_embedding_shape=list(asset.skin_embedding.shape),
                pbr_shape=list(asset.pbr.shape) if asset.pbr is not None else None,
                sha256=asset_sha
            ))
            print(f"Generated {asset_id:24s} [{entry['split']:10s}]: {len(asset.coords):4d} voxels, {len(rot_skeleton.joints):2d} joints, SHA-256: {asset_sha[:12]}...")

    manifest = dict(
        version=1,
        skinae_checkpoint_sha256=skinae_sha,
        assets=records
    )
    manifest_path = out_dir / 'manifest.json'
    manifest_bytes = json.dumps(manifest, indent=2).encode('utf-8')
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    prompts_path = out_dir / 'prompts.json'
    prompts_path.write_text(json.dumps(prompts_map, indent=2) + '\n')

    report_path = out_dir / 'merge-report.json'
    report = dict(
        scope='Curated multi-asset 32^3 rigged dataset with 6-channel skeleton displacement and 4-channel skinning latents',
        skinae_checkpoint_sha256=skinae_sha,
        manifest_sha256=manifest_sha,
        total_assets=len(records),
        train_assets=sum(1 for r in records if r['split'] == 'train'),
        validation_assets=sum(1 for r in records if r['split'] == 'validation'),
        duplicates_removed=len(duplicates),
        duplicates=duplicates,
        assets=reports
    )
    report_path.write_text(json.dumps(report, indent=2) + '\n')

    # Verify dataset with AssetDataset
    train_ds = AssetDataset(manifest_path, split='train')
    val_ds = AssetDataset(manifest_path, split='validation')
    print(f"\nVerification passed! Loaded {len(train_ds)} train assets and {len(val_ds)} validation assets.")

    # Create zip bundle for Kaggle dataset
    zip_dir = Path('kaggle-rig-multiasset-32-dataset')
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / 'rig-multiasset-32.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, arcname='manifest.json')
        zf.write(report_path, arcname='merge-report.json')
        zf.write(prompts_path, arcname='prompts.json')
        for rec in records:
            asset_file = out_dir / rec['path']
            zf.write(asset_file, arcname=rec['path'])

    meta = {
        "title": "T3D Rig Multi-Asset 32",
        "id": "dileepkumarchakali/t3d-rig-multiasset-32",
        "licenses": [{"name": "CC0-1.0"}]
    }
    (zip_dir / 'dataset-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')

    print(f"Dataset bundle created: {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == '__main__':
    main()
