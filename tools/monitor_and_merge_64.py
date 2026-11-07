"""Monitor the 5 conversion shards, download completed outputs, merge into 64³ dataset, package, and launch training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import kaggle


def download_shard(shard_idx):
    slug = f'dileepkumarchakali/t3d-kenney-corpus-64-conversion-shard-{shard_idx}'
    out = Path(f'artifacts/kaggle-kenney-corpus-64-shard{shard_idx}')
    report_file = out / f'kenney-64-shard-{shard_idx}-report.json'
    archive_file = out / f'kenney-64-shard-{shard_idx}.zip'
    if report_file.exists() and archive_file.exists():
        print(f"Shard {shard_idx} already downloaded.", flush=True)
        return True

    status = kaggle.api.kernels_status(slug)
    if 'COMPLETE' not in str(status.status):
        print(f"Shard {shard_idx} status: {status.status}", flush=True)
        return False

    print(f"Downloading Shard {shard_idx}...", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    kaggle.api.kernels_output(slug, path=str(out))
    if report_file.exists() and archive_file.exists():
        print(f"Shard {shard_idx} successfully downloaded.", flush=True)
        return True
    else:
        print(f"Warning: Shard {shard_idx} files missing after download.", flush=True)
        return False


def check_and_download_all():
    all_ready = True
    for i in range(5):
        ready = download_shard(i)
        if not ready:
            all_ready = False
    return all_ready


def merge_and_publish():
    print("\n--- Merging 64³ Corpus Shards ---", flush=True)
    out_dir = Path('artifacts/kenney-corpus-64')
    subprocess.run([
        sys.executable, '-m', 'tools.merge_kenney_64_shards',
        '--shards-dir', 'artifacts',
        '--output', str(out_dir)
    ], cwd=str(Path.cwd()), env=dict(os.environ, PYTHONPATH=str(Path.cwd())), check=True)

    manifest_file = out_dir / 'manifest.json'
    if not manifest_file.exists():
        raise FileNotFoundError("Merged manifest.json not found")

    manifest = json.loads(manifest_file.read_text())
    print(f"Merged manifest contains {len(manifest['assets'])} assets.", flush=True)

    # Package dataset for Kaggle upload
    print("\n--- Packaging 64³ Dataset Archive ---", flush=True)
    kaggle_dir = Path('kaggle-kenney-corpus-64')
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    archive_path = kaggle_dir / 'kenney-corpus-64.zip'

    # Build zip containing manifest.json, merge-report.json, and all .t3d files
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.write(manifest_file, arcname='manifest.json')
        report_file = out_dir / 'merge-report.json'
        if report_file.exists():
            z.write(report_file, arcname='merge-report.json')
        for asset in manifest['assets']:
            asset_path = out_dir / asset['path']
            z.write(asset_path, arcname=asset['path'])

    archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"Built {archive_path} ({archive_size_mb:.2f} MB)", flush=True)

    # Write dataset metadata
    metadata = {
        "title": "T3D Kenney Corpus 64",
        "id": "dileepkumarchakali/t3d-kenney-corpus-64",
        "licenses": [{"name": "CC0-1.0"}]
    }
    (kaggle_dir / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

    # Publish dataset to Kaggle
    print("\n--- Publishing Dataset to Kaggle ---", flush=True)
    try:
        status = kaggle.api.dataset_status('dileepkumarchakali/t3d-kenney-corpus-64')
        print(f"Dataset already exists with status: {status}. Creating new version...", flush=True)
        kaggle.api.dataset_create_version(str(kaggle_dir), version_notes='64³ Kenney Corpus merged deduplicated dataset', dir_mode='zip')
    except Exception as e:
        print(f"Creating new dataset on Kaggle: {e}", flush=True)
        kaggle.api.dataset_create_new(folder=str(kaggle_dir), dir_mode='zip', quiet=False)

    print("Waiting for dataset to reach 'ready' status...", flush=True)
    for _ in range(30):
        try:
            status = kaggle.api.dataset_status('dileepkumarchakali/t3d-kenney-corpus-64')
            print(f"Dataset status: {status}", flush=True)
            if str(status).lower() == 'ready':
                break
        except Exception:
            pass
        time.sleep(5)

    # Launch Kaggle GPU training
    print("\n--- Launching 64³ Corpus SC-VAE Training on Kaggle GPU ---", flush=True)
    train_dir = Path('kaggle-kenney-corpus-64-training')
    push_res = kaggle.api.kernels_push(str(train_dir))
    print(f"Kernel push response: {push_res}", flush=True)
    time.sleep(3)
    k_status = kaggle.api.kernels_status('dileepkumarchakali/transformer-native-3d-kenney-64-corpus-training')
    print(f"Training kernel status: {k_status}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poll', action='store_true', help='Poll in loop until all shards complete')
    parser.add_argument('--interval', type=int, default=30, help='Polling interval in seconds')
    args = parser.parse_args()

    if args.poll:
        print(f"Polling 5 conversion shards every {args.interval}s...", flush=True)
        while True:
            if check_and_download_all():
                print("All 5 shards complete and downloaded!", flush=True)
                merge_and_publish()
                break
            time.sleep(args.interval)
    else:
        if check_and_download_all():
            print("All 5 shards complete and downloaded!", flush=True)
            merge_and_publish()
        else:
            print("Some shards are still running. Re-run or use --poll.", flush=True)


if __name__ == '__main__':
    main()
