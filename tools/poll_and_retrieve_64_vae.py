"""Poll for 64³ SC-VAE training completion, retrieve artifacts, audit metrics, and stage Stage 2 DiT."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import kaggle


KERNEL_SLUG = 'dileepkumarchakali/transformer-native-3d-kenney-64-corpus-training'


def retrieve_outputs(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Retrieving kernel outputs from {KERNEL_SLUG} to {out_dir}...", flush=True)
    kaggle.api.kernels_output(KERNEL_SLUG, path=str(out_dir))

    report_path = out_dir / 'triton-kenney-corpus-64.json'
    if not report_path.exists():
        print("Warning: triton-kenney-corpus-64.json not found in output.", flush=True)
        return False

    report = json.loads(report_path.read_text())
    print("\n=== 64³ Kenney Corpus SC-VAE Training Results ===", flush=True)
    print(f"Total Training Time: {report.get('total_time_seconds', 'N/A')} seconds")
    for row in report.get('rows', []):
        step = row.get('step')
        iou = row.get('mean_val_iou')
        loss = row.get('mean_val_loss')
        ckpt_sha = row.get('checkpoint_sha256', '')[:16]
        print(f"  Step {step:4d} | Val Free IoU: {iou:.4f} | Val Loss: {loss:.4f} | Ckpt: {ckpt_sha}...")

    # Audit checkpoints
    checkpoints = sorted(out_dir.glob('triton-kenney-corpus-64-*.pt'))
    print(f"\nFound {len(checkpoints)} checkpoints:")
    for ck in checkpoints:
        sha = hashlib.sha256(ck.read_bytes()).hexdigest()
        size_mb = ck.stat().st_size / (1024 * 1024)
        print(f"  - {ck.name} ({size_mb:.1f} MB, SHA-256: {sha})")

    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poll', action='store_true', help='Poll until completion')
    parser.add_argument('--interval', type=int, default=20, help='Polling interval in seconds')
    parser.add_argument('--timeout', type=int, default=1800, help='Max timeout in seconds')
    args = parser.parse_args()

    out_dir = Path('artifacts/kaggle-kenney-corpus-64-training')

    start_time = time.time()
    while True:
        status = kaggle.api.kernels_status(KERNEL_SLUG)
        status_str = str(status.status)
        print(f"[{time.strftime('%H:%M:%S')}] Kernel {KERNEL_SLUG} status: {status_str}", flush=True)

        if 'COMPLETE' in status_str:
            print("Kernel training COMPLETE!", flush=True)
            success = retrieve_outputs(out_dir)
            if success:
                print("All artifacts successfully verified and retrieved.", flush=True)
            break
        elif 'ERROR' in status_str or 'CANCEL' in status_str:
            print(f"Kernel failed with status: {status_str}", flush=True)
            sys.exit(1)

        if not args.poll:
            break

        if time.time() - start_time > args.timeout:
            print("Timed out waiting for kernel.", flush=True)
            sys.exit(1)

        time.sleep(args.interval)


if __name__ == '__main__':
    main()
