"""Poll for 64³ Occupancy DiT training completion, retrieve artifacts, and verify metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import kaggle


KERNEL_SLUG = 'dileepkumarchakali/transformer-native-3d-kenney-64-occupancy-dit'


def retrieve_outputs(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Retrieving kernel outputs from {KERNEL_SLUG} to {out_dir}...", flush=True)
    kaggle.api.kernels_output(KERNEL_SLUG, path=str(out_dir))

    report_path = out_dir / 'occupancy-dit-64-report.json'
    if not report_path.exists():
        print("Warning: occupancy-dit-64-report.json not found in output.", flush=True)
        return False

    report = json.loads(report_path.read_text())
    print("\n=== 64³ Occupancy DiT Training Results ===", flush=True)
    print(f"Device: {report.get('device', 'N/A')}")
    print(f"Total Training Time: {report.get('elapsed_seconds', 'N/A')} seconds")
    print(f"Mean Val Occupancy IoU: {report.get('mean_val_occupancy_iou', 'N/A')}")
    print(f"Mean Val Precision: {report.get('mean_val_precision', 'N/A')}")
    print(f"Mean Val Recall: {report.get('mean_val_recall', 'N/A')}")
    print(f"Mean Prompt Difference Cells: {report.get('mean_prompt_difference_cells', 'N/A')}")
    print(f"Checkpoint SHA-256: {report.get('checkpoint_sha256', 'N/A')}")

    # Audit checkpoints
    checkpoints = sorted(out_dir.glob('occupancy-dit-64-*.pt'))
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

    out_dir = Path('artifacts/kaggle-kenney-64-occupancy-dit')

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
