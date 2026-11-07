"""Retrieve and verify Kaggle Version 6 geometry DiT outputs and evaluate on VM."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    kernel = 'dileepkumarchakali/transformer-native-3d-kenney-64-geometry-dit'
    api = KaggleApi()
    api.authenticate()
    
    status_info = api.kernels_status(kernel)
    status_str = str(getattr(status_info, 'status', ''))
    failure_msg = getattr(status_info, 'failure_message', '') or getattr(status_info, 'failureMessage', '')
    print(f"Kernel: {kernel}")
    print(f"Status: {status_str}")
    if failure_msg:
        print(f"Failure message: {failure_msg}")
        return False
        
    if 'COMPLETE' not in status_str:
        print(f"Kernel is not complete yet (status={status_str}). Try again shortly.")
        return False
        
    out_dir = Path('artifacts/kaggle-kenney-64-geometry-dit-v6')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading kernel output to {out_dir}...")
    api.kernels_output(kernel, str(out_dir))
    
    files = list(out_dir.glob('*'))
    print(f"Retrieved {len(files)} files:")
    for f in sorted(files):
        sz_kb = f.stat().st_size / 1024.0
        digest = compute_sha256(f) if f.is_file() else ''
        print(f"  - {f.name} ({sz_kb:.1f} KB, SHA-256: {digest[:16]}...)")
        
    # Check for report and final checkpoint
    report_file = out_dir / 'geometry-dit-64-report.json'
    ckpt_4000 = out_dir / 'geometry-dit-64-4000.pt'
    
    if report_file.exists():
        with open(report_file) as f:
            report = json.load(f)
        print("\n--- Geometry DiT v6 Report Summary ---")
        print(f"Scope: {report.get('scope')}")
        print(f"Elapsed seconds: {report.get('elapsed_seconds'):.1f}")
        print(f"Total steps: {report.get('total_steps')}")
        print(f"Mean Val Latent MSE: {report.get('mean_val_latent_mse')}")
        print(f"Mean Steerability Delta: {report.get('mean_steerability_delta')}")
        print(f"Mean Mesh Triangles: {report.get('mean_mesh_triangles')}")
        print(f"Mean Surface Distance: {report.get('mean_surface_distance')}")
        print(f"Mean F-Score (0.01 tol): {report.get('mean_fscore_001')}")
        
    if ckpt_4000.exists():
        target_dir = Path('artifacts/kaggle-kenney-64-geometry-dit')
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / 'geometry-dit-64-4000.pt'
        shutil.copy2(ckpt_4000, target_path)
        print(f"\nSynchronized active checkpoint to {target_path}")
        
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
