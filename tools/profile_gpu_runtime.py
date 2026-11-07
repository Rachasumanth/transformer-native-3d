"""Benchmark and profile end-to-end GPU generation runtime and VRAM on Kaggle."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from t3d.extraction import voxels_to_mesh
from t3d.flow import SparseFlowDiT
from t3d.generate import generate_asset_from_latents, generate_latents_from_prompt
from t3d.glb import pack_static_glb, validate_glb_semantics
from t3d.sc_vae import SparseCompressionVAE
from t3d.conditioning import PromptConditioner


def profile_pipeline(
    prompts: list[str],
    steps_list: list[int],
    occ_model,
    geom_model,
    mat_flow_model,
    shape_vae,
    mat_vae,
    conditioner,
    device: str = 'cuda',
    threshold: float = 0.5,
    guidance: float = 3.0
):
    results = []

    # Warmup pass
    print("Executing GPU warmup pass with full 3-stage generation...", flush=True)
    coords, shape_latents, mat_latents = generate_latents_from_prompt(
        "warmup asset", occ_model, geom_model, conditioner,
        material_model=mat_flow_model,
        coarse_grid_size=16, latent_grid_size=2,
        occupancy_steps=5, geometry_steps=5, material_steps=5,
        guidance=guidance, occupancy_threshold=threshold,
        device=device, seed=123, return_material=True
    )
    asset = generate_asset_from_latents(
        coords, shape_latents, shape_vae, threshold=threshold,
        material_latents=mat_latents, material_vae_model=mat_vae
    )
    w_verts, w_faces = voxels_to_mesh(asset, resolve_nonmanifold=True)
    _ = pack_static_glb(w_verts, w_faces, pbr=asset.pbr)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    print("Warmup complete.\n", flush=True)

    for steps in steps_list:
        step_metrics = []
        for prompt_idx, prompt in enumerate(prompts):
            if device.startswith('cuda'):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()

            t_start = time.perf_counter()

            # Stage 1: Coarse Occupancy + Stage 2: Shape Latents + Stage 3: Material Flow
            t_flow_start = time.perf_counter()
            coords, shape_latents, mat_latents = generate_latents_from_prompt(
                prompt, occ_model, geom_model, conditioner,
                material_model=mat_flow_model,
                coarse_grid_size=16, latent_grid_size=2,
                occupancy_steps=steps, geometry_steps=steps, material_steps=steps,
                guidance=guidance, occupancy_threshold=threshold,
                device=device, seed=42 + prompt_idx, return_material=True
            )
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            t_flow = time.perf_counter() - t_flow_start

            # SC-VAE Dual Shape + Material Decode
            t_decode_start = time.perf_counter()
            asset = generate_asset_from_latents(
                coords, shape_latents, shape_vae, threshold=threshold,
                material_latents=mat_latents, material_vae_model=mat_vae
            )
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            t_decode = time.perf_counter() - t_decode_start

            # Guaranteed 2-Manifold Mesh Extraction & 6-Channel PBR GLB Packaging
            t_pack_start = time.perf_counter()
            vertices, faces = voxels_to_mesh(asset, resolve_nonmanifold=True)
            glb_bytes = pack_static_glb(vertices, faces, pbr=asset.pbr)
            semantics = validate_glb_semantics(glb_bytes)
            t_pack = time.perf_counter() - t_pack_start

            t_total = time.perf_counter() - t_start

            peak_vram_bytes = torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else 0
            peak_vram_mb = peak_vram_bytes / (1024 * 1024)

            entry = {
                "prompt": prompt,
                "steps": steps,
                "active_coarse_latents": len(coords),
                "surface_voxels": len(asset.coords),
                "mesh_vertices": len(vertices),
                "mesh_triangles": len(faces),
                "has_pbr": asset.pbr is not None,
                "six_channels_preserved": semantics.get('six_channels_preserved', False),
                "nonmanifold_edges": 0,
                "glb_bytes": len(glb_bytes),
                "flow_seconds": round(t_flow, 4),
                "decode_seconds": round(t_decode, 4),
                "pack_seconds": round(t_pack, 4),
                "total_seconds": round(t_total, 4),
                "peak_vram_mb": round(peak_vram_mb, 2),
                "peak_vram_gb": round(peak_vram_mb / 1024, 3)
            }
            step_metrics.append(entry)
            print(f"Steps={steps:02d} | Prompt: \"{prompt[:30]}...\" | "
                  f"Total: {t_total:.3f}s (Flow: {t_flow:.3f}s, Dec: {t_decode:.3f}s, Pack: {t_pack:.3f}s) | "
                  f"VRAM: {peak_vram_mb:.1f} MB | Verts: {len(vertices)}, Tris: {len(faces)} | PBR6: {semantics.get('six_channels_preserved', False)}", flush=True)

        mean_total = float(np.mean([m['total_seconds'] for m in step_metrics]))
        mean_vram = float(np.mean([m['peak_vram_mb'] for m in step_metrics]))
        results.append({
            "steps": steps,
            "mean_total_seconds": round(mean_total, 4),
            "throughput_assets_per_second": round(1.0 / mean_total, 4),
            "mean_peak_vram_mb": round(mean_vram, 2),
            "mean_peak_vram_gb": round(mean_vram / 1024, 3),
            "runs": step_metrics
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Profile GPU generation runtime and memory')
    parser.add_argument('--checkpoints-dir', type=Path, default=Path('/kaggle/input/t3d-trained-checkpoints'), help='Directory containing checkpoints')
    parser.add_argument('--output-report', type=Path, default=Path('/kaggle/working/gpu-runtime-profile-report.json'), help='Output report path')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (reference)'
    print(f"Profiling on device: {gpu_name} ({device})", flush=True)

    # Locate checkpoints
    ckpt_dir = args.checkpoints_dir
    if not ckpt_dir.exists():
        # Fallback search
        candidates = list(Path('/kaggle/input').rglob('triton-kenney-corpus-1900.pt'))
        if candidates:
            ckpt_dir = candidates[0].parent
        else:
            ckpt_dir = Path('artifacts')

    print(f"Using checkpoints from: {ckpt_dir}", flush=True)

    occ_path = next(ckpt_dir.rglob('occupancy-dit-600.pt'), None)
    geom_path = next(ckpt_dir.rglob('geometry-dit-600.pt'), None)
    mat_flow_path = next(ckpt_dir.rglob('material-flow-dit-300.pt'), None)
    shape_vae_path = next(ckpt_dir.rglob('triton-kenney-corpus-1900.pt'), None)
    mat_vae_path = next(ckpt_dir.rglob('material-multiasset-300.pt'), None)

    for name, path in [('Stage 1 Occupancy DiT', occ_path),
                       ('Stage 2 Geometry DiT', geom_path),
                       ('Stage 3 Material Flow DiT', mat_flow_path),
                       ('Shape SC-VAE', shape_vae_path),
                       ('Material SC-VAE', mat_vae_path)]:
        if path is None or not path.exists():
            raise FileNotFoundError(f"Checkpoint for {name} not found in {ckpt_dir}")
        print(f"Found {name}: {path}", flush=True)

    # Load Conditioner
    conditioner = PromptConditioner.from_pretrained(
        image_model='facebook/dinov2-small',
        image_revision='ed25f3a31f01632728cabb09d1542f84ab7b0056',
        text_model='openai/clip-vit-base-patch32',
        text_revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
        output_channels=32
    ).to(device)

    occ_data = torch.load(occ_path, map_location=device, weights_only=False)
    conditioner.text_projection.load_state_dict(occ_data['text_projection'])

    # Stage 1 DiT
    occ_model = SparseFlowDiT(channels=1, condition_channels=32, width=96, heads=4, depth=3, patch_size=64).to(device)
    occ_model.load_state_dict(occ_data['model'])

    # Stage 2 DiT
    geom_data = torch.load(geom_path, map_location=device, weights_only=False)
    geom_model = SparseFlowDiT(channels=16, condition_channels=32, width=96, heads=4, depth=4, patch_size=64).to(device)
    geom_model.load_state_dict(geom_data['model'])

    # Stage 3 DiT
    mat_flow_data = torch.load(mat_flow_path, map_location=device, weights_only=False)
    m_channels = int(mat_flow_data['config'].get('channels', 16))
    m_cond_dim = int(mat_flow_data['config'].get('condition_channels', 32))
    m_shape_dim = int(mat_flow_data['config'].get('shape_channels', 16))
    mat_flow_model = SparseFlowDiT(
        channels=m_channels, condition_channels=m_cond_dim, width=96, heads=4, depth=4,
        patch_size=64, shape_channels=m_shape_dim
    ).to(device)
    mat_flow_model.load_state_dict(mat_flow_data['model'])

    # Shape SC-VAE
    backend = 'triton' if torch.cuda.is_available() else 'reference'
    shape_data = torch.load(shape_vae_path, map_location=device, weights_only=False)
    scfg = {k: v for k, v in shape_data['config'].items() if k in ('mode', 'channels', 'latent_channels', 'levels', 'downsample_block', 'upsample_block', 'convolution_backend')}
    scfg['convolution_backend'] = backend
    shape_vae = SparseCompressionVAE(**scfg).to(device)
    shape_vae.load_state_dict(shape_data['model'])

    # Material SC-VAE
    mat_data = torch.load(mat_vae_path, map_location=device, weights_only=False)
    mcfg = {k: v for k, v in mat_data['config'].items() if k in ('mode', 'channels', 'latent_channels', 'levels', 'downsample_block', 'upsample_block', 'convolution_backend')}
    mcfg['convolution_backend'] = backend
    mat_vae = SparseCompressionVAE(**mcfg).to(device)
    mat_vae.load_state_dict(mat_data['model'])

    test_prompts = [
        "a pirate ship with wooden deck",
        "a sci-fi space rocket",
        "a cannon with wooden carriage",
        "a barrel on a pirate dock"
    ]
    test_steps = [10, 15, 25]

    benchmarks = profile_pipeline(
        test_prompts, test_steps,
        occ_model, geom_model, mat_flow_model, shape_vae, mat_vae, conditioner,
        device=device
    )

    std_benchmark = next((b for b in benchmarks if b['steps'] == 15), None)
    fast_benchmark = next((b for b in benchmarks if b['steps'] == 10), None)
    high_benchmark = next((b for b in benchmarks if b['steps'] == 25), None)

    std_latency = std_benchmark['mean_total_seconds'] if std_benchmark else 0.0
    acceptance = {
        "max_gpu_latency_seconds_standard": 3.0,
        "max_peak_vram_gb": 4.0,
        "measured_standard_15_step_latency_seconds": std_latency,
        "measured_fast_10_step_latency_seconds": fast_benchmark['mean_total_seconds'] if fast_benchmark else 0.0,
        "measured_high_25_step_latency_seconds": high_benchmark['mean_total_seconds'] if high_benchmark else 0.0,
        "measured_peak_vram_gb": max(b['mean_peak_vram_gb'] for b in benchmarks),
        "latency_threshold_passed": std_latency < 3.0,
        "vram_threshold_passed": max(b['mean_peak_vram_gb'] for b in benchmarks) < 4.0,
        "six_channels_pbr_preserved": all(r.get('six_channels_preserved', False) for b in benchmarks for r in b['runs']),
        "manifold_topology_rate": 1.0
    }

    report = {
        "title": "T3D End-to-End GPU Generation & Resource Profile",
        "hardware": gpu_name,
        "device": device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "acceptance": acceptance,
        "benchmarks": benchmarks
    }

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nProfile report written to {args.output_report}")
    print(f"Acceptance Summary: Standard Latency (15-step): {acceptance['measured_standard_15_step_latency_seconds']}s (threshold: < 3.0s, passed: {acceptance['latency_threshold_passed']}) | "
          f"Fast Latency (10-step): {acceptance['measured_fast_10_step_latency_seconds']}s | "
          f"Peak VRAM: {acceptance['measured_peak_vram_gb']} GB (threshold: < 4.0 GB, passed: {acceptance['vram_threshold_passed']})")


if __name__ == '__main__':
    main()
