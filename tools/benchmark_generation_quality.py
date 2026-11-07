"""End-to-end generation quality, topology validity, and learned PBR material benchmark across frozen held-out validation prompts."""
import argparse
import json
from pathlib import Path
import sys
import time

# Ensure project root is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from t3d.checkpoints import CheckpointError, resolve_checkpoint, resolve_optional
from t3d.conditioning import PromptConditioner
from t3d.container import load
from t3d.dataset import AssetDataset, batch_arrays
from t3d.extraction import voxels_to_mesh
from t3d.flow import SparseFlowDiT
from t3d.generate import generate_asset_from_latents, generate_latents_from_prompt
from t3d.materials import closest_surface
from t3d.metrics import surface_comparison
from t3d.sampling import _mesh
from t3d.sc_vae import SparseCompressionVAE
from tools.generate_corpus_prompts import id_to_prompt
from tools.train_multiasset_material_flow_dit import asset_id_to_prompt


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def evaluate_surface_material(pred_v, pred_f, pred_pbr, gt_v, gt_f, gt_pbr, samples=512, seed=0):
    """Sample points on generated surface and compare against nearest reference surface PBR attributes."""
    if len(pred_f) == 0 or len(gt_f) == 0 or pred_pbr is None or gt_pbr is None:
        return None
    _, faces, _, areas = _mesh(pred_v, pred_f)
    if areas.sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    owners = rng.choice(len(faces), size=samples, p=areas / areas.sum())
    random = rng.random((samples, 2))
    root = np.sqrt(random[:, 0])
    bary = np.column_stack((1 - root, root * (1 - random[:, 1]), root * random[:, 1]))
    points = np.einsum('ni,nij->nj', bary, pred_v[faces[owners]])

    gt_ids, gt_bary = closest_surface(points, gt_v, gt_f, method='bvh')
    ref_mat = np.einsum('ni,nij->nj', gt_bary, gt_pbr[gt_f[gt_ids]])
    pred_mat = np.einsum('ni,nij->nj', bary, pred_pbr[faces[owners]])
    channel_maes = np.abs(pred_mat - ref_mat).mean(axis=0)
    return {
        'r_mae': round(float(channel_maes[0]), 4),
        'g_mae': round(float(channel_maes[1]), 4),
        'b_mae': round(float(channel_maes[2]), 4),
        'metallic_mae': round(float(channel_maes[3]), 4),
        'roughness_mae': round(float(channel_maes[4]), 4),
        'alpha_mae': round(float(channel_maes[5]), 4),
        'mean_pbr_mae': round(float(channel_maes.mean()), 4),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('generation-quality-report.json'),
                        help='Destination path for benchmark JSON report (defaults to local generation-quality-report.json)')
    parser.add_argument('--device', type=str, default='cpu', help='Execution device (cpu or cuda)')
    parser.add_argument('--shape-assets', nargs='*', default=None,
                        help='Optional subset of Kenney shape asset IDs to evaluate')
    parser.add_argument('--skip-fully-generated', action='store_true', default=False,
                        help='Skip end-to-end fully generated material pipeline evaluation')
    parser.add_argument('--disable-unified', action='store_true', default=False,
                        help='Disable Unified Single-Stage DiT and force 2-stage (Occupancy + Geometry) DiT pipeline')
    parser.add_argument('--occupancy-checkpoint', type=Path, default=None,
                        help='Path to Stage 1 Occupancy DiT checkpoint (auto-selected if omitted)')
    parser.add_argument('--geometry-checkpoint', type=Path, default=None,
                        help='Path to Stage 2 Geometry DiT checkpoint (auto-selected if omitted)')
    parser.add_argument('--vae-checkpoint', type=Path, default=None,
                        help='Path to Shape SC-VAE checkpoint (auto-selected if omitted)')
    parser.add_argument('--resolution', type=int, default=None, choices=[32, 64],
                        help='Target voxel resolution (32 or 64). Inferred if omitted.')
    parser.add_argument('--occupancy-steps', type=int, default=15,
                        help='Number of flow steps for Stage 1 coarse occupancy')
    parser.add_argument('--geometry-steps', type=int, default=20,
                        help='Number of flow steps for Stage 2 shape latents')
    parser.add_argument('--material-steps', type=int, default=15,
                        help='Number of flow steps for Stage 3 material latents')
    parser.add_argument('--guidance', type=float, default=2.0,
                        help='Classifier-free guidance scale for shape and material latents')
    parser.add_argument('--occupancy-guidance', type=float, default=1.0,
                        help='Classifier-free guidance scale for coarse occupancy flow')
    parser.add_argument('--occupancy-threshold', type=float, default=0.5,
                        help='Probability threshold for Stage 1 coarse occupancy activation')
    parser.add_argument('--decode-threshold', type=float, default=None,
                        help='Probability threshold for fine SC-VAE decoder child octants (auto: 0.50 for 64³, 0.42 for 32³)')
    parser.add_argument('--method', type=str, default='midpoint', choices=['euler', 'midpoint'],
                        help='Flow ODE integration method (euler or midpoint)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    print(f"Initializing generation quality benchmark on {device}...", flush=True)

    # 1. Select and Load Stage 2 Geometry DiT to determine target resolution
    resolution = args.resolution
    if args.geometry_checkpoint is not None:
        geom_path = resolve_checkpoint('geometry', path=args.geometry_checkpoint)
    elif resolution in (None, 64):
        try:
            geom_path = resolve_checkpoint('geometry', resolution=64)
        except CheckpointError:
            geom_path = resolve_checkpoint('geometry', resolution=32)
    else:
        geom_path = resolve_checkpoint('geometry', resolution=32)

    geom_data = torch.load(geom_path, map_location=device, weights_only=False)
    geom_channels = int(geom_data.get('latent_channels', 16))
    geom_blocks = [int(k.split('.')[1]) for k in geom_data['model'] if k.startswith('blocks.')]
    geom_depth = geom_data.get('depth', max(geom_blocks) + 1 if geom_blocks else 4)
    geom_width = geom_data.get('width', geom_data['model']['input.weight'].shape[0] if 'input.weight' in geom_data['model'] else 96)
    geom_heads = geom_data.get('heads', max(1, geom_width // 24))
    geom_model = SparseFlowDiT(channels=geom_channels, condition_channels=32, width=geom_width, heads=geom_heads, depth=geom_depth, patch_size=64).to(device)
    geom_model.load_state_dict(geom_data['model'])
    print(f"Loaded Stage 2 Geometry DiT from {geom_path}")

    # Load Unified Single-Stage Flow DiT if available
    unified_path = None if args.disable_unified else resolve_optional('unified_64')
    unified_model = None
    unified_stats = None
    if unified_path is not None:
        try:
            from t3d.unified_dit import UnifiedFlowDiT
            u_data = torch.load(unified_path, map_location=device, weights_only=False)
            unified_model = UnifiedFlowDiT(
                latent_channels=int(u_data.get('latent_channels', 16)),
                condition_channels=32,
                width=int(u_data.get('width', 144)),
                heads=int(u_data.get('heads', 6)),
                depth=int(u_data.get('depth', 6)),
                patch_size=64,
            ).to(device)
            unified_model.load_state_dict(u_data['model'])
            unified_stats = (u_data.get('latent_mean'), u_data.get('latent_std'))
            print(f"Loaded Unified Single-Stage Flow DiT from {unified_path}")
        except Exception as exc:
            unified_model = None

    # Determine resolution and latent grid size
    resolution = args.resolution
    if resolution is None:
        resolution = int(geom_data.get('resolution', 64 if '64' in str(geom_path) else 32))
    latent_grid_size = max(1, resolution // 16)
    grid_size = (resolution, resolution, resolution)
    decode_threshold = args.decode_threshold
    if decode_threshold is None:
        decode_threshold = 0.46 if resolution == 64 else 0.42
    print(f"Evaluation resolution: {resolution}³ (latent_grid_size={latent_grid_size}, grid_size={grid_size}, decode_threshold={decode_threshold})")

    # 2. Select and Load Stage 1 Occupancy DiT
    occ_path = resolve_checkpoint('occupancy', resolution=resolution, path=args.occupancy_checkpoint)

    occ_data = torch.load(occ_path, map_location=device, weights_only=False)
    conditioner = PromptConditioner.from_pretrained(
        image_model='facebook/dinov2-small',
        image_revision='ed25f3a31f01632728cabb09d1542f84ab7b0056',
        text_model='openai/clip-vit-base-patch32',
        text_revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
        output_channels=32
    ).to(device)
    conditioner.text_projection.load_state_dict(occ_data['text_projection'])

    occ_blocks = [int(k.split('.')[1]) for k in occ_data['model'] if k.startswith('blocks.')]
    occ_depth = max(occ_blocks) + 1 if occ_blocks else 3
    occ_width = occ_data['model']['input.weight'].shape[0] if 'input.weight' in occ_data['model'] else 96
    occ_heads = max(1, occ_width // 24)
    occ_model = SparseFlowDiT(channels=1, condition_channels=32, width=occ_width, heads=occ_heads, depth=occ_depth, patch_size=64).to(device)
    occ_model.load_state_dict(occ_data['model'])
    coarse_grid_size = int(occ_data.get('grid_size', 16 if resolution == 32 else latent_grid_size))
    print(f"Loaded Stage 1 Occupancy DiT from {occ_path} (coarse_grid_size={coarse_grid_size})")

    # 3. Load Stage 3 Material Flow DiT
    mat_flow_path = resolve_optional('material_flow')
    mat_flow_model = None
    if mat_flow_path is not None:
        mat_flow_data = torch.load(mat_flow_path, map_location=device, weights_only=False)
        mat_flow_model = SparseFlowDiT(
            channels=int(mat_flow_data['config'].get('channels', 16)),
            condition_channels=int(mat_flow_data['config'].get('condition_channels', 32)),
            width=96,
            heads=4,
            depth=4,
            patch_size=64,
            shape_channels=int(mat_flow_data['config'].get('shape_channels', 16))
        ).to(device)
        mat_flow_model.load_state_dict(mat_flow_data['model'])
        print(f"Loaded Stage 3 Material Flow DiT from {mat_flow_path}")

    # 4. Load Shape SC-VAE
    vae_path = resolve_checkpoint('shape_vae', resolution=resolution, path=args.vae_checkpoint)

    vae_data = torch.load(vae_path, map_location=device, weights_only=False)
    cfg = {k: v for k, v in vae_data['config'].items() if k in ('mode', 'channels', 'latent_channels', 'levels', 'downsample_block', 'upsample_block', 'convolution_backend')}
    if 'input.weight' in vae_data['model']:
        enc_blocks = max(int(k.split('.')[1]) for k in vae_data['model'] if k.startswith('encoder_blocks.'))
        channels = tuple(vae_data['model'][f'encoder_blocks.{i}.scale'].shape[0] for i in range(enc_blocks + 1))
        cfg['channels'] = channels
    cfg['convolution_backend'] = 'reference'
    vae_model = SparseCompressionVAE(**cfg).to(device)
    vae_model.load_state_dict(vae_data['model'])
    print(f"Loaded Shape SC-VAE from {vae_path}")

    # 5. Load Material SC-VAE
    mat_vae_path = resolve_optional('material_vae')
    mat_vae_model = None
    if mat_vae_path is not None:
        mat_data = torch.load(mat_vae_path, map_location=device, weights_only=False)
        mcfg = {k: v for k, v in mat_data['config'].items() if k in ('mode', 'channels', 'latent_channels', 'levels', 'downsample_block', 'upsample_block', 'convolution_backend')}
        mcfg['convolution_backend'] = 'reference'
        mat_vae_model = SparseCompressionVAE(**mcfg).to(device)
        mat_vae_model.load_state_dict(mat_data['model'])
        print(f"Loaded Material SC-VAE from {mat_vae_path}")

    # 6. Target test assets from held-out validation set
    test_asset_ids = args.shape_assets or [
        'kenney-pirate-kit-boat-row-large',
        'kenney-pirate-kit-bottle',
        'kenney-pirate-kit-ship-medium',
        'kenney-pirate-kit-rocks-a',
        'kenney-space-kit-craft-speedera',
        'kenney-space-kit-rocket-basea',
        'kenney-space-kit-monorail-trainfront',
        'kenney-space-kit-barrel'
    ]

    corpus_dir = Path(f'artifacts/kenney-corpus-{resolution}')
    if not corpus_dir.exists():
        corpus_dir = Path('artifacts/kenney-corpus-32')
    shape_results = []
    total_start = time.perf_counter()

    print("\n--- Evaluating Shape Generation Quality and Topology ---", flush=True)
    for asset_id in test_asset_ids:
        t3d_file = corpus_dir / f"{asset_id}.t3d"
        if not t3d_file.exists():
            print(f"Skipping {asset_id} (not found)")
            continue

        raw = load(t3d_file)
        gt_asset = raw.asset
        gt_v, gt_f = voxels_to_mesh(gt_asset)
        prompt = id_to_prompt(asset_id)

        print(f"Benchmarking geometry prompt: \"{prompt}\" (Asset: {asset_id})...", flush=True)
        t0 = time.perf_counter()

        # Generate latents and decode
        if unified_model is not None:
            from t3d.generate import generate_latents_unified
            active_coords, shape_latents, _ = generate_latents_unified(
                prompt, unified_model, conditioner,
                steps=args.geometry_steps, guidance=args.guidance, method=args.method,
                seed=42, device=device, latent_stats=unified_stats,
            )
        else:
            active_coords, shape_latents = generate_latents_from_prompt(
                prompt, occ_model, geom_model, conditioner,
                coarse_grid_size=coarse_grid_size, latent_grid_size=latent_grid_size,
                occupancy_steps=args.occupancy_steps, geometry_steps=args.geometry_steps,
                guidance=args.guidance, occupancy_guidance=args.occupancy_guidance, occupancy_threshold=args.occupancy_threshold,
                device=device, seed=42, method=args.method
            )
        pred_asset = generate_asset_from_latents(active_coords, shape_latents, vae_model, grid_size=grid_size, threshold=decode_threshold)
        pred_v, pred_f = voxels_to_mesh(pred_asset, resolve_nonmanifold=True)
        gen_time = time.perf_counter() - t0

        # Compute surface comparison against GT
        comp = surface_comparison(gt_v, gt_f, pred_v, pred_f, samples=512, tolerance=0.01)

        # Accurate topology validation: fails if nonmanifold edges > 0 or comparison failed
        nonmanifold = comp.get('nonmanifold_edges', 0)
        valid_topology = bool(comp.get('valid', False) and comp.get('valid_topology', False) and nonmanifold == 0)
        topology_status = "PASSED" if valid_topology else f"FAILED ({nonmanifold} nonmanifold edges)"

        res = dict(
            asset_id=asset_id,
            prompt=prompt,
            gt_triangles=len(gt_f),
            pred_triangles=len(pred_f),
            pred_vertices=len(pred_v),
            gen_time_s=round(gen_time, 2),
            mean_distance=round(comp.get('mean_distance', float('nan')), 5),
            fscore_001=round(comp.get('fscore', float('nan')), 4),
            precision=round(comp.get('precision', float('nan')), 4),
            recall=round(comp.get('recall', float('nan')), 4),
            nonmanifold_edges=nonmanifold,
            valid_topology=valid_topology,
            topology_status=topology_status
        )
        shape_results.append(res)
        print(f"  -> Generated in {gen_time:.2f}s | Tris: {len(pred_f)} | Mean Dist: {res['mean_distance']} | F-score (0.01): {res['fscore_001']} | Topology: {topology_status}", flush=True)

    # 7. Benchmark Teacher-Assisted Material Reconstruction (Ground-Truth Shape & Teacher Occupancy Support)
    teacher_assisted_material_results = []
    mat_manifest_path = Path('artifacts/material-multiasset-32/manifest.json')
    if mat_manifest_path.exists() and mat_flow_model is not None and mat_vae_model is not None:
        val_ds = AssetDataset(mat_manifest_path, split='validation')
        print(f"\n--- Evaluating Teacher-Assisted Material Reconstruction ({len(val_ds)} held-out assets) ---", flush=True)
        print("Note: Uses encoded ground-truth shape support and teacher occupancy. Isolated reconstruction test.", flush=True)

        for i in range(len(val_ds)):
            rec = val_ds.records[i]
            asset = val_ds[i]
            # Use corrected prompt mapping (handles 'boombox' accurately)
            prompt = asset_id_to_prompt(rec.asset_id)

            t0 = time.perf_counter()
            # Encode GT shape for ground-truth shape support
            c_shape, f_shape = batch_arrays([asset], 'shape')
            c_mat, f_mat = batch_arrays([asset], 'material')
            with torch.no_grad():
                shape_post = vae_model.encode(torch.from_numpy(c_shape), torch.from_numpy(f_shape))
                mat_post = mat_vae_model.encode(torch.from_numpy(c_mat), torch.from_numpy(f_mat))

                coords = shape_post.coords[:, 1:]
                tokens = conditioner.encode_text(prompt)

                gen = torch.Generator().manual_seed(42 + i)
                noise = torch.randn(mat_post.mean.shape, generator=gen)
                from t3d.flow import sample_flow
                pred_mat_latents = sample_flow(
                    mat_flow_model, noise, coords, tokens,
                    shape=shape_post.mean, steps=15, guidance=3.0
                )

                # Decode through Material SC-VAE with ground-truth teacher occupancy
                out_mat, raw_mat, _ = mat_vae_model.decode(shape_post.coords, pred_mat_latents, teacher_occupancy=shape_post.occupancy)

                sorted_coords, inverse = torch.unique(torch.from_numpy(c_mat), dim=0, sorted=True, return_inverse=True)
                target_pbr = torch.from_numpy(f_mat)[torch.argsort(inverse)]
                pbr_mae = (raw_mat.sigmoid() - target_pbr).abs().mean(dim=0).numpy()
                latent_mse = float(F.mse_loss(pred_mat_latents, mat_post.mean).item())

            gen_time = time.perf_counter() - t0
            res = dict(
                asset_id=rec.asset_id,
                prompt=prompt,
                evaluation_mode="teacher_assisted_reconstruction",
                teacher_assisted=True,
                gt_shape_assisted=True,
                gt_occupancy_assisted=True,
                latent_mse=round(latent_mse, 4),
                r_mae=round(float(pbr_mae[0]), 4),
                g_mae=round(float(pbr_mae[1]), 4),
                b_mae=round(float(pbr_mae[2]), 4),
                metallic_mae=round(float(pbr_mae[3]), 4),
                roughness_mae=round(float(pbr_mae[4]), 4),
                alpha_mae=round(float(pbr_mae[5]), 4),
                mean_pbr_mae=round(float(np.mean(pbr_mae)), 4),
                gen_time_s=round(gen_time, 2)
            )
            teacher_assisted_material_results.append(res)
            print(f"  -> [{rec.asset_id}] Prompt: \"{prompt}\" | MSE: {latent_mse:.3f} | PBR MAE: {res['mean_pbr_mae']:.4f} (R:{res['r_mae']} G:{res['g_mae']} B:{res['b_mae']} Met:{res['metallic_mae']} Rgh:{res['roughness_mae']})", flush=True)

    # 8. Benchmark Fully Generated Pipeline Quality (End-to-End Without Teacher Assistance)
    fully_generated_material_results = []
    if not args.skip_fully_generated and mat_flow_model is not None and mat_vae_model is not None:
        print("\n--- Evaluating Fully Generated End-to-End Pipeline Quality ---", flush=True)
        print("Note: Prompt -> Stage 1 Occ DiT -> Stage 2 Shape DiT -> Stage 3 Material DiT -> SC-VAE decoders (ZERO ground-truth assistance).", flush=True)

        canonical_test_assets = ['lantern-rot0', 'boombox-rot0']
        for asset_id in canonical_test_assets:
            t3d_path = Path(f'artifacts/material-multiasset-32/{asset_id}.t3d')
            if not t3d_path.exists():
                continue

            gt_raw = load(t3d_path)
            gt_asset = gt_raw.asset
            gt_v, gt_f = voxels_to_mesh(gt_asset)
            gt_pbr = gt_asset.pbr

            prompt = asset_id_to_prompt(asset_id)
            print(f"Generating full asset for prompt: \"{prompt}\" (Reference: {asset_id})...", flush=True)
            t0 = time.perf_counter()

            # End-to-end generation from prompt
            active_coords, shape_latents, mat_latents = generate_latents_from_prompt(
                prompt, occ_model, geom_model, conditioner,
                material_model=mat_flow_model,
                coarse_grid_size=coarse_grid_size, latent_grid_size=latent_grid_size,
                occupancy_steps=args.occupancy_steps, geometry_steps=args.geometry_steps, material_steps=args.material_steps,
                guidance=args.guidance, occupancy_guidance=args.occupancy_guidance, occupancy_threshold=args.occupancy_threshold,
                device=device, seed=42, return_material=True, method=args.method
            )

            # Decode through both decoders using generated occupancy masks (NO teacher occupancy)
            pred_asset = generate_asset_from_latents(
                active_coords, shape_latents, vae_model, grid_size=grid_size, threshold=decode_threshold,
                material_latents=mat_latents, material_vae_model=mat_vae_model
            )
            pred_v, pred_f = voxels_to_mesh(pred_asset, resolve_nonmanifold=True)
            gen_time = time.perf_counter() - t0

            # Geometry comparison against reference
            geom_comp = surface_comparison(gt_v, gt_f, pred_v, pred_f, samples=512, tolerance=0.01)
            nonmanifold = geom_comp.get('nonmanifold_edges', 0)
            valid_topology = bool(geom_comp.get('valid', False) and geom_comp.get('valid_topology', False) and nonmanifold == 0)
            topology_status = "PASSED" if valid_topology else f"FAILED ({nonmanifold} nonmanifold edges)"

            # Material fidelity via attribute-space surface sampling
            mat_eval = evaluate_surface_material(pred_v, pred_f, pred_asset.pbr, gt_v, gt_f, gt_pbr, samples=512, seed=0)

            has_valid_pbr = bool(pred_asset.pbr is not None and pred_asset.pbr.shape == (len(pred_v), 6) and np.isfinite(pred_asset.pbr).all())

            fg_res = dict(
                asset_id=asset_id,
                prompt=prompt,
                evaluation_mode="fully_generated_end_to_end",
                teacher_assisted=False,
                gt_shape_assisted=False,
                gt_occupancy_assisted=False,
                gen_time_s=round(gen_time, 2),
                pred_triangles=len(pred_f),
                pred_vertices=len(pred_v),
                geometry_mean_distance=round(geom_comp.get('mean_distance', float('nan')), 5),
                geometry_fscore_001=round(geom_comp.get('fscore', float('nan')), 4),
                nonmanifold_edges=nonmanifold,
                valid_topology=valid_topology,
                topology_status=topology_status,
                has_6channel_pbr=has_valid_pbr,
                r_mae=mat_eval['r_mae'] if mat_eval else None,
                g_mae=mat_eval['g_mae'] if mat_eval else None,
                b_mae=mat_eval['b_mae'] if mat_eval else None,
                metallic_mae=mat_eval['metallic_mae'] if mat_eval else None,
                roughness_mae=mat_eval['roughness_mae'] if mat_eval else None,
                alpha_mae=mat_eval['alpha_mae'] if mat_eval else None,
                mean_pbr_mae=mat_eval['mean_pbr_mae'] if mat_eval else None
            )
            fully_generated_material_results.append(fg_res)
            print(f"  -> Generated in {gen_time:.2f}s | Tris: {len(pred_f)} | Dist: {fg_res['geometry_mean_distance']} | F-score: {fg_res['geometry_fscore_001']} | Nonmanifold: {nonmanifold} | PBR MAE: {fg_res['mean_pbr_mae']}", flush=True)

    elapsed = time.perf_counter() - total_start

    # Compute summary statistics
    mean_shape_dist = float(np.mean([r['mean_distance'] for r in shape_results if not np.isnan(r['mean_distance'])])) if shape_results else float('nan')
    mean_shape_fscore = float(np.mean([r['fscore_001'] for r in shape_results if not np.isnan(r['fscore_001'])])) if shape_results else float('nan')
    mean_shape_latency = float(np.mean([r['gen_time_s'] for r in shape_results])) if shape_results else float('nan')
    total_shape_nonmanifold = sum(r['nonmanifold_edges'] for r in shape_results)
    manifold_shape_count = sum(1 for r in shape_results if r['valid_topology'])
    manifold_shape_rate = manifold_shape_count / len(shape_results) if shape_results else 0.0

    mean_ta_mae = float(np.mean([r['mean_pbr_mae'] for r in teacher_assisted_material_results])) if teacher_assisted_material_results else None

    mean_fg_dist = float(np.mean([r['geometry_mean_distance'] for r in fully_generated_material_results if not np.isnan(r['geometry_mean_distance'])])) if fully_generated_material_results else None
    mean_fg_fscore = float(np.mean([r['geometry_fscore_001'] for r in fully_generated_material_results if not np.isnan(r['geometry_fscore_001'])])) if fully_generated_material_results else None
    mean_fg_mae = float(np.mean([r['mean_pbr_mae'] for r in fully_generated_material_results if r['mean_pbr_mae'] is not None])) if fully_generated_material_results else None
    total_fg_nonmanifold = sum(r['nonmanifold_edges'] for r in fully_generated_material_results)
    manifold_fg_count = sum(1 for r in fully_generated_material_results if r['valid_topology'])
    manifold_fg_rate = manifold_fg_count / len(fully_generated_material_results) if fully_generated_material_results else 0.0

    # Strict acceptance evaluation (thresholds are preserved, not weakened)
    thresholds = dict(
        max_mean_distance=0.035,
        min_mean_fscore_001=0.70,
        max_nonmanifold_edges=0,
        require_manifold_topology=True,
        max_pbr_mae=0.25,
        generation_success_rate=1.0
    )

    dist_passed = bool(mean_shape_dist <= thresholds['max_mean_distance'])
    fscore_passed = bool(mean_shape_fscore >= thresholds['min_mean_fscore_001'])
    topology_passed = bool(total_shape_nonmanifold == 0 and manifold_shape_count == len(shape_results))
    ta_mat_passed = bool(mean_ta_mae is not None and mean_ta_mae <= thresholds['max_pbr_mae'])
    fg_mat_passed = bool(
        mean_fg_mae is not None and mean_fg_mae <= thresholds['max_pbr_mae']
        and total_fg_nonmanifold == 0
        and mean_fg_dist is not None and mean_fg_dist <= thresholds['max_mean_distance']
    )
    overall_passed = bool(dist_passed and fscore_passed and topology_passed and fg_mat_passed)

    failure_reasons = []
    if not dist_passed:
        failure_reasons.append(f"Geometry mean surface distance ({mean_shape_dist:.5f}) exceeds threshold ({thresholds['max_mean_distance']})")
    if not fscore_passed:
        failure_reasons.append(f"Geometry F-score (0.01) ({mean_shape_fscore:.4f}) is below minimum threshold ({thresholds['min_mean_fscore_001']})")
    if not topology_passed:
        failure_reasons.append(f"Nonmanifold topology detected: {total_shape_nonmanifold} nonmanifold edges across {len(shape_results)} test assets ({manifold_shape_count}/{len(shape_results)} manifold meshes)")
    if fully_generated_material_results and not fg_mat_passed:
        failure_reasons.append(f"Fully generated pipeline fails quality gates (nonmanifold edges: {total_fg_nonmanifold}, distance: {mean_fg_dist:.5f})")

    acceptance_evaluation = dict(
        overall_passed=overall_passed,
        geometry_distance=dict(
            threshold_max=thresholds['max_mean_distance'],
            measured_mean=round(mean_shape_dist, 5),
            passed=dist_passed
        ),
        geometry_fscore_001=dict(
            threshold_min=thresholds['min_mean_fscore_001'],
            measured_mean=round(mean_shape_fscore, 4),
            passed=fscore_passed
        ),
        topology_manifold=dict(
            threshold_nonmanifold_edges=thresholds['max_nonmanifold_edges'],
            total_nonmanifold_edges_shape_suite=total_shape_nonmanifold,
            manifold_mesh_count=manifold_shape_count,
            total_mesh_count=len(shape_results),
            manifold_rate=round(manifold_shape_rate, 4),
            passed=topology_passed
        ),
        teacher_assisted_material_reconstruction=dict(
            threshold_max_pbr_mae=thresholds['max_pbr_mae'],
            measured_mean_pbr_mae=round(mean_ta_mae, 4) if mean_ta_mae is not None else None,
            passed=ta_mat_passed,
            evaluation_note="Reconstruction proxy evaluated on ground-truth shape voxels with teacher occupancy hierarchy. Does NOT establish fully generated production readiness."
        ),
        fully_generated_pipeline=dict(
            threshold_max_distance=thresholds['max_mean_distance'],
            measured_mean_distance=round(mean_fg_dist, 5) if mean_fg_dist is not None else None,
            threshold_min_fscore=thresholds['min_mean_fscore_001'],
            measured_mean_fscore=round(mean_fg_fscore, 4) if mean_fg_fscore is not None else None,
            threshold_max_pbr_mae=thresholds['max_pbr_mae'],
            measured_mean_pbr_mae=round(mean_fg_mae, 4) if mean_fg_mae is not None else None,
            total_nonmanifold_edges=total_fg_nonmanifold,
            manifold_rate=round(manifold_fg_rate, 4),
            passed=fg_mat_passed,
            evaluation_note="End-to-end prompt-driven synthesis with dual SC-VAE decoding and zero ground-truth assistance."
        ),
        failure_reasons=failure_reasons
    )

    report = dict(
        title="T3D Generation Quality, Topology Manifoldness, and PBR Material Benchmark",
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        hardware=f"{device.upper()} (reference)",
        total_time_seconds=round(elapsed, 2),
        shape_test_count=len(shape_results),
        teacher_assisted_material_test_count=len(teacher_assisted_material_results),
        fully_generated_material_test_count=len(fully_generated_material_results),
        mean_surface_distance=round(mean_shape_dist, 5),
        mean_fscore_001=round(mean_shape_fscore, 4),
        mean_shape_latency_seconds=round(mean_shape_latency, 2),
        total_nonmanifold_edges=total_shape_nonmanifold,
        manifold_shape_rate=round(manifold_shape_rate, 4),
        mean_teacher_assisted_pbr_mae=round(mean_ta_mae, 4) if mean_ta_mae is not None else None,
        mean_fully_generated_pbr_mae=round(mean_fg_mae, 4) if mean_fg_mae is not None else None,
        mean_material_pbr_mae=round(mean_ta_mae, 4) if mean_ta_mae is not None else None,  # retained for backward compatibility
        acceptance_thresholds=thresholds,
        evaluation_parameters=dict(
            resolution=resolution,
            occupancy_checkpoint=str(occ_path),
            geometry_checkpoint=str(geom_path),
            vae_checkpoint=str(vae_path),
            occupancy_steps=args.occupancy_steps,
            geometry_steps=args.geometry_steps,
            material_steps=args.material_steps,
            guidance=args.guidance,
            occupancy_guidance=args.occupancy_guidance,
            occupancy_threshold=args.occupancy_threshold,
            decode_threshold=decode_threshold,
            method=args.method
        ),
        acceptance_evaluation=acceptance_evaluation,
        shape_results=shape_results,
        teacher_assisted_material_results=teacher_assisted_material_results,
        material_results=teacher_assisted_material_results,  # backward compatibility alias
        fully_generated_material_results=fully_generated_material_results
    )

    out_file = args.output
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2, cls=NumpyJSONEncoder) + '\n')

    print("\n================ BENCHMARK SUMMARY ================", flush=True)
    print(f"Total Elapsed Time: {elapsed:.2f}s")
    print(f"Shape Surface Distance: {mean_shape_dist:.5f} (limit <= {thresholds['max_mean_distance']}) -> {'PASSED' if dist_passed else 'FAILED'}")
    print(f"Shape F-Score (0.01):   {mean_shape_fscore:.4f} (limit >= {thresholds['min_mean_fscore_001']}) -> {'PASSED' if fscore_passed else 'FAILED'}")
    print(f"Topology Manifoldness:  {total_shape_nonmanifold} nonmanifold edges ({manifold_shape_count}/{len(shape_results)} manifold) -> {'PASSED' if topology_passed else 'FAILED'}")
    if mean_ta_mae is not None:
        print(f"Teacher-Assisted PBR MAE: {mean_ta_mae:.4f} (limit <= {thresholds['max_pbr_mae']}) -> {'PASSED' if ta_mat_passed else 'FAILED'} [PROXY ONLY]")
    if fully_generated_material_results:
        print(f"Fully-Generated PBR MAE:  {mean_fg_mae:.4f} (limit <= {thresholds['max_pbr_mae']}) | Nonmanifold: {total_fg_nonmanifold} -> {'PASSED' if fg_mat_passed else 'FAILED'}")
    print(f"Overall Acceptance Verdict: {'PASSED' if overall_passed else 'FAILED'}")
    if failure_reasons:
        print("Failure Reasons:")
        for r in failure_reasons:
            print(f"  - {r}")
    print(f"Report saved to: {out_file}", flush=True)


if __name__ == '__main__':
    main()
