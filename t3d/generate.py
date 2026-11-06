"""End-to-end Text-to-3D generation pipeline and CLI for .t3d and .glb."""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from .container import dumps
from .dataset import batch_arrays
from .export import export_raw
from .extraction import voxels_to_mesh
from .flow import sample_flow
from .schema import VoxelAsset


@torch.no_grad()
def generate_latents_from_prompt(
    prompt: str,
    occupancy_model,
    geometry_model,
    conditioner,
    *,
    material_model=None,
    coarse_grid_size: int = 16,
    latent_grid_size: int = 2,
    occupancy_steps: int = 15,
    geometry_steps: int = 20,
    material_steps: int = 15,
    guidance: float = 2.0,
    occupancy_guidance: float = 1.0,
    occupancy_threshold: float = 0.5,
    device: str = 'cuda',
    seed: int = 42,
    orders: list | None = None,
    return_material: bool = False,
    method: str = 'midpoint'
):
    """Generate coarse active coordinates, shape latents, and optional material latents from a text prompt."""
    occupancy_model.eval()
    geometry_model.eval()
    conditioner.eval()

    tokens = conditioner.encode_text(prompt).to(device)

    # Stage 1: Coarse Occupancy Flow
    grid_coords = torch.cartesian_prod(*(torch.arange(coarse_grid_size, device=device) for _ in range(3)))
    gen_occ = torch.Generator(device=device).manual_seed(seed)
    occ_noise = torch.randn((len(grid_coords), 1), device=device, generator=gen_occ)

    pred_occ = sample_flow(
        occupancy_model, occ_noise, grid_coords, tokens,
        steps=occupancy_steps, guidance=occupancy_guidance, orders=orders, method=method
    )
    active_mask = pred_occ[:, 0] > occupancy_threshold
    if not active_mask.any():
        # Fallback to top-k cells if thresholding yields empty volume
        topk = torch.topk(pred_occ[:, 0], k=min(64, len(pred_occ))).indices
        active_mask[topk] = True

    active_coords = grid_coords[active_mask]
    # Downsample active coarse coordinates to the SC-VAE latent grid (e.g. 16 // 2 = 8 scale for 2³ latent grid)
    latent_scale = max(1, coarse_grid_size // latent_grid_size)
    latent_coords = torch.unique(torch.div(active_coords, latent_scale, rounding_mode='floor'), dim=0)

    # Stage 2: Latent Shape Flow at active latent coordinates (canonical generator)
    gen_geom = torch.Generator(device=device).manual_seed(seed)
    geom_noise = torch.randn((len(latent_coords), geometry_model.channels), device=device, generator=gen_geom)
    shape_latents = sample_flow(
        geometry_model, geom_noise, latent_coords, tokens,
        steps=geometry_steps, guidance=guidance, method=method
    )

    # Stage 3: Shape-Conditioned Material Flow at active latent coordinates
    mat_latents = None
    if material_model is not None:
        material_model.eval()
        gen_mat = torch.Generator(device=device).manual_seed(seed)
        mat_noise = torch.randn((len(latent_coords), material_model.channels), device=device, generator=gen_mat)
        mat_latents = sample_flow(
            material_model, mat_noise, latent_coords, tokens,
            shape=shape_latents, steps=material_steps, guidance=guidance, method=method
        )

    if return_material:
        return latent_coords, shape_latents, mat_latents
    return latent_coords, shape_latents


@torch.no_grad()
def generate_asset_from_latents(
    active_coords: torch.Tensor,
    shape_latents: torch.Tensor,
    vae_model,
    *,
    grid_size: tuple[int, int, int] = (32, 32, 32),
    aabb: list | None = None,
    threshold: float = 0.4,
    material_latents: torch.Tensor | None = None,
    material_vae_model = None,
    rig_model = None,
    tokens: torch.Tensor | None = None,
    rig_steps: int = 25,
    guidance: float = 3.0
) -> VoxelAsset:
    """Decode shape (and optional material and rig) latents into a verified VoxelAsset."""
    if aabb is None:
        aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]

    vae_model.eval()
    # Batch dimension 0 prepended for decoding
    batched_coords = torch.cat([torch.zeros((len(active_coords), 1), device=active_coords.device, dtype=active_coords.dtype), active_coords], dim=1)

    out, raw, logits = vae_model.decode(batched_coords, shape_latents, threshold=threshold)
    if len(out) == 0:
        raise ValueError('VAE decoding produced zero active surface voxels')

    valid = (out[:, 1:] < out.new_tensor(grid_size)).all(dim=1)
    out, raw = out[valid], raw[valid]

    coords_np = out[:, 1:].cpu().numpy().astype(int)
    dual_np = raw[:, :3].sigmoid().cpu().numpy()
    flags_np = (raw[:, 3:6].sigmoid() > threshold).cpu().numpy()
    split_np = raw[:, 6:].cpu().numpy()

    pbr_np = None
    if material_vae_model is not None:
        material_vae_model.eval()
        if material_latents is None:
            latent_dim = material_vae_model.from_latent.in_features
            material_latents = torch.zeros((len(active_coords), latent_dim), device=active_coords.device, dtype=shape_latents.dtype)
        shape_masks = [(l.sigmoid() > threshold) for l in logits[::-1]]
        out_mat, raw_mat, _ = material_vae_model.decode(batched_coords, material_latents, teacher_occupancy=shape_masks)
        raw_mat = raw_mat[valid]
        pbr_np = raw_mat.sigmoid().cpu().numpy()

    skel_np = None
    skin_np = None
    if rig_model is not None and tokens is not None:
        rig_model.eval()
        shape_feat = torch.cat([raw[:, :3].sigmoid(), (raw[:, 3:6].sigmoid() > threshold).float(), raw[:, 6:7]], dim=-1)
        rig_noise = torch.randn((len(out), rig_model.channels), device=out.device)
        pred_rig = sample_flow(
            rig_model, rig_noise, out[:, 1:], tokens,
            shape=shape_feat, steps=rig_steps, guidance=guidance
        )
        skel_np = pred_rig[:, :6].cpu().numpy()
        skin_np = pred_rig[:, 6:].cpu().numpy()

    asset = VoxelAsset(
        coords_np, dual_np, flags_np, grid_size, np.asarray(aabb, dtype=np.float32),
        split_weight=split_np, pbr=pbr_np, skeleton_field=skel_np, skin_embedding=skin_np
    )
    return asset.canonical()


def main():
    parser = argparse.ArgumentParser(description='Generate 3D models from text prompts using T3D flow models')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt describing the desired 3D object')
    parser.add_argument('--occupancy-checkpoint', type=Path, default=None, help='Path to Stage 1 Occupancy DiT checkpoint (auto-selected if omitted)')
    parser.add_argument('--geometry-checkpoint', type=Path, default=None, help='Path to Stage 2 Geometry DiT checkpoint (auto-selected if omitted)')
    parser.add_argument('--material-flow-checkpoint', type=Path, default=None, help='Path to Stage 3 Material Flow DiT checkpoint')
    parser.add_argument('--rig-flow-checkpoint', type=Path, default=None, help='Path to Stage 2 Rig Flow DiT checkpoint')
    parser.add_argument('--vae-checkpoint', type=Path, default=None, help='Path to Shape SC-VAE checkpoint (auto-selected if omitted)')
    parser.add_argument('--material-checkpoint', type=Path, default=None, help='Path to Material SC-VAE checkpoint')
    parser.add_argument('--skinae-checkpoint', type=Path, default=None, help='Path to SkinAE checkpoint for rig decoding')
    parser.add_argument('--resolution', type=int, default=None, choices=[32, 64], help='Target voxel resolution (32 or 64). Inferred from checkpoint if omitted.')
    parser.add_argument('--output', type=Path, required=True, help='Output path (.glb or .t3d)')
    parser.add_argument('--steps', type=int, default=20, help='Flow matching sampling steps')
    parser.add_argument('--guidance', type=float, default=2.0, help='Classifier-free guidance weight for geometry and material')
    parser.add_argument('--occupancy-guidance', type=float, default=1.0, help='Classifier-free guidance weight for coarse occupancy flow matching')
    parser.add_argument('--rig-guidance', type=float, default=1.0, help='Classifier-free guidance weight for rig flow matching')
    parser.add_argument('--threshold', type=float, default=0.5, help='Occupancy decision threshold')
    parser.add_argument('--decode-threshold', type=float, default=None, help='SC-VAE decoder child octant threshold (auto: 0.50 for 64³, 0.42 for 32³)')
    parser.add_argument('--method', type=str, default='midpoint', choices=['euler', 'midpoint'], help='Flow sampling integration method (euler or midpoint)')
    parser.add_argument('--material', action='store_true', help='Enable Stage 3 PBR material flow and texture decoding')
    parser.add_argument('--rig', action='store_true', help='Enable Stage 2 Rig flow and SkinAE skeleton/skinning decoding')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Auto-resolve checkpoints if not explicitly provided
    geom_path = args.geometry_checkpoint
    if geom_path is None:
        cand_64_4000 = Path('artifacts/kaggle-kenney-64-geometry-dit/geometry-dit-64-4000.pt')
        cand_64_3000 = Path('artifacts/kaggle-kenney-64-geometry-dit/geometry-dit-64-3000.pt')
        cand_64_2000 = Path('artifacts/kaggle-kenney-64-geometry-dit/geometry-dit-64-2000.pt')
        cand_64 = Path('artifacts/kaggle-kenney-64-geometry-dit/geometry-dit-64-600.pt')
        if cand_64_4000.exists():
            geom_path = cand_64_4000
        elif cand_64_3000.exists():
            geom_path = cand_64_3000
        elif cand_64_2000.exists():
            geom_path = cand_64_2000
        elif cand_64.exists():
            geom_path = cand_64
        else:
            geom_path = Path('artifacts/kaggle-kenney-geometry-dit-v38/geometry-dit-600.pt')

    if not geom_path.exists():
        raise FileNotFoundError(f"Stage 2 Geometry DiT checkpoint not found: {geom_path}")

    resolution = args.resolution
    if resolution is None:
        resolution = 64 if '64' in geom_path.name else 32

    occ_path = args.occupancy_checkpoint
    if occ_path is None:
        if resolution == 64:
            cand_occ_64_4000 = Path('artifacts/kaggle-kenney-64-occupancy-dit/occupancy-dit-64-4000.pt')
            cand_occ_64_3000 = Path('artifacts/kaggle-kenney-64-occupancy-dit/occupancy-dit-64-3000.pt')
            cand_occ_64_2000 = Path('artifacts/kaggle-kenney-64-occupancy-dit/occupancy-dit-64-2000.pt')
            cand_occ_64_1500 = Path('artifacts/kaggle-kenney-64-occupancy-dit/occupancy-dit-64-1500.pt')
            cand_occ_64 = Path('artifacts/kaggle-kenney-64-occupancy-dit/occupancy-dit-64-1000.pt')
            cand_occ_64_500 = Path('artifacts/kaggle-kenney-64-occupancy-dit/occupancy-dit-64-500.pt')
            if cand_occ_64_4000.exists():
                occ_path = cand_occ_64_4000
            elif cand_occ_64_3000.exists():
                occ_path = cand_occ_64_3000
            elif cand_occ_64_2000.exists():
                occ_path = cand_occ_64_2000
            elif cand_occ_64_1500.exists():
                occ_path = cand_occ_64_1500
            elif cand_occ_64.exists():
                occ_path = cand_occ_64
            elif cand_occ_64_500.exists():
                occ_path = cand_occ_64_500
            else:
                occ_path = Path('artifacts/kaggle-kenney-occupancy-dit-v37/occupancy-dit-600.pt')
        else:
            occ_path = Path('artifacts/kaggle-kenney-occupancy-dit-v37/occupancy-dit-600.pt')

    if not occ_path.exists():
        raise FileNotFoundError(f"Stage 1 Occupancy DiT checkpoint not found: {occ_path}")

    vae_path = args.vae_checkpoint
    if vae_path is None:
        if resolution == 64:
            cand_vae_64 = Path('artifacts/kaggle-kenney-corpus-64-training/triton-kenney-corpus-64-1200.pt')
            if cand_vae_64.exists():
                vae_path = cand_vae_64
            else:
                vae_path = Path('artifacts/kaggle-kenney-corpus-training-v35/triton-kenney-corpus-1900.pt')
        else:
            vae_path = Path('artifacts/kaggle-kenney-corpus-training-v35/triton-kenney-corpus-1900.pt')

    if not vae_path.exists():
        raise FileNotFoundError(f"Shape SC-VAE checkpoint not found: {vae_path}")

    # Load Stage 1 Occupancy DiT
    occ_data = torch.load(occ_path, map_location=device, weights_only=False)
    from .flow import SparseFlowDiT
    from .conditioning import PromptConditioner

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

    # Load Stage 2 Geometry DiT
    geom_data = torch.load(geom_path, map_location=device, weights_only=False)
    geom_channels = int(geom_data.get('latent_channels', 16))
    geom_blocks = [int(k.split('.')[1]) for k in geom_data['model'] if k.startswith('blocks.')]
    geom_depth = geom_data.get('depth', max(geom_blocks) + 1 if geom_blocks else 4)
    geom_width = geom_data.get('width', geom_data['model']['input.weight'].shape[0] if 'input.weight' in geom_data['model'] else 96)
    geom_heads = geom_data.get('heads', max(1, geom_width // 24))
    geom_model = SparseFlowDiT(channels=geom_channels, condition_channels=32, width=geom_width, heads=geom_heads, depth=geom_depth, patch_size=64).to(device)
    geom_model.load_state_dict(geom_data['model'])

    # Auto-resolve Stage 3 Material checkpoints if requested or provided
    mat_flow_path = args.material_flow_checkpoint
    mat_vae_path = args.material_checkpoint
    if args.material or mat_flow_path is not None or mat_vae_path is not None:
        if mat_flow_path is None:
            cand_mf_300 = Path('artifacts/kaggle-material-flow-dit-v1/material-flow-dit-300.pt')
            cand_mf_600 = Path('artifacts/kaggle-material-flow-dit-v1/material-flow-dit-600.pt')
            if cand_mf_300.exists():
                mat_flow_path = cand_mf_300
            elif cand_mf_600.exists():
                mat_flow_path = cand_mf_600
        if mat_vae_path is None:
            cand_mvae = Path('artifacts/kaggle-material-multiasset-32-v1/material-multiasset-300.pt')
            if cand_mvae.exists():
                mat_vae_path = cand_mvae

    # Load Stage 3 Material Flow DiT if resolved
    mat_flow_model = None
    if mat_flow_path is not None and mat_flow_path.exists():
        mat_flow_data = torch.load(mat_flow_path, map_location=device, weights_only=False)
        m_channels = int(mat_flow_data['config'].get('channels', 16))
        m_cond_dim = int(mat_flow_data['config'].get('condition_channels', 32))
        m_shape_dim = int(mat_flow_data['config'].get('shape_channels', 16))
        m_width = int(mat_flow_data['config'].get('width', 96))
        m_heads = int(mat_flow_data['config'].get('heads', 4))
        m_depth = int(mat_flow_data['config'].get('depth', 4))
        m_patch_size = int(mat_flow_data['config'].get('patch_size', 64))
        mat_flow_model = SparseFlowDiT(
            channels=m_channels,
            condition_channels=m_cond_dim,
            width=m_width,
            heads=m_heads,
            depth=m_depth,
            patch_size=m_patch_size,
            shape_channels=m_shape_dim
        ).to(device)
        mat_flow_model.load_state_dict(mat_flow_data['model'])
        print(f"Loaded Stage 3 Material Flow DiT from {mat_flow_path}")

    # Auto-resolve Stage 2 Rig checkpoints if requested or provided
    rig_flow_path = args.rig_flow_checkpoint
    skinae_path = args.skinae_checkpoint
    if args.rig or rig_flow_path is not None or skinae_path is not None:
        if rig_flow_path is None:
            cand_rf_v4_2500 = Path('artifacts/kaggle-rig-flow-dit-v4/rig-flow-dit-2500.pt')
            cand_rf_v4_2000 = Path('artifacts/kaggle-rig-flow-dit-v4/rig-flow-dit-2000.pt')
            cand_rf_v4_1500 = Path('artifacts/kaggle-rig-flow-dit-v4/rig-flow-dit-1500.pt')
            cand_rf_v3_1500 = Path('artifacts/kaggle-rig-flow-dit-v3/rig-flow-dit-1500.pt')
            cand_rf_v3_2000 = Path('artifacts/kaggle-rig-flow-dit-v3/rig-flow-dit-2000.pt')
            cand_rf_v3_1000 = Path('artifacts/kaggle-rig-flow-dit-v3/rig-flow-dit-1000.pt')
            cand_rf_1000 = Path('artifacts/kaggle-rig-flow-dit-v2/rig-flow-dit-1000.pt')
            cand_rf_250 = Path('artifacts/kaggle-rig-flow-dit-v2/rig-flow-dit-250.pt')
            if cand_rf_v4_2500.exists():
                rig_flow_path = cand_rf_v4_2500
            elif cand_rf_v4_2000.exists():
                rig_flow_path = cand_rf_v4_2000
            elif cand_rf_v4_1500.exists():
                rig_flow_path = cand_rf_v4_1500
            elif cand_rf_v3_1500.exists():
                rig_flow_path = cand_rf_v3_1500
            elif cand_rf_v3_2000.exists():
                rig_flow_path = cand_rf_v3_2000
            elif cand_rf_v3_1000.exists():
                rig_flow_path = cand_rf_v3_1000
            elif cand_rf_1000.exists():
                rig_flow_path = cand_rf_1000
            elif cand_rf_250.exists():
                rig_flow_path = cand_rf_250
        if skinae_path is None:
            cand_skin = Path('artifacts/kaggle-skin-family-bind-select-v8/loss-1.0/skin-pose-pilot.pt')
            cand_skin_v1 = Path('artifacts/kaggle-skin-pose-v1/skin-pose-pilot.pt')
            if cand_skin.exists():
                skinae_path = cand_skin
            elif cand_skin_v1.exists():
                skinae_path = cand_skin_v1

    # Load Stage 2 Rig Flow DiT if resolved
    rig_flow_model = None
    if rig_flow_path is not None and rig_flow_path.exists():
        rig_flow_data = torch.load(rig_flow_path, map_location=device, weights_only=False)
        r_channels = int(rig_flow_data['config'].get('channels', 10))
        r_cond_dim = int(rig_flow_data['config'].get('condition_channels', 32))
        r_shape_dim = int(rig_flow_data['config'].get('shape_channels', 7))
        r_width = int(rig_flow_data['config'].get('width', 96))
        r_heads = int(rig_flow_data['config'].get('heads', 4))
        r_depth = int(rig_flow_data['config'].get('depth', 4))
        r_patch_size = int(rig_flow_data['config'].get('patch_size', 64))
        rig_flow_model = SparseFlowDiT(
            channels=r_channels,
            condition_channels=r_cond_dim,
            width=r_width,
            heads=r_heads,
            depth=r_depth,
            patch_size=r_patch_size,
            shape_channels=r_shape_dim
        ).to(device)
        rig_flow_model.load_state_dict(rig_flow_data['model'])
        print(f"Loaded Stage 2 Rig Flow DiT from {rig_flow_path} (width={r_width}, depth={r_depth})")

    # Load SkinAE if resolved
    skinae_model = None
    if skinae_path is not None and skinae_path.exists():
        from .skin_ae import SkinAE
        skin_data = torch.load(skinae_path, map_location=device, weights_only=True)
        skinae_model = SkinAE(**skin_data['config']).to(device)
        skinae_model.load_state_dict(skin_data['model'])
        skinae_model.freeze()
        print(f"Loaded SkinAE from {skinae_path}")

    # Load Shape SC-VAE
    from .sc_vae import SparseCompressionVAE
    vae_data = torch.load(vae_path, map_location=device, weights_only=False)
    cfg = {k: v for k, v in vae_data['config'].items() if k in ('mode', 'channels', 'latent_channels', 'levels', 'downsample_block', 'upsample_block', 'convolution_backend')}
    if 'input.weight' in vae_data['model']:
        enc_blocks = max(int(k.split('.')[1]) for k in vae_data['model'] if k.startswith('encoder_blocks.'))
        channels = tuple(vae_data['model'][f'encoder_blocks.{i}.scale'].shape[0] for i in range(enc_blocks + 1))
        cfg['channels'] = channels
    if not torch.cuda.is_available():
        cfg['convolution_backend'] = 'reference'
    vae_model = SparseCompressionVAE(**cfg).to(device)
    vae_model.load_state_dict(vae_data['model'])

    # Load Material SC-VAE if resolved
    mat_vae_model = None
    if mat_vae_path is not None and mat_vae_path.exists():
        mat_data = torch.load(mat_vae_path, map_location=device, weights_only=False)
        mcfg = {k: v for k, v in mat_data['config'].items() if k in ('mode', 'channels', 'latent_channels', 'levels', 'downsample_block', 'upsample_block', 'convolution_backend')}
        if 'input.weight' in mat_data['model']:
            m_enc_blocks = max(int(k.split('.')[1]) for k in mat_data['model'] if k.startswith('encoder_blocks.'))
            m_channels = tuple(mat_data['model'][f'encoder_blocks.{i}.scale'].shape[0] for i in range(m_enc_blocks + 1))
            mcfg['channels'] = m_channels
        if not torch.cuda.is_available():
            mcfg['convolution_backend'] = 'reference'
        mat_vae_model = SparseCompressionVAE(**mcfg).to(device)
        mat_vae_model.load_state_dict(mat_data['model'])
        print(f"Loaded Material SC-VAE from {mat_vae_path}")

    resolution = args.resolution
    if resolution is None:
        if geom_data.get('resolution') == 64 or '64' in str(geom_path) or '64' in str(vae_path):
            resolution = 64
        else:
            resolution = 32
    latent_grid_size = max(1, resolution // 16)
    grid_size = (resolution, resolution, resolution)
    coarse_grid_size = int(occ_data.get('grid_size', 16 if resolution == 32 else latent_grid_size))
    print(f"Configured generation at {resolution}³ resolution (coarse_grid_size={coarse_grid_size}, latent_grid_size={latent_grid_size}, grid_size={grid_size})")

    print(f"Generating 3D model for prompt: \"{args.prompt}\"...")
    active_coords, shape_latents, mat_latents = generate_latents_from_prompt(
        args.prompt, occ_model, geom_model, conditioner,
        material_model=mat_flow_model,
        coarse_grid_size=coarse_grid_size,
        latent_grid_size=latent_grid_size,
        occupancy_steps=args.steps, geometry_steps=args.steps, material_steps=args.steps,
        guidance=args.guidance, occupancy_guidance=args.occupancy_guidance, occupancy_threshold=args.threshold,
        device=device, seed=args.seed, return_material=True, method=args.method
    )
    mat_desc = f"and {len(mat_latents)} material latents" if mat_latents is not None else ""
    print(f"Stage 1, 2 & 3 generated {len(active_coords)} active coarse latents {mat_desc}")

    tokens = conditioner.encode_text(args.prompt).to(device)
    decode_th = args.decode_threshold
    if decode_th is None:
        decode_th = 0.46 if resolution == 64 else 0.42
    asset = generate_asset_from_latents(
        active_coords, shape_latents, vae_model,
        grid_size=grid_size,
        threshold=decode_th, material_latents=mat_latents, material_vae_model=mat_vae_model,
        rig_model=rig_flow_model, tokens=tokens, rig_steps=args.steps, guidance=args.rig_guidance
    )
    has_pbr = asset.pbr is not None
    has_rig = asset.skeleton_field is not None and asset.skin_embedding is not None
    print(f"Decoded VoxelAsset with {len(asset.coords)} surface voxels (PBR: {has_pbr}, Rig: {has_rig})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == '.t3d':
        args.output.write_bytes(dumps(asset))
        print(f"Saved .t3d container to {args.output}")
    else:
        # Export as GLB
        vertices, faces = voxels_to_mesh(asset, resolve_nonmanifold=True)
        skeleton = None
        weights = None
        if has_rig and skinae_model is not None:
            from .articulation import infer_skeleton_from_field, robust_infer_skeleton, sample_sparse_embeddings
            from .glb import limit_skin_influences
            centers = asset.aabb[0] + (asset.coords.astype(float) + 0.5) * asset.voxel_size
            tolerance = max(0.08, float(np.min(asset.voxel_size)) * 2.5)
            skeleton = robust_infer_skeleton(centers, asset.skeleton_field, asset.voxel_size, asset.aabb)
            if skeleton is None:
                try:
                    skeleton = infer_skeleton_from_field(centers, asset.skeleton_field, tolerance=tolerance, max_joints=1024, min_votes=3)
                except Exception:
                    skeleton = None
            if skeleton is not None:
                j_t = torch.as_tensor(skeleton.joints, device=device, dtype=torch.float32)
                p_t = torch.as_tensor(skeleton.parents, device=device, dtype=torch.int64)
                with torch.no_grad():
                    joint_embeddings = skinae_model.encode_joints(j_t, p_t)
                    v_tensor = torch.as_tensor(vertices, device=device, dtype=torch.float32)
                    vertex_embeddings, supported = sample_sparse_embeddings(
                        v_tensor, asset.coords, asset.skin_embedding,
                        grid_size=asset.grid_size, aabb=asset.aabb
                    )
                    if not bool(supported.all()):
                        missing = ~supported
                        missing_idx = missing.nonzero(as_tuple=False).squeeze(-1)
                        c_np = asset.aabb[0] + (asset.coords.astype(float) + 0.5) * asset.voxel_size
                        c_tensor = torch.as_tensor(c_np, device=device, dtype=torch.float32)
                        v_missing = v_tensor[missing_idx]
                        near_list = []
                        chunk_size = 512
                        for start in range(0, len(v_missing), chunk_size):
                            chunk = v_missing[start:start + chunk_size]
                            dist = torch.cdist(chunk, c_tensor)
                            near_list.append(torch.argmin(dist, dim=1))
                        near_idx = torch.cat(near_list).cpu().numpy()
                        v_emb_np = vertex_embeddings.cpu().numpy()
                        v_emb_np[missing.cpu().numpy()] = asset.skin_embedding[near_idx]
                        vertex_embeddings = torch.as_tensor(v_emb_np, device=device, dtype=torch.float32)
                    logits = skinae_model.decode_logits(vertex_embeddings, joint_embeddings)
                    weights = torch.softmax(logits, dim=-1).cpu().numpy()
                weights, _ = limit_skin_influences(weights, max_influences=16)

        from .glb import pack_static_glb, validate_glb_semantics
        glb_bytes = pack_static_glb(vertices, faces, pbr=asset.pbr, skin_weights=weights, skeleton=skeleton)
        validation_report = validate_glb_semantics(glb_bytes)
        if skeleton is not None:
            print(f"Attached learned rig ({len(skeleton.joints)} joints, {weights.shape[1]} skin influences) to GLB")
        if asset.pbr is not None:
            print(f"Validated 6-channel PBR semantics (six_channels_preserved={validation_report['six_channels_preserved']})")

        desc_parts = []
        if has_rig and skinae_model is not None:
            desc_parts.append("rigged")
        if has_pbr:
            desc_parts.append("textured PBR")
        desc = " ".join(desc_parts) if desc_parts else "static"
        args.output.write_bytes(glb_bytes)
        print(f"Exported production {desc} GLB ({len(vertices)} verts, {len(faces)} faces) to {args.output}")


if __name__ == '__main__':
    main()
