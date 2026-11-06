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
    coarse_grid_size: int = 16,
    occupancy_steps: int = 25,
    geometry_steps: int = 25,
    guidance: float = 3.0,
    occupancy_threshold: float = 0.5,
    device: str = 'cuda',
    seed: int = 42,
    orders: list | None = None
):
    """Generate coarse active coordinates and shape latents from a text prompt."""
    occupancy_model.eval()
    geometry_model.eval()
    conditioner.eval()

    tokens = conditioner.encode_text(prompt).to(device)

    # Stage 1: Coarse Occupancy Flow
    grid_coords = torch.cartesian_prod(*(torch.arange(coarse_grid_size, device=device) for _ in range(3)))
    gen = torch.Generator(device=device).manual_seed(seed)
    occ_noise = torch.randn((len(grid_coords), 1), device=device, generator=gen)

    pred_occ = sample_flow(
        occupancy_model, occ_noise, grid_coords, tokens,
        steps=occupancy_steps, guidance=guidance, orders=orders
    )
    active_mask = pred_occ[:, 0] > occupancy_threshold
    if not active_mask.any():
        # Fallback to top-k cells if thresholding yields empty volume
        topk = torch.topk(pred_occ[:, 0], k=min(64, len(pred_occ))).indices
        active_mask[topk] = True

    active_coords = grid_coords[active_mask]
    # Downsample active coarse 16³ coordinates to the 2³ SC-VAE latent grid (16 // 8 = 2)
    latent_scale = max(1, coarse_grid_size // 2)
    latent_coords = torch.unique(torch.div(active_coords, latent_scale, rounding_mode='floor'), dim=0)

    # Stage 2: Latent Shape Flow at active latent coordinates
    geom_noise = torch.randn((len(latent_coords), geometry_model.channels), device=device, generator=gen)
    shape_latents = sample_flow(
        geometry_model, geom_noise, latent_coords, tokens,
        steps=geometry_steps, guidance=guidance
    )

    return latent_coords, shape_latents


@torch.no_grad()
def generate_asset_from_latents(
    active_coords: torch.Tensor,
    shape_latents: torch.Tensor,
    vae_model,
    *,
    grid_size: tuple[int, int, int] = (32, 32, 32),
    aabb: list | None = None,
    threshold: float = 0.5
) -> VoxelAsset:
    """Decode shape latents into a verified VoxelAsset."""
    if aabb is None:
        aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]

    vae_model.eval()
    # Batch dimension 0 prepended for decoding
    batched_coords = torch.cat([torch.zeros((len(active_coords), 1), device=active_coords.device, dtype=active_coords.dtype), active_coords], dim=1)

    out, raw, _ = vae_model.decode(batched_coords, shape_latents, threshold=threshold)
    if len(out) == 0:
        raise ValueError('VAE decoding produced zero active surface voxels')

    valid = (out[:, 1:] < out.new_tensor(grid_size)).all(dim=1)
    out, raw = out[valid], raw[valid]

    coords_np = out[:, 1:].cpu().numpy().astype(int)
    dual_np = raw[:, :3].sigmoid().cpu().numpy()
    flags_np = (raw[:, 3:6].sigmoid() > threshold).cpu().numpy()
    split_np = raw[:, 6:].cpu().numpy()

    asset = VoxelAsset(coords_np, dual_np, flags_np, grid_size, np.asarray(aabb, dtype=np.float32), split_weight=split_np)
    return asset.canonical()


def main():
    parser = argparse.ArgumentParser(description='Generate 3D models from text prompts using T3D flow models')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt describing the desired 3D object')
    parser.add_argument('--occupancy-checkpoint', type=Path, required=True, help='Path to Stage 1 Occupancy DiT checkpoint')
    parser.add_argument('--geometry-checkpoint', type=Path, required=True, help='Path to Stage 2 Geometry DiT checkpoint')
    parser.add_argument('--vae-checkpoint', type=Path, required=True, help='Path to SC-VAE checkpoint')
    parser.add_argument('--output', type=Path, required=True, help='Output path (.glb or .t3d)')
    parser.add_argument('--steps', type=int, default=25, help='Flow matching Euler sampling steps')
    parser.add_argument('--guidance', type=float, default=3.0, help='Classifier-free guidance weight')
    parser.add_argument('--threshold', type=float, default=0.5, help='Occupancy decision threshold')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Stage 1 Occupancy DiT
    occ_data = torch.load(args.occupancy_checkpoint, map_location=device, weights_only=False)
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

    occ_model = SparseFlowDiT(channels=1, condition_channels=32, width=96, heads=4, depth=3, patch_size=64).to(device)
    occ_model.load_state_dict(occ_data['model'])

    # Load Stage 2 Geometry DiT
    geom_data = torch.load(args.geometry_checkpoint, map_location=device, weights_only=False)
    geom_channels = int(geom_data.get('latent_channels', 16))
    geom_model = SparseFlowDiT(channels=geom_channels, condition_channels=32, width=96, heads=4, depth=4, patch_size=64).to(device)
    geom_model.load_state_dict(geom_data['model'])

    # Load SC-VAE
    from .sc_vae import SparseCompressionVAE
    vae_data = torch.load(args.vae_checkpoint, map_location=device, weights_only=False)
    cfg = {k: v for k, v in vae_data['config'].items() if k in ('mode', 'channels', 'latent_channels', 'levels', 'downsample_block', 'upsample_block', 'convolution_backend')}
    if not torch.cuda.is_available():
        cfg['convolution_backend'] = 'reference'
    vae_model = SparseCompressionVAE(**cfg).to(device)
    vae_model.load_state_dict(vae_data['model'])

    print(f"Generating 3D model for prompt: \"{args.prompt}\"...")
    active_coords, shape_latents = generate_latents_from_prompt(
        args.prompt, occ_model, geom_model, conditioner,
        occupancy_steps=args.steps, geometry_steps=args.steps,
        guidance=args.guidance, occupancy_threshold=args.threshold,
        device=device, seed=args.seed
    )
    print(f"Stage 1 & 2 generated {len(active_coords)} active coarse latents")

    asset = generate_asset_from_latents(active_coords, shape_latents, vae_model, threshold=args.threshold)
    print(f"Decoded VoxelAsset with {len(asset.coords)} surface voxels")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == '.t3d':
        args.output.write_bytes(dumps(asset))
        print(f"Saved .t3d container to {args.output}")
    else:
        # Export as GLB
        vertices, faces = voxels_to_mesh(asset)
        from .glb import pack_static_glb
        glb_bytes = pack_static_glb(vertices, faces)
        args.output.write_bytes(glb_bytes)
        print(f"Exported production GLB ({len(vertices)} verts, {len(faces)} faces) to {args.output}")


if __name__ == '__main__':
    main()
