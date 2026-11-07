"""Diagnose error decomposition between Stage 1 coarse occupancy and Stage 2 geometry latents."""
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from t3d.conditioning import PromptConditioner
from t3d.flow import SparseFlowDiT, sample_flow
from t3d.sc_vae import SparseCompressionVAE
from t3d.container import load
from t3d.dataset import batch_arrays
from t3d.extraction import voxels_to_mesh
from t3d.metrics import surface_comparison
from t3d.schema import VoxelAsset
from tools.generate_corpus_prompts import id_to_prompt

def main():
    device = 'cpu'
    
    # 1. Load SC-VAE
    vae_path = Path('artifacts/kaggle-kenney-corpus-64-training/triton-kenney-corpus-64-1200.pt')
    vae_data = torch.load(vae_path, map_location=device, weights_only=False)
    cfg = {k: v for k, v in vae_data['config'].items() if k in ('mode', 'channels', 'latent_channels', 'levels', 'downsample_block', 'upsample_block', 'convolution_backend')}
    if 'input.weight' in vae_data['model']:
        enc_blocks = max(int(k.split('.')[1]) for k in vae_data['model'] if k.startswith('encoder_blocks.'))
        channels = tuple(vae_data['model'][f'encoder_blocks.{i}.scale'].shape[0] for i in range(enc_blocks + 1))
        cfg['channels'] = channels
    cfg['convolution_backend'] = 'reference'
    vae_model = SparseCompressionVAE(**cfg).to(device).eval()
    vae_model.load_state_dict(vae_data['model'])

    # 2. Load Conditioner
    occ_path = Path('artifacts/kaggle-kenney-64-occupancy-dit/occupancy-dit-64-2000.pt')
    occ_data = torch.load(occ_path, map_location=device, weights_only=False)
    conditioner = PromptConditioner.from_pretrained(
        image_model='facebook/dinov2-small',
        image_revision='ed25f3a31f01632728cabb09d1542f84ab7b0056',
        text_model='openai/clip-vit-base-patch32',
        text_revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
        output_channels=32
    ).to(device).eval()
    conditioner.text_projection.load_state_dict(occ_data['text_projection'])

    # 3. Load Stage 1 DiT
    occ_blocks = [int(k.split('.')[1]) for k in occ_data['model'] if k.startswith('blocks.')]
    occ_depth = max(occ_blocks) + 1 if occ_blocks else 3
    occ_width = occ_data['model']['input.weight'].shape[0] if 'input.weight' in occ_data['model'] else 96
    occ_heads = max(1, occ_width // 24)
    occ_model = SparseFlowDiT(channels=1, condition_channels=32, width=occ_width, heads=occ_heads, depth=occ_depth, patch_size=64).to(device).eval()
    occ_model.load_state_dict(occ_data['model'])

    # 4. Load Stage 2 DiT
    geom_path = Path('artifacts/kaggle-kenney-64-geometry-dit/geometry-dit-64-4000.pt')
    geom_data = torch.load(geom_path, map_location=device, weights_only=False)
    geom_channels = int(geom_data.get('latent_channels', 16))
    geom_blocks = [int(k.split('.')[1]) for k in geom_data['model'] if k.startswith('blocks.')]
    geom_depth = geom_data.get('depth', max(geom_blocks) + 1 if geom_blocks else 4)
    geom_width = geom_data.get('width', geom_data['model']['input.weight'].shape[0] if 'input.weight' in geom_data['model'] else 96)
    geom_heads = geom_data.get('heads', max(1, geom_width // 24))
    geom_model = SparseFlowDiT(channels=geom_channels, condition_channels=32, width=geom_width, heads=geom_heads, depth=geom_depth, patch_size=64).to(device).eval()
    geom_model.load_state_dict(geom_data['model'])

    # Test on boat-row-large
    asset_id = 'kenney-pirate-kit-boat-row-large'
    t3d_file = Path(f'artifacts/kenney-corpus-64/{asset_id}.t3d')
    raw = load(t3d_file)
    gt_asset = raw.asset
    gt_v, gt_f = voxels_to_mesh(gt_asset)
    prompt = id_to_prompt(asset_id)

    # Encode GT to get GT coords and GT latents
    c, f = batch_arrays([gt_asset], 'shape')
    with torch.no_grad():
        post = vae_model.encode(torch.from_numpy(c), torch.from_numpy(f))
        gt_coords = post.coords[:, 1:]
        gt_latents = post.mean
        tokens = conditioner.encode_text(prompt)

        # 1. Oracle: GT coords + GT latents
        batched_gt_coords = post.coords
        out1, raw1, _ = vae_model.decode(batched_gt_coords, gt_latents, threshold=0.50)
        v1 = (out1[:, 1:] < 64).all(dim=1)
        a1 = VoxelAsset(out1[v1, 1:].numpy().astype(int), raw1[v1, :3].sigmoid().numpy(), (raw1[v1, 3:6].sigmoid() > 0.5).numpy(), (64,64,64), gt_asset.aabb, split_weight=raw1[v1, 6:].numpy())
        mesh_v1, mesh_f1 = voxels_to_mesh(a1, resolve_nonmanifold=True)
        comp1 = surface_comparison(gt_v, gt_f, mesh_v1, mesh_f1, samples=512, tolerance=0.01)
        print(f"[Case 1: Oracle GT Coords + GT Latents] Dist: {comp1['mean_distance']:.5f} | F-score: {comp1['fscore']:.4f} | Tris: {len(mesh_f1)}")

        # 2. GT Coords + Generated Latents (Isolates Stage 2 Error)
        gen = torch.Generator().manual_seed(42)
        noise = torch.randn_like(gt_latents)
        pred_latents = sample_flow(geom_model, noise, gt_coords, tokens, steps=15, guidance=2.5)
        out2, raw2, _ = vae_model.decode(batched_gt_coords, pred_latents, threshold=0.50)
        v2 = (out2[:, 1:] < 64).all(dim=1)
        a2 = VoxelAsset(out2[v2, 1:].numpy().astype(int), raw2[v2, :3].sigmoid().numpy(), (raw2[v2, 3:6].sigmoid() > 0.5).numpy(), (64,64,64), gt_asset.aabb, split_weight=raw2[v2, 6:].numpy())
        mesh_v2, mesh_f2 = voxels_to_mesh(a2, resolve_nonmanifold=True)
        comp2 = surface_comparison(gt_v, gt_f, mesh_v2, mesh_f2, samples=512, tolerance=0.01)
        print(f"[Case 2: GT Coords + Generated Latents] Dist: {comp2['mean_distance']:.5f} | F-score: {comp2['fscore']:.4f} | Tris: {len(mesh_f2)}")

        # 3. Stage 1 Predicted Coords + GT Latents (where available, 0 elsewhere)
        from t3d.generate import generate_latents_from_prompt
        pred_coords, _ = generate_latents_from_prompt(
            prompt, occ_model, geom_model, conditioner,
            coarse_grid_size=4, latent_grid_size=4,
            occupancy_steps=12, geometry_steps=12,
            guidance=2.5, occupancy_guidance=1.0, occupancy_threshold=0.50,
            device=device, seed=42
        )
        print(f"Stage 1 predicted {len(pred_coords)} coords (GT coords: {len(gt_coords)})")

        # For predicted coords, look up GT latents if coord matches, else zero
        lat_dict = {tuple(c.tolist()): l for c, l in zip(gt_coords, gt_latents)}
        interp_latents = torch.stack([lat_dict.get(tuple(pc.tolist()), torch.zeros(16)) for pc in pred_coords])
        batched_pred_coords = torch.cat([torch.zeros(len(pred_coords), 1, dtype=torch.long), pred_coords.long()], dim=-1)
        out3, raw3, _ = vae_model.decode(batched_pred_coords, interp_latents, threshold=0.50)
        if len(out3) > 0:
            v3 = (out3[:, 1:] < 64).all(dim=1)
            a3 = VoxelAsset(out3[v3, 1:].numpy().astype(int), raw3[v3, :3].sigmoid().numpy(), (raw3[v3, 3:6].sigmoid() > 0.5).numpy(), (64,64,64), gt_asset.aabb, split_weight=raw3[v3, 6:].numpy())
            mesh_v3, mesh_f3 = voxels_to_mesh(a3, resolve_nonmanifold=True)
            comp3 = surface_comparison(gt_v, gt_f, mesh_v3, mesh_f3, samples=512, tolerance=0.01)
            print(f"[Case 3: Pred Coords + Oracle Latents] Dist: {comp3['mean_distance']:.5f} | F-score: {comp3['fscore']:.4f} | Tris: {len(mesh_f3)}")

        # 4. End-to-End: Pred Coords + Pred Latents
        noise_pred = torch.randn(len(pred_coords), 16)
        pred_latents_full = sample_flow(geom_model, noise_pred, pred_coords, tokens, steps=15, guidance=2.5)
        out4, raw4, _ = vae_model.decode(batched_pred_coords, pred_latents_full, threshold=0.50)
        v4 = (out4[:, 1:] < 64).all(dim=1)
        a4 = VoxelAsset(out4[v4, 1:].numpy().astype(int), raw4[v4, :3].sigmoid().numpy(), (raw4[v4, 3:6].sigmoid() > 0.5).numpy(), (64,64,64), gt_asset.aabb, split_weight=raw4[v4, 6:].numpy())
        mesh_v4, mesh_f4 = voxels_to_mesh(a4, resolve_nonmanifold=True)
        comp4 = surface_comparison(gt_v, gt_f, mesh_v4, mesh_f4, samples=512, tolerance=0.01)
        print(f"[Case 4: Pred Coords + Pred Latents (End-to-End)] Dist: {comp4['mean_distance']:.5f} | F-score: {comp4['fscore']:.4f} | Tris: {len(mesh_f4)}")

if __name__ == '__main__':
    main()
