"""Compute channel-wise mean and std for SC-VAE latents across training split."""
import json
from pathlib import Path
import numpy as np
import torch
from t3d.dataset import AssetDataset, batch_arrays
from t3d.sc_vae import SparseCompressionVAE

def main():
    device = 'cpu'
    manifest_path = Path('artifacts/kenney-corpus-64-id-expanded/manifest.json')
    if not manifest_path.exists():
        manifest_path = Path('artifacts/kenney-corpus-64/manifest.json')
    vae_path = Path('artifacts/kaggle-kenney-corpus-64-training/triton-kenney-corpus-64-1200.pt')

    print(f"Loading VAE from {vae_path}...")
    vae_data = torch.load(vae_path, map_location=device, weights_only=False)
    cfg = {k: v for k, v in vae_data['config'].items() if k in ('mode', 'channels', 'latent_channels', 'levels', 'downsample_block', 'upsample_block')}
    n = max(int(k.split('.')[1]) for k in vae_data['model'] if k.startswith('encoder_blocks.'))
    cfg['channels'] = tuple(vae_data['model'][f'encoder_blocks.{i}.scale'].shape[0] for i in range(n + 1))
    cfg['convolution_backend'] = 'reference'
    vae = SparseCompressionVAE(**cfg).eval()
    vae.load_state_dict(vae_data['model'])

    ds = AssetDataset(str(manifest_path), split='train')
    print(f"Loaded train dataset with {len(ds)} assets.")

    all_means = []
    total_active_cells = 0

    print("Encoding assets to collect latent distribution...")
    with torch.no_grad():
        for i in range(len(ds)):
            asset = ds[i]
            coords, feats = batch_arrays([asset], 'shape')
            post = vae.encode(torch.from_numpy(coords), torch.from_numpy(feats))
            all_means.append(post.mean.cpu())
            total_active_cells += len(post.mean)
            if (i + 1) % 200 == 0 or i == len(ds) - 1:
                print(f"  Processed {i + 1}/{len(ds)} assets ({total_active_cells} active cells)...")

    cat_latents = torch.cat(all_means, dim=0) # [Total_Active, 16]
    mean = cat_latents.mean(dim=0).tolist()
    std = cat_latents.std(dim=0).tolist()

    stats = {
        'num_assets': len(ds),
        'total_active_cells': total_active_cells,
        'latent_channels': len(mean),
        'mean': [round(x, 6) for x in mean],
        'std': [round(x, 6) for x in std],
    }

    out_path = Path('artifacts/latent_stats_64.json')
    out_path.write_text(json.dumps(stats, indent=2) + '\n')
    print(f"Saved stats to {out_path}:")
    print(f"Mean: {stats['mean']}")
    print(f"Std:  {stats['std']}")

if __name__ == '__main__':
    main()
