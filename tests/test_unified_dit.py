"""Unit tests for UnifiedFlowDiT token packing, unpacking, forward pass, and sampling."""
import torch
import pytest

from t3d.unified_dit import (
    UnifiedFlowDiT,
    get_grid_cells,
    pack_unified_token,
    unpack_unified_token,
    sample_logit_normal_time,
    sample_unified_flow,
    GRID_SIZE,
    TOTAL_CELLS,
)


def test_grid_cells_shape():
    cells = get_grid_cells()
    assert cells.shape == (TOTAL_CELLS, 3)
    assert cells.min() == 0
    assert cells.max() == GRID_SIZE - 1


def test_pack_and_unpack_unified_token():
    device = torch.device('cpu')
    K = 10
    latent_dim = 16
    coords = torch.tensor([
        [0, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 2, 2, 2],
        [0, 3, 3, 3],
        [0, 0, 1, 2],
        [0, 1, 2, 3],
        [0, 2, 3, 0],
        [0, 3, 0, 1],
        [0, 1, 0, 3],
        [0, 2, 1, 0],
    ], dtype=torch.long)
    latents = torch.randn((K, latent_dim))
    mean_stats = torch.zeros(latent_dim)
    std_stats = torch.ones(latent_dim) * 2.0

    target, occ_mask, cells = pack_unified_token(coords, latents, mean_stats, std_stats, device=device)
    assert target.shape == (64, 1 + latent_dim)
    assert occ_mask.sum() == K
    # Check that empty cells have -1.0 occupancy and 0.0 latents
    empty_mask = ~occ_mask
    assert torch.all(target[empty_mask, 0] == -1.0)
    assert torch.all(target[empty_mask, 1:] == 0.0)

    # Unpack
    pred_clean = target.clone()
    bcoords, out_latents, out_mask = unpack_unified_token(pred_clean, mean_stats, std_stats, occ_threshold=0.0)
    assert len(bcoords) == K
    assert out_mask.sum() == K

    # In target, tokens are stored at flat grid indices
    c = coords[:, 1:]
    flat = (c[:, 0] * 4 + c[:, 1]) * 4 + c[:, 2]
    sort_idx = torch.argsort(flat)
    expected_latents = latents[sort_idx]
    assert torch.allclose(out_latents, expected_latents, atol=1e-5)


def test_logit_normal_time_sampling():
    t = sample_logit_normal_time((1000,), torch.device('cpu'))
    assert torch.all(t > 0.0) and torch.all(t < 1.0)
    # Check that mid values (0.2 to 0.8) have high representation
    mid_ratio = ((t >= 0.2) & (t <= 0.8)).float().mean()
    assert mid_ratio > 0.5


def test_unified_dit_forward_and_sample():
    torch.manual_seed(42)
    latent_dim = 16
    cond_dim = 32
    # head_dim must be divisible by 6 (e.g. width=144, heads=6 -> head_dim=24)
    model = UnifiedFlowDiT(
        latent_channels=latent_dim,
        condition_channels=cond_dim,
        width=144,
        heads=6,
        depth=2,
        patch_size=64,
    )
    cells = get_grid_cells()
    x = torch.randn((64, 1 + latent_dim))
    time = torch.tensor(0.5)
    condition = torch.randn((7, cond_dim))

    # Test forward
    out = model(x, cells, time, condition)
    assert out.shape == (64, 1 + latent_dim)

    # Test unconditional forward
    out_uncond = model(x, cells, time, None)
    assert out_uncond.shape == (64, 1 + latent_dim)

    # Test sampling
    x_sample = sample_unified_flow(model, condition, coords=cells, steps=2, guidance=2.0, method='euler')
    assert x_sample.shape == (64, 1 + latent_dim)
