"""Unified Single-Stage Flow DiT for joint occupancy and geometry generation on full 4³ grid.

Eliminates the Stage 1 -> Stage 2 cascading bottleneck by predicting both coarse cell occupancy
o_i in {-1, +1} and continuous geometry latents z_i in R^C simultaneously on all 64 grid cells.
Uses straight-line Rectified Flow Matching with Logit-Normal time sampling and Interval CFG.
"""
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from t3d.flow import GramSchmidtRoPE, SerializedAttention, FlowBlock, flow_pair

GRID_SIZE = 4
TOTAL_CELLS = 64  # 4 x 4 x 4


def get_grid_cells(device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Return cartesian grid coordinates for all 64 cells with shape (64, 3)."""
    return torch.cartesian_prod(*(torch.arange(GRID_SIZE, device=device) for _ in range(3)))


def pack_unified_token(
    posterior_coords: torch.Tensor,
    posterior_mean: torch.Tensor,
    mean_stats: Optional[torch.Tensor] = None,
    std_stats: Optional[torch.Tensor] = None,
    grid_size: int = GRID_SIZE,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack sparse posterior into dense (64, 1 + C) target tensor.

    Args:
        posterior_coords: (K, 3) or (K, 4 with batch) active cell coordinates.
        posterior_mean: (K, C) continuous latents.
        mean_stats: (C,) per-channel standardization mean.
        std_stats: (C,) per-channel standardization std.
        grid_size: Coarse grid dimension (default 4).
        device: Target device.

    Returns:
        target: (64, 1 + C) unified tensor [occupancy (-1 or +1), standardized latents].
        occ_mask: (64,) boolean mask of active cells.
        cells: (64, 3) coordinate tensor.
    """
    dev = device or posterior_mean.device
    cells = torch.cartesian_prod(*(torch.arange(grid_size, device=dev) for _ in range(3)))
    c_dim = posterior_mean.shape[1]
    target = torch.zeros((len(cells), 1 + c_dim), device=dev, dtype=posterior_mean.dtype)
    target[:, 0] = -1.0  # Void / empty default: -1.0 occupancy, 0.0 latents
    occ_mask = torch.zeros(len(cells), device=dev, dtype=torch.bool)

    if len(posterior_coords) > 0:
        coords = posterior_coords[:, 1:] if posterior_coords.shape[1] == 4 else posterior_coords
        flat = (coords[:, 0] * grid_size + coords[:, 1]) * grid_size + coords[:, 2]

        z = posterior_mean
        if mean_stats is not None and std_stats is not None:
            m = mean_stats.to(z.device, dtype=z.dtype)
            s = std_stats.to(z.device, dtype=z.dtype).clamp_min(1e-5)
            z = (z - m) / s

        target[flat, 0] = 1.0  # Active occupancy
        target[flat, 1:] = z
        occ_mask[flat] = True

    return target, occ_mask, cells


def unpack_unified_token(
    pred_clean: torch.Tensor,
    mean_stats: Optional[torch.Tensor] = None,
    std_stats: Optional[torch.Tensor] = None,
    occ_threshold: float = 0.0,
    grid_size: int = GRID_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack dense (64, 1 + C) predicted clean tensor into active coordinates and denormalized latents.

    Args:
        pred_clean: (64, 1 + C) tensor with [pred_occupancy, pred_latents].
        mean_stats: (C,) per-channel standardization mean.
        std_stats: (C,) per-channel standardization std.
        occ_threshold: Decision threshold on occupancy channel (default 0.0).
        grid_size: Coarse grid dimension (default 4).

    Returns:
        active_bcoords: (K, 4) batch-prefixed coordinates [batch_idx=0, x, y, z].
        active_latents: (K, C) denormalized continuous latents.
        active_mask: (64,) boolean mask.
    """
    dev = pred_clean.device
    cells = torch.cartesian_prod(*(torch.arange(grid_size, device=dev) for _ in range(3)))
    occ_logits = pred_clean[:, 0]
    active_mask = occ_logits >= occ_threshold

    if not active_mask.any():
        # Fallback to top-4 highest confidence cells if threshold rejected everything
        top_indices = torch.topk(occ_logits, k=min(4, len(occ_logits))).indices
        active_mask[top_indices] = True

    active_cells = cells[active_mask]
    bcoords = torch.cat([torch.zeros((len(active_cells), 1), dtype=torch.long, device=dev), active_cells], dim=1)

    latents = pred_clean[active_mask, 1:]
    if mean_stats is not None and std_stats is not None:
        m = mean_stats.to(latents.device, dtype=latents.dtype)
        s = std_stats.to(latents.device, dtype=latents.dtype)
        latents = latents * s + m

    return bcoords, latents, active_mask


def sample_logit_normal_time(shape: torch.Size, device: torch.device, sigma: float = 1.0) -> torch.Tensor:
    """Sample timestep t in (0, 1) from a Logit-Normal distribution.

    logit(t) ~ Normal(0, sigma^2) => t = sigmoid(u), u ~ Normal(0, sigma^2).
    Concentrates gradient updates in mid-trajectory t in [0.2, 0.8] where ODE paths bend most.
    """
    u = torch.randn(shape, device=device) * sigma
    return torch.sigmoid(u)


class UnifiedFlowDiT(nn.Module):
    """Dense Diffusion Transformer on 64 spatial cells for joint occupancy & shape latents."""

    def __init__(
        self,
        latent_channels: int = 16,
        condition_channels: int = 32,
        width: int = 192,
        heads: int = 6,
        depth: int = 6,
        patch_size: int = 64,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.total_channels = 1 + latent_channels  # 1 occupancy + C geometry
        self.condition_channels = condition_channels
        self.width = width
        self.heads = heads
        self.depth = depth
        self.patch_size = patch_size

        self.input_proj = nn.Linear(self.total_channels, width)
        self.prompt_proj = nn.Linear(condition_channels, width)
        self.null_prompt = nn.Parameter(torch.zeros(1, width))

        half = width // 2
        self.register_buffer(
            'time_frequencies',
            torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / max(1, half - 1))
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width)
        )

        self.blocks = nn.ModuleList([
            FlowBlock(width, heads, patch_size) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, self.total_channels)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        time: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        order: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (64, 1 + C) noisy tokens.
            coords: (64, 3) cell coordinates.
            time: scalar or (1,) timestep in [0, 1].
            condition: (seq_len, condition_channels) text tokens, or None for unconditioned.
            order: Optional permutation order of tokens (defaults to identity 0..63).

        Returns:
            velocity_pred: (64, 1 + C) predicted vector field.
        """
        n = len(x)
        if order is None:
            order = torch.arange(n, device=x.device)

        # Time embedding
        t_val = time.view(1).to(dtype=torch.float32)
        phase = 1000.0 * t_val * self.time_frequencies
        t_embed = torch.cat([phase.cos(), phase.sin()], dim=-1)
        time_context = self.time_mlp(t_embed.to(x.dtype)).squeeze(0)

        # Prompt conditioning
        prompt = self.null_prompt if condition is None else self.prompt_proj(condition)

        features = self.input_proj(x)
        for block in self.blocks:
            features = block(features, coords, time_context, prompt, order)

        return self.output_proj(self.norm(features))


@torch.no_grad()
def sample_unified_flow(
    model: UnifiedFlowDiT,
    condition: Optional[torch.Tensor],
    coords: Optional[torch.Tensor] = None,
    steps: int = 30,
    guidance: float = 3.5,
    method: str = 'midpoint',
    cfg_min: float = 0.25,
    cfg_max: float = 0.85,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample clean unified tokens from noise with Interval CFG.

    Args:
        model: Trained UnifiedFlowDiT.
        condition: Prompt tokens tensor (seq_len, cond_dim) or None.
        coords: (64, 3) grid coordinates (defaults to get_grid_cells).
        steps: Integration steps (default 30).
        guidance: CFG scale during active interval (default 3.5).
        method: ODE solver ('midpoint' or 'euler').
        cfg_min: Lower bound of CFG interval (default 0.25). Below this, guidance=1.0.
        cfg_max: Upper bound of CFG interval (default 0.85). Above this, guidance=1.0.
        seed: Random seed for initial noise.
        device: Device.

    Returns:
        x_clean: (64, 1 + C) denoised clean tokens at t=0.
    """
    dev = device or next(model.parameters()).device
    if coords is None:
        coords = get_grid_cells(dev)

    generator = torch.Generator(device=dev)
    if seed is not None:
        generator.manual_seed(seed)

    x = torch.randn((len(coords), model.total_channels), device=dev, generator=generator)
    dt = 1.0 / steps
    order = torch.arange(len(coords), device=dev)

    for i in range(steps):
        t_scalar = float(i * dt)
        t_tensor = x.new_tensor(t_scalar)

        # Interval CFG check
        effective_w = guidance if (cfg_min <= t_scalar <= cfg_max) else 1.0

        if method == 'euler':
            cond = model(x, coords, t_tensor, condition, order=order)
            if effective_w != 1.0 and condition is not None:
                uncond = model(x, coords, t_tensor, None, order=order)
                v = uncond + effective_w * (cond - uncond)
            else:
                v = cond
            x = x + dt * v

        elif method == 'midpoint':
            t_mid_tensor = x.new_tensor(t_scalar + 0.5 * dt)
            cond = model(x, coords, t_tensor, condition, order=order)
            if effective_w != 1.0 and condition is not None:
                uncond = model(x, coords, t_tensor, None, order=order)
                v1 = uncond + effective_w * (cond - uncond)
            else:
                v1 = cond

            x_mid = x + 0.5 * dt * v1
            cond_mid = model(x_mid, coords, t_mid_tensor, condition, order=order)
            if effective_w != 1.0 and condition is not None:
                uncond_mid = model(x_mid, coords, t_mid_tensor, None, order=order)
                v_mid = uncond_mid + effective_w * (cond_mid - uncond_mid)
            else:
                v_mid = cond_mid

            x = x + dt * v_mid

    return x
