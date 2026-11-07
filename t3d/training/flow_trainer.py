"""Optimal Transport Rectified Flow trainer with Logit-Normal timestep sampling

and Decoder-in-the-Loop fine-tuning for T3D models.
"""
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from ..sc_vae import SparseCompressionVAE


def sample_logit_normal(
    size: Union[int, Tuple[int, ...]] = (),
    m: float = 0.0,
    s: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-5,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample timesteps from a Logit-Normal distribution t = sigma(u), u ~ N(m, s^2).

    Concentrates gradient updates on mid-trajectories (t in [0.3, 0.7]) where ambiguous
    topological decisions occur. Clamped to [eps, 1 - eps] for numerical stability.

    Args:
        size: Output tensor shape.
        m: Mean of underlying Gaussian.
        s: Standard deviation of underlying Gaussian (must be positive).
        device: Device to place the tensor on.
        dtype: Data type of the tensor.
        eps: Small margin to clamp away from exact 0 and 1.
        generator: Optional PyTorch random generator.

    Returns:
        Tensor of sampled timesteps in (0, 1).
    """
    if s <= 0:
        raise ValueError(f"Standard deviation s must be positive, got {s}")
    if not (0 < eps < 0.5):
        raise ValueError(f"eps must lie strictly in (0, 0.5), got {eps}")

    u = torch.randn(size, device=device, dtype=dtype, generator=generator) * s + m
    t = torch.sigmoid(u)
    return t.clamp(eps, 1.0 - eps)


def rectified_flow_pair(
    clean: torch.Tensor,
    noise: torch.Tensor,
    time: Union[float, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Connect Gaussian noise x_0 to clean target x_1 along a linear trajectory.

    x_t = (1 - t) * x_0 + t * x_1
    v_target = x_1 - x_0

    Args:
        clean: Clean target tensor x_1.
        noise: Standard normal noise tensor x_0 of matching shape.
        time: Scalar or broadcastable time tensor in [0, 1].

    Returns:
        Tuple of (interpolated x_t, velocity target x_1 - x_0).
    """
    if clean.shape != noise.shape:
        raise ValueError(f"Clean shape {clean.shape} does not match noise shape {noise.shape}")
    t = torch.as_tensor(time, device=clean.device, dtype=clean.dtype)
    if (t < 0.0).any() or (t > 1.0).any():
        raise ValueError(f"Timestep t must lie in [0, 1], got min {t.min()} and max {t.max()}")

    noisy = (1.0 - t) * noise + t * clean
    target_velocity = clean - noise
    return noisy, target_velocity


def chamfer_distance(
    p1: torch.Tensor,
    p2: torch.Tensor,
    squared: bool = True,
) -> torch.Tensor:
    """Compute the bidirectional Chamfer Distance between point clouds p1 and p2.

    D(p1, p2) = (1 / |p1|) sum_{x in p1} min_{y in p2} ||x - y||^2
              + (1 / |p2|) sum_{y in p2} min_{x in p1} ||y - x||^2

    Supports unbatched (N1, 3), (N2, 3) or batched (B, N1, 3), (B, N2, 3) point clouds.

    Args:
        p1: Point cloud tensor of shape (..., N1, 3).
        p2: Point cloud tensor of shape (..., N2, 3).
        squared: If True, uses squared Euclidean distance ||x - y||^2; else ||x - y||_2.

    Returns:
        Scalar differentiable tensor representing the bidirectional Chamfer distance.
    """
    if p1.ndim < 2 or p2.ndim < 2 or p1.shape[-1] != 3 or p2.shape[-1] != 3:
        raise ValueError(f"Expected point clouds with trailing dimension 3, got {p1.shape} and {p2.shape}")

    if p1.shape[-2] == 0 or p2.shape[-2] == 0:
        return torch.zeros((), device=p1.device, dtype=p1.dtype, requires_grad=p1.requires_grad or p2.requires_grad)

    # Compute pairwise Euclidean distances
    # p1: (..., N1, 3), p2: (..., N2, 3)
    dists = torch.cdist(p1, p2, p=2.0)
    if squared:
        dists = dists.square()

    # Nearest point in p2 for each point in p1: min along last dimension (N2)
    min_dist_p1_to_p2 = dists.min(dim=-1).values
    # Nearest point in p1 for each point in p2: min along second-to-last dimension (N1)
    min_dist_p2_to_p1 = dists.min(dim=-2).values

    chamfer = min_dist_p1_to_p2.mean() + min_dist_p2_to_p1.mean()
    return chamfer


def rectified_flow_loss(
    model: nn.Module,
    x_1: torch.Tensor,
    coords: torch.Tensor,
    condition: Optional[torch.Tensor] = None,
    *,
    noise: Optional[torch.Tensor] = None,
    time: Optional[Union[float, torch.Tensor]] = None,
    surface_ratio: Optional[torch.Tensor] = None,
    base_weight: float = 1.0,
    surface_scale: float = 2.0,
    time_weight_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    orders: Optional[List[torch.Tensor]] = None,
    shape: Optional[torch.Tensor] = None,
    m: float = 0.0,
    s: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the Optimal Transport Rectified Flow loss for a single asset or batch.

    L_flow = E_{t, x_0, x_1} [ w(t) * W_cell * ||v_theta(x_t, t, c) - (x_1 - x_0)||^2 ]

    Args:
        model: Flow velocity network (e.g. SparseFlowDiT).
        x_1: Clean data target latents, shape (N, C).
        coords: Token / cell spatial coordinates, shape (N, 3).
        condition: Prompt or corrupted condition tokens, shape (P, C_cond) or None.
        noise: Optional explicit noise tensor x_0 ~ N(0, I).
        time: Optional explicit scalar timestep in [0, 1]. Sampled via Logit-Normal if None.
        surface_ratio: Optional fraction of surface voxels per coarse cell, shape (N,).
        base_weight: Baseline loss weight for all cells.
        surface_scale: Multiplicative scale added to surface-dense cells.
        time_weight_fn: Optional time weighting function w(t).
        orders: Serialization permutations for patch attention.
        shape: Optional shape conditioning branch.
        m: Logit-Normal mean.
        s: Logit-Normal standard deviation.

    Returns:
        Tuple of (scalar flow loss tensor, metrics dict).
    """
    if x_1.ndim != 2:
        raise ValueError(f"Expected clean target x_1 with shape (N, C), got {x_1.shape}")

    device = x_1.device
    dtype = x_1.dtype

    if noise is None:
        noise = torch.randn_like(x_1)
    elif noise.shape != x_1.shape:
        raise ValueError(f"Noise shape {noise.shape} does not match clean shape {x_1.shape}")

    if time is None:
        time = sample_logit_normal((), m=m, s=s, device=device, dtype=dtype)
    else:
        time = torch.as_tensor(time, device=device, dtype=dtype)

    noisy, target_v = rectified_flow_pair(x_1, noise, time)
    pred_v = model(noisy, coords, time, condition, orders=orders, shape=shape)

    # Elementwise squared error per token: sum or mean over latent channels C
    err = (pred_v - target_v).square().mean(dim=-1)  # (N,)

    # Surface ratio cell weighting
    if surface_ratio is not None:
        sr = torch.as_tensor(surface_ratio, device=device, dtype=dtype)
        if sr.shape != (len(x_1),):
            raise ValueError(f"surface_ratio shape {sr.shape} does not match token length {len(x_1)}")
        cell_weights = base_weight + surface_scale * sr.clamp(0.0, 1.0)
        norm_weights = cell_weights / cell_weights.mean().clamp_min(1e-6)
        loss = (err * norm_weights).mean()
    else:
        loss = err.mean()

    # Optional time weighting w(t)
    if time_weight_fn is not None:
        w_t = time_weight_fn(time)
        loss = loss * w_t

    metrics = {
        "flow_loss": loss.item(),
        "time": time.item(),
    }
    return loss, metrics


def decoder_in_the_loop_loss(
    decoder: SparseCompressionVAE,
    x_t: torch.Tensor,
    pred_v: torch.Tensor,
    time: Union[float, torch.Tensor],
    coords: torch.Tensor,
    *,
    teacher_occupancy: Optional[List[torch.Tensor]] = None,
    target_coords: Optional[torch.Tensor] = None,
    target_features: Optional[torch.Tensor] = None,
    target_occupancy: Optional[List[torch.Tensor]] = None,
    target_points: Optional[torch.Tensor] = None,
    chamfer_weight: float = 1.0,
    occupancy_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Decode predicted continuous latents z_hat into voxel SDF during training,

    computing surface occupancy and Chamfer loss gradients into the flow network.

    z_hat = x_t + (1 - t) * v_pred

    Args:
        decoder: Frozen SparseCompressionVAE instance.
        x_t: Noisy state tensor at time t, shape (N, C).
        pred_v: Predicted velocity from flow network, shape (N, C).
        time: Timestep in [0, 1].
        coords: Coarse coordinates, shape (N, 3) or (N, 4).
        teacher_occupancy: Optional teacher occupancy masks for upsampling stages.
        target_coords: Ground truth fine voxel coordinates, shape (M, 3) or (M, 4).
        target_features: Ground truth fine voxel shape features (M, 7).
        target_occupancy: Ground truth multi-stage occupancy masks (list of 4 tensors).
        target_points: Optional explicit target surface points for Chamfer loss.
        chamfer_weight: Weight for Chamfer distance loss term.
        occupancy_weight: Weight for multi-stage occupancy BCE loss term.

    Returns:
        Tuple of (scalar decoder loss tensor, metrics dict).
    """
    t = torch.as_tensor(time, device=x_t.device, dtype=x_t.dtype)
    # Recover predicted clean latent via Tweedie / linear flow relation:
    # x_t = (1 - t)*x_0 + t*x_1, v = x_1 - x_0 => x_t + (1 - t)*v = x_1
    z_hat = x_t + (1.0 - t) * pred_v

    # Prepare 4-channel coordinates for SC-VAE: (batch_idx, x, y, z)
    if coords.ndim != 2:
        raise ValueError(f"coords must be a 2D tensor, got {coords.shape}")
    if coords.shape[1] == 3:
        vae_coords = torch.cat(
            [torch.zeros((len(coords), 1), device=coords.device, dtype=torch.int64), coords.long()],
            dim=1,
        )
    elif coords.shape[1] == 4:
        vae_coords = coords.long()
    else:
        raise ValueError(f"Expected coords with 3 or 4 channels, got {coords.shape[1]}")

    # Forward through frozen decoder
    occupancy_mask_teacher = teacher_occupancy if teacher_occupancy is not None else target_occupancy
    out_coords, raw, logits = decoder.decode(vae_coords, z_hat, teacher_occupancy=occupancy_mask_teacher)

    total_decoder_loss = torch.zeros((), device=x_t.device, dtype=x_t.dtype)
    metrics: Dict[str, float] = {}

    # 1. Surface Occupancy Loss
    target_occ = target_occupancy if target_occupancy is not None else teacher_occupancy
    if target_occ is not None and occupancy_weight > 0.0:
        if len(logits) != len(target_occ):
            raise ValueError(f"Decoder stages ({len(logits)}) do not match target occupancy ({len(target_occ)})")
        occ_stage_losses = [
            F.binary_cross_entropy_with_logits(pred_logit, mask.to(pred_logit.dtype))
            for pred_logit, mask in zip(logits, target_occ)
        ]
        occ_loss = torch.stack(occ_stage_losses).mean()
        # If fine-level edge occupancy flags are available, include them
        if target_features is not None and target_features.shape[-1] >= 6:
            edge_occ_loss = F.binary_cross_entropy_with_logits(raw[:, 3:6], target_features[:, 3:6])
            occ_loss = occ_loss + edge_occ_loss
        total_decoder_loss = total_decoder_loss + occupancy_weight * occ_loss
        metrics["decoder_occupancy_loss"] = occ_loss.item()

    # 2. Chamfer Loss
    if chamfer_weight > 0.0:
        # Predicted dual vertices in fine voxel grid space
        c_pred = out_coords[:, 1:4].float() if out_coords.shape[1] == 4 else out_coords[:, :3].float()
        pred_points = c_pred + raw[:, :3].sigmoid()

        if target_points is not None:
            gt_points = target_points
        elif target_coords is not None and target_features is not None:
            c_gt = target_coords[:, 1:4].float() if target_coords.shape[1] == 4 else target_coords[:, :3].float()
            gt_points = c_gt + target_features[:, :3]
        else:
            gt_points = None

        if gt_points is not None:
            c_loss = chamfer_distance(pred_points, gt_points, squared=True)
            total_decoder_loss = total_decoder_loss + chamfer_weight * c_loss
            metrics["decoder_chamfer_loss"] = c_loss.item()

    metrics["decoder_total_loss"] = total_decoder_loss.item()
    return total_decoder_loss, metrics


@torch.no_grad()
def sample_ode(
    model: nn.Module,
    x_0: torch.Tensor,
    coords: torch.Tensor,
    condition: Optional[torch.Tensor] = None,
    *,
    steps: int = 50,
    solver: str = "euler",
    guidance: float = 1.0,
    guidance_interval: Tuple[float, float] = (0.0, 1.0),
    orders: Optional[List[torch.Tensor]] = None,
    shape: Optional[torch.Tensor] = None,
    return_trajectory: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """Integrate the Rectified Flow ODE trajectory from t=0 (noise) to t=1 (data).

    Supports Classifier-Free Guidance (CFG):
      v_guided = v_uncond + w * (v_cond - v_uncond)
    applied within guidance_interval = [t_min, t_max].

    Solvers:
      - 'euler': 1st order Euler forward integration
      - 'heun': 2nd order Heun predictor-corrector integration

    Args:
        model: Flow velocity network v_theta(x, coords, t, condition).
        x_0: Initial Gaussian noise tensor at t=0, shape (N, C).
        coords: Token / cell coordinates, shape (N, 3).
        condition: Prompt / conditioning features, or None.
        steps: Number of integration steps K >= 1.
        solver: ODE solver algorithm ('euler' or 'heun').
        guidance: CFG scale w >= 1.0.
        guidance_interval: Interval [t_min, t_max] in [0, 1] where CFG is active.
        orders: Serialization permutations for patch attention.
        shape: Optional shape conditioning branch.
        return_trajectory: If True, returns (final_state, list_of_intermediate_states).

    Returns:
        Final state x_1 or tuple of (x_1, trajectory).
    """
    if steps <= 0:
        raise ValueError(f"Steps must be a positive integer, got {steps}")
    if solver not in ("euler", "heun"):
        raise ValueError(f"Solver must be 'euler' or 'heun', got '{solver}'")
    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError(f"Guidance must be a finite nonnegative float, got {guidance}")
    if len(guidance_interval) != 2 or not (0.0 <= guidance_interval[0] <= guidance_interval[1] <= 1.0):
        raise ValueError(f"guidance_interval must be [t_min, t_max] within [0, 1], got {guidance_interval}")

    def eval_velocity(x_curr: torch.Tensor, t_curr: Union[float, torch.Tensor]) -> torch.Tensor:
        t_scalar = float(t_curr.item()) if isinstance(t_curr, torch.Tensor) else float(t_curr)
        t_ten = x_curr.new_tensor(t_scalar)
        use_cfg = (
            guidance != 1.0
            and condition is not None
            and (guidance_interval[0] <= t_scalar <= guidance_interval[1])
        )
        if not use_cfg:
            return model(x_curr, coords, t_ten, condition, orders=orders, shape=shape)

        v_cond = model(x_curr, coords, t_ten, condition, orders=orders, shape=shape)
        v_uncond = model(x_curr, coords, t_ten, None, orders=orders, shape=shape)
        return v_uncond + guidance * (v_cond - v_uncond)

    x = x_0.clone()
    trajectory = [x.clone()] if return_trajectory else []
    h = 1.0 / steps

    for k in range(steps):
        t_k = k / steps
        if solver == "euler":
            v = eval_velocity(x, t_k)
            x = x + h * v
        elif solver == "heun":
            v1 = eval_velocity(x, t_k)
            x_pred = x + h * v1
            t_next = min(1.0, (k + 1) / steps)
            v2 = eval_velocity(x_pred, t_next)
            x = x + 0.5 * h * (v1 + v2)

        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return x, trajectory
    return x


@dataclass
class FlowTrainerConfig:
    """Configuration dataclass for FlowTrainer."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    logit_normal_m: float = 0.0
    logit_normal_s: float = 1.0
    base_weight: float = 1.0
    surface_scale: float = 2.0
    cfg_dropout_prob: float = 0.1
    flow_weight: float = 1.0
    decoder_weight: float = 0.0
    chamfer_weight: float = 1.0
    occupancy_weight: float = 1.0
    grad_clip_norm: Optional[float] = 1.0
    device: str = "cpu"


class FlowTrainer:
    """High-level trainer for Optimal Transport Rectified Flow with Decoder-in-the-Loop.

    Supports:
      - Logit-Normal timestep sampling focusing updates on ambiguous topology decisions.
      - Surface ratio loss weighting prioritizing dense surface cells.
      - Frozen SC-VAE decoder-in-the-loop fine-tuning providing occupancy and Chamfer gradients.
      - Resumable state checkpoints with version fingerprints.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        decoder: Optional[SparseCompressionVAE] = None,
        config: Optional[FlowTrainerConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        manifest_sha256: str = "",
        **kwargs,
    ):
        self.config = config or FlowTrainerConfig(**kwargs) if kwargs else (config or FlowTrainerConfig())
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)

        if decoder is not None:
            self.decoder: Optional[SparseCompressionVAE] = decoder.to(self.device)
            self.decoder.eval()
            for p in self.decoder.parameters():
                p.requires_grad = False
        else:
            self.decoder = None

        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = optimizer

        self.manifest_sha256 = manifest_sha256
        self.steps = 0
        self.epoch = 0

    def compute_loss(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the combined Rectified Flow + Decoder-in-the-Loop loss.

        Expected batch keys:
          - 'x_0_lat' or 'clean': (N, C) or (B, N, C) continuous latents.
          - 'coords': (N, 3) or (N, 4) coarse voxel coordinates.
          - 'c_raw' or 'condition': Optional (P, C_cond) or (B, P, C_cond) condition latents.
          - 'surface_ratio': Optional (N,) or (B, N) surface density weighting.
          - 'target_occupancy' or 'teacher_occupancy': Optional multi-stage occupancy masks.
          - 'target_features': Optional (M, 7) fine voxel targets.
          - 'target_coords': Optional (M, 3) or (M, 4) fine voxel coordinates.
          - 'target_points': Optional (P_gt, 3) ground truth surface points.
        """
        x_1 = batch.get("x_0_lat", batch.get("clean"))
        if x_1 is None:
            raise KeyError("Batch must contain 'x_0_lat' or 'clean' target continuous latents")
        x_1 = x_1.to(self.device)

        coords = batch.get("coords")
        if coords is None:
            # If coordinates are not provided, synthesize standard grid coordinates
            n_tokens = x_1.shape[-2]
            res = round(n_tokens ** (1.0 / 3.0))
            if res ** 3 == n_tokens:
                grid = torch.stack(
                    torch.meshgrid(
                        torch.arange(res, device=self.device),
                        torch.arange(res, device=self.device),
                        torch.arange(res, device=self.device),
                        indexing="ij",
                    ),
                    dim=-1,
                ).reshape(-1, 3)
                coords = grid
            else:
                raise ValueError(f"coords missing and token count {n_tokens} is not a perfect cube")
        else:
            coords = coords.to(self.device)

        condition = batch.get("c_raw", batch.get("condition"))
        if condition is not None:
            condition = condition.to(self.device)
            # CFG conditioning dropout during training
            if self.config.cfg_dropout_prob > 0.0 and self.model.training:
                if torch.rand((), device=self.device).item() < self.config.cfg_dropout_prob:
                    condition = None

        surface_ratio = batch.get("surface_ratio")
        if surface_ratio is not None:
            surface_ratio = surface_ratio.to(self.device)

        # Handle batched (B, N, C) vs single asset (N, C)
        if x_1.ndim == 3:
            b_size = x_1.shape[0]
            losses = []
            metrics_acc: Dict[str, float] = {}
            for b in range(b_size):
                b_x1 = x_1[b]
                b_coords = coords[b] if coords.ndim == 3 else coords
                b_cond = condition[b] if condition is not None and condition.ndim == 3 else condition
                b_sr = surface_ratio[b] if surface_ratio is not None and surface_ratio.ndim == 2 else surface_ratio

                b_sub = {
                    "x_0_lat": b_x1,
                    "coords": b_coords,
                    "condition": b_cond,
                    "surface_ratio": b_sr,
                }
                for k in ("target_occupancy", "teacher_occupancy", "target_features", "target_coords", "target_points"):
                    if k in batch:
                        b_sub[k] = batch[k]

                sub_loss, sub_metrics = self._compute_single_loss(b_sub)
                losses.append(sub_loss)
                for k, v in sub_metrics.items():
                    metrics_acc[k] = metrics_acc.get(k, 0.0) + v / b_size
            total_loss = torch.stack(losses).mean()
            return total_loss, metrics_acc

        return self._compute_single_loss(
            {
                "x_0_lat": x_1,
                "coords": coords,
                "condition": condition,
                "surface_ratio": surface_ratio,
                **{k: batch[k] for k in ("target_occupancy", "teacher_occupancy", "target_features", "target_coords", "target_points") if k in batch},
            }
        )

    def _compute_single_loss(self, sub_batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        x_1 = sub_batch["x_0_lat"]
        coords = sub_batch["coords"]
        condition = sub_batch.get("condition")
        surface_ratio = sub_batch.get("surface_ratio")

        # 1. Timestep sampling via Logit-Normal
        time = sample_logit_normal(
            (),
            m=self.config.logit_normal_m,
            s=self.config.logit_normal_s,
            device=self.device,
            dtype=x_1.dtype,
        )

        noise = torch.randn_like(x_1)
        noisy, target_v = rectified_flow_pair(x_1, noise, time)
        pred_v = self.model(noisy, coords, time, condition)

        # Flow loss with surface ratio weighting
        err = (pred_v - target_v).square().mean(dim=-1)
        if surface_ratio is not None:
            sr = surface_ratio.clamp(0.0, 1.0)
            cell_weights = self.config.base_weight + self.config.surface_scale * sr
            norm_weights = cell_weights / cell_weights.mean().clamp_min(1e-6)
            f_loss = (err * norm_weights).mean()
        else:
            f_loss = err.mean()

        total_loss = self.config.flow_weight * f_loss
        metrics = {"flow_loss": f_loss.item(), "time": time.item()}

        # 2. Decoder-in-the-Loop Fine-Tuning
        if self.decoder is not None and self.config.decoder_weight > 0.0:
            dec_loss, dec_metrics = decoder_in_the_loop_loss(
                decoder=self.decoder,
                x_t=noisy,
                pred_v=pred_v,
                time=time,
                coords=coords,
                teacher_occupancy=sub_batch.get("teacher_occupancy"),
                target_coords=sub_batch.get("target_coords"),
                target_features=sub_batch.get("target_features"),
                target_occupancy=sub_batch.get("target_occupancy"),
                target_points=sub_batch.get("target_points"),
                chamfer_weight=self.config.chamfer_weight,
                occupancy_weight=self.config.occupancy_weight,
            )
            total_loss = total_loss + self.config.decoder_weight * dec_loss
            metrics.update(dec_metrics)

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a single optimization step."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss, metrics = self.compute_loss(batch)
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite training loss encountered: {loss.item()}")

        loss.backward()

        if self.config.grad_clip_norm is not None and self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

        # Verify gradient validity
        for name, param in self.model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise ValueError(f"Nonfinite gradients in parameter {name}")

        self.optimizer.step()
        self.steps += 1
        return metrics

    @torch.no_grad()
    def sample(
        self,
        coords: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        *,
        channels: Optional[int] = None,
        steps: int = 50,
        solver: str = "euler",
        guidance: float = 1.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        orders: Optional[List[torch.Tensor]] = None,
        shape: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Generate clean latents from Gaussian noise using sample_ode."""
        self.model.eval()
        c = channels or getattr(self.model, "channels", None)
        if c is None:
            raise ValueError("channels must be specified or present on model")

        coords = coords.to(self.device)
        noise = torch.randn((len(coords), c), device=self.device, dtype=torch.float32)

        return sample_ode(
            model=self.model,
            x_0=noise,
            coords=coords,
            condition=condition.to(self.device) if condition is not None else None,
            steps=steps,
            solver=solver,
            guidance=guidance,
            guidance_interval=guidance_interval,
            orders=orders,
            shape=shape.to(self.device) if shape is not None else None,
            return_trajectory=return_trajectory,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save a training checkpoint."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 1,
            "manifest_sha256": self.manifest_sha256,
            "steps": self.steps,
            "epoch": self.epoch,
            "config": self.config.__dict__,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".checkpoint-flow-")
        os.close(fd)
        try:
            with open(tmp, "wb") as f:
                torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)
        finally:
            Path(tmp).unlink(missing_ok=True)

    @classmethod
    def resume(
        cls,
        path: Union[str, Path],
        model: nn.Module,
        *,
        decoder: Optional[SparseCompressionVAE] = None,
        manifest_sha256: str = "",
        device: Optional[str] = None,
    ) -> "FlowTrainer":
        """Resume trainer from checkpoint."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("version") != 1:
            raise ValueError(f"Unsupported checkpoint version: {state.get('version')}")
        if manifest_sha256 and state.get("manifest_sha256") != manifest_sha256:
            raise ValueError("Checkpoint dataset manifest fingerprint mismatch")

        cfg_dict = state["config"]
        if device is not None:
            cfg_dict["device"] = device
        cfg = FlowTrainerConfig(**cfg_dict)

        trainer = cls(model=model, decoder=decoder, config=cfg, manifest_sha256=state.get("manifest_sha256", ""))
        trainer.model.load_state_dict(state["model_state"])
        trainer.optimizer.load_state_dict(state["optimizer_state"])
        trainer.steps = state["steps"]
        trainer.epoch = state["epoch"]
        return trainer
