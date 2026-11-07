"""Four-stage sparse shape/material variational autoencoder reference.

Implements trainable architecture and tensor losses; rendering supervision,
production kernels and real-asset training are separate pending work.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .sparse_compression import _validate, SparseDownsample, SparseUpsample
from .sparse_blocks import SparseConvNeXtBlock


class HermiteNormalHead(nn.Module):
    """Multi-task prediction head for continuous 3D surface normals at dual edge intersections.

    Predicts 3D continuous surface normals (nabla SDF / unit vectors in R^3) for Hermite
    Dual Contouring sharp feature reconstruction. Enforces unit vector normalization
    n_hat = n / (||n||_2 + eps) on active voxels, and [0, 0, 0] on void voxels.
    """
    def __init__(self, in_channels: int, hidden_dim: int | None = None, eps: float = 1e-6):
        super().__init__()
        if in_channels <= 0:
            raise ValueError('in_channels must be positive')
        self.in_channels = in_channels
        self.eps = eps
        hidden = hidden_dim if hidden_dim is not None else max(in_channels, 32)
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, features: torch.Tensor, active_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Predict normalized 3D continuous surface normals.

        Args:
            features: Tensor of shape (B, N, C) or (N, C) or (..., C).
            active_mask: Optional boolean or float mask indicating active voxels.
                         If None, voxels whose features are all-zero are treated as void,
                         while all non-zero voxels are treated as active.

        Returns:
            Normal vectors of shape (B, N, 3) or (N, 3) or (..., 3):
            Unit vectors ||n_hat||_2 == 1.0 on active voxels, [0, 0, 0] on void voxels.
        """
        if features.shape[-1] != self.in_channels:
            raise ValueError(f'Expected feature dimension {self.in_channels}, got {features.shape[-1]}')
        raw = self.net(features)
        norm = torch.linalg.norm(raw, ord=2, dim=-1, keepdim=True)
        unit = raw / norm.clamp_min(self.eps)
        if active_mask is not None:
            mask = active_mask
            if mask.dim() == features.dim() - 1:
                mask = mask.unsqueeze(-1)
            return torch.where(mask.bool(), unit, torch.zeros_like(unit))
        else:
            is_void = (features == 0).all(dim=-1, keepdim=True)
            return torch.where(is_void, torch.zeros_like(unit), unit)


HermiteHead = HermiteNormalHead


def hermite_normal_loss(
    pred_normals: torch.Tensor,
    gt_normals: torch.Tensor,
    active_mask: torch.Tensor | None = None,
    reduction: str = 'mean',
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute Hermite normal cosine loss: L = 1 - <n_hat, n_gt>.

    Args:
        pred_normals: Predicted unit normals of shape (..., 3).
        gt_normals: Ground-truth unit normals of shape (..., 3).
        active_mask: Optional boolean/float mask indicating active voxels.
        reduction: 'mean', 'sum', or 'none'.
        eps: Small constant for numerical stability.

    Returns:
        Loss value.
    """
    if pred_normals.shape != gt_normals.shape:
        raise ValueError(f'pred_normals shape {pred_normals.shape} does not match gt_normals shape {gt_normals.shape}')
    if pred_normals.shape[-1] != 3:
        raise ValueError(f'Expected 3-channel normal vectors, got {pred_normals.shape[-1]}')
    if reduction not in ('mean', 'sum', 'none'):
        raise ValueError(f'Unsupported reduction: {reduction}')

    pred_norm = torch.linalg.norm(pred_normals, ord=2, dim=-1, keepdim=True)
    gt_norm = torch.linalg.norm(gt_normals, ord=2, dim=-1, keepdim=True)
    pred_unit = torch.where(pred_norm > eps, pred_normals / pred_norm.clamp_min(eps), pred_normals)
    gt_unit = torch.where(gt_norm > eps, gt_normals / gt_norm.clamp_min(eps), gt_normals)

    cos_sim = torch.clamp((pred_unit * gt_unit).sum(dim=-1), -1.0, 1.0)
    loss = 1.0 - cos_sim

    if active_mask is not None:
        mask = active_mask
        if mask.shape[-1] == 1 and mask.dim() == loss.dim() + 1:
            mask = mask.squeeze(-1)
        mask = mask.bool()
        if reduction == 'none':
            return torch.where(mask, loss, torch.zeros_like(loss))
        elif mask.sum() == 0:
            return torch.tensor(0.0, device=pred_normals.device, dtype=pred_normals.dtype, requires_grad=True)
        loss = loss[mask]

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


@dataclass
class SparsePosterior:
    coords: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor
    occupancy: list

    def sample(self, stochastic=True):
        return self.mean + torch.exp(.5*self.logvar)*torch.randn_like(self.mean) if stochastic else self.mean

    def kl(self):
        return .5*(self.mean.square()+self.logvar.exp()-1-self.logvar).mean()


class SparseCompressionVAE(nn.Module):
    def __init__(self, mode='shape', channels=(32, 64, 128, 256, 512), latent_channels=16, *, convolution_backend='reference'):
        super().__init__()
        if mode not in ('shape', 'material') or len(channels) != 5 or min(channels) <= 0 or latent_channels <= 0:
            raise ValueError('Specify shape/material mode, five positive widths, and positive latent channels')
        self.mode, self.channels = mode, tuple(channels)
        inputs = 7 if mode == 'shape' else 6
        self.input = nn.Linear(inputs, channels[0])
        self.encoder_blocks = nn.ModuleList([SparseConvNeXtBlock(c, backend=convolution_backend) for c in channels])
        self.down = nn.ModuleList([SparseDownsample(channels[i], channels[i+1]) for i in range(4)])
        self.moments = nn.Linear(channels[-1], 2*latent_channels)
        self.from_latent = nn.Linear(latent_channels, channels[-1])
        self.decoder_blocks = nn.ModuleList([SparseConvNeXtBlock(c, backend=convolution_backend) for c in reversed(channels)])
        self.up = nn.ModuleList([SparseUpsample(channels[i+1], channels[i]) for i in reversed(range(4))])
        self.output = nn.Linear(channels[0], inputs)
        if mode == 'shape':
            self.hermite_head = HermiteNormalHead(channels[0])
        else:
            self.hermite_head = None

    def encode(self, coords, features):
        _validate(coords, features)
        if len(coords) == 0:
            raise ValueError('Encoding requires at least one active voxel')
        features = self.input(features)
        occupancy = []
        for i, down in enumerate(self.down):
            features = self.encoder_blocks[i](coords, features)
            coords, features, mask = down(coords, features)
            occupancy.append(mask)
        features = self.encoder_blocks[4](coords, features)
        mean, logvar = self.moments(features).chunk(2, dim=1)
        return SparsePosterior(coords, mean, logvar.clamp(-20, 10), occupancy)

    def decode(self, coords, latent, *, teacher_occupancy=None, threshold=.5, max_active=1000000, trace=None, return_hermite=False):
        _validate(coords, latent)
        if teacher_occupancy is not None and len(teacher_occupancy) != 4:
            raise ValueError('Teacher occupancy must have four encoder levels')
        if trace is not None and not isinstance(trace, list):
            raise ValueError('trace must be a list')
        features = self.from_latent(latent)
        logits = []
        for i, up in enumerate(self.up):
            features = self.decoder_blocks[i](coords, features)
            teacher = None if teacher_occupancy is None else teacher_occupancy[3-i]
            # Bound potential child allocation before the upsampler executes.
            if len(coords)*8 > max_active:
                raise ValueError('Decoder child allocation exceeds max_active')
            parents = coords
            coords, features, prediction = up(coords, features, teacher, threshold=threshold)
            if trace is not None:
                selected = prediction.sigmoid() > threshold if teacher is None else teacher
                trace.append(dict(level=i, parents=parents.detach().clone(), logits=prediction.detach().clone(),
                                  selected=selected.detach().clone(), children=coords.detach().clone(),
                                  teacher_forced=teacher is not None))
            logits.append(prediction)
        features = self.decoder_blocks[4](coords, features)
        raw = self.output(features)
        if return_hermite:
            hermite = self.hermite_head(features) if self.hermite_head is not None else None
            return coords, raw, logits, hermite
        return coords, raw, logits

    def forward(self, coords, features, *, stochastic=True, return_hermite=False):
        posterior = self.encode(coords, features)
        output = self.decode(posterior.coords, posterior.sample(stochastic),
                             teacher_occupancy=posterior.occupancy,
                             return_hermite=return_hermite)
        return posterior, output


def reconstruction_loss(model, coords, features, *, stochastic=True, kl_weight=1e-6, occupancy_weight=1.,
                        edge_positive_weight=None, render_fn=None, render_weight=0., occupancy_level_weights=None,
                        hermite_target=None, hermite_weight=0.):
    """Shape: bounded dual xyz, edge BCE, split MSE; material: bounded PBR L1.

    Teacher masks reconstruct source support; they do not pass geometry features
    around the bottleneck. Decoder coordinates must exactly match sorted source
    coordinates. Values and masks are supervised independently of the KL term.
    """
    if kl_weight < 0 or occupancy_weight < 0 or render_weight < 0 or hermite_weight < 0:
        raise ValueError('Loss weights must be nonnegative')
    use_hermite = hermite_weight > 0 and hermite_target is not None
    has_hermite = model.mode == 'shape' and getattr(model, 'hermite_head', None) is not None
    posterior, decoded = model(coords, features, stochastic=stochastic, return_hermite=has_hermite)
    if has_hermite:
        out_coords, raw, logits, hermite_pred = decoded
    else:
        out_coords, raw, logits = decoded
    sorted_coords, inverse = torch.unique(coords, dim=0, sorted=True, return_inverse=True)
    if not torch.equal(out_coords, sorted_coords):
        raise ValueError('Decoded support differs from source support')
    target = features[torch.argsort(inverse)]
    if model.mode == 'shape':
        if ((target[:, :6] < 0) | (target[:, :6] > 1)).any():
            raise ValueError('Dual coordinates and edge targets must be in [0,1]')
        weight = None if edge_positive_weight is None else torch.as_tensor(edge_positive_weight, device=raw.device, dtype=raw.dtype)
        if weight is not None and (weight.shape not in (torch.Size([]), torch.Size([3])) or not torch.isfinite(weight).all() or (weight <= 0).any()):
            raise ValueError('edge_positive_weight must be finite positive scalar or three-vector')
        attribute = (F.mse_loss(raw[:, :3].sigmoid(), target[:, :3])
                     + F.binary_cross_entropy_with_logits(raw[:, 3:6], target[:, 3:6], pos_weight=weight)
                     + F.mse_loss(raw[:, 6:], target[:, 6:]))
    else:
        if ((target < 0) | (target > 1)).any():
            raise ValueError('PBR targets must be in [0,1]')
        attribute = F.l1_loss(raw.sigmoid(), target)
    stage_losses = torch.stack([F.binary_cross_entropy_with_logits(prediction, mask.to(prediction.dtype))
                                for prediction, mask in zip(logits, reversed(posterior.occupancy))])
    if occupancy_level_weights is None:
        occupancy = stage_losses.mean()
    else:
        weights = torch.as_tensor(occupancy_level_weights, dtype=stage_losses.dtype, device=stage_losses.device)
        if weights.shape != stage_losses.shape or not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError('Occupancy level weights must be four finite nonnegative values with positive sum')
        occupancy = (stage_losses * weights).sum() / weights.sum()
    kl = posterior.kl()
    total = attribute + occupancy_weight*occupancy + kl_weight*kl
    parts = dict(attribute=attribute, occupancy=occupancy, kl=kl)
    if has_hermite:
        if use_hermite:
            target_hermite = hermite_target[torch.argsort(inverse)]
            hermite_loss = hermite_normal_loss(hermite_pred, target_hermite)
            total = total + hermite_weight * hermite_loss
            parts['hermite'] = hermite_loss
        else:
            total = total + 0.0 * hermite_pred.sum()
    elif use_hermite:
        raise ValueError('Hermite normal supervision requires shape mode with hermite_head')
    if render_weight:
        if model.mode != 'shape' or render_fn is None:
            raise ValueError('shape render callback required')
        render = render_fn(out_coords, target, raw)
        total = total + render_weight * render
        parts['render'] = render
    return total, parts
