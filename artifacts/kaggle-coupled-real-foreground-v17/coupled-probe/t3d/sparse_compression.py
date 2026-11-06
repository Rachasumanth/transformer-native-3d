"""Differentiable sparse SC-VAE resampling foundations (PyTorch CPU/CUDA).

Coordinates are (batch,x,y,z) int64. Octants use x*4+y*2+z. Missing
children contribute zeros; grouping averages contiguous stacked channels.
Learned residual branches here are linear references; ConvNeXt sparse blocks,
variational heads and reconstruction losses are separate pending components.
"""
import torch
from torch import nn


def _validate(coords, features):
    if (coords.ndim != 2 or coords.shape[1] != 4 or coords.dtype != torch.int64
            or features.ndim != 2 or len(coords) != len(features)
            or coords.device != features.device or not features.is_floating_point()
            or features.shape[1] == 0):
        raise ValueError('Expected same-device int64 (N,4) coordinates and floating (N,C) features')
    if (coords < 0).any() or not torch.isfinite(features).all():
        raise ValueError('Coordinates must be nonnegative and features finite')
    if len(torch.unique(coords, dim=0)) != len(coords):
        raise ValueError('Sparse coordinates must be unique within a batch')


def pack_children(coords, features):
    """Return parent coordinates, zero-filled (P,8,C), and occupancy (P,8)."""
    _validate(coords, features)
    parent = coords.clone()
    parent[:, 1:] = torch.div(parent[:, 1:], 2, rounding_mode='floor')
    parents, inverse = torch.unique(parent, dim=0, sorted=True, return_inverse=True)
    octants = ((coords[:, 1:] % 2) * coords.new_tensor([4, 2, 1])).sum(dim=1)
    dense = features.new_zeros((len(parents), 8, features.shape[1]))
    dense[inverse, octants] = features
    occupancy = torch.zeros((len(parents), 8), dtype=torch.bool, device=coords.device)
    occupancy[inverse, octants] = True
    return parents, dense, occupancy


def downsample_shortcut(coords, features, out_channels):
    parents, dense, occupancy = pack_children(coords, features)
    if out_channels <= 0 or 8*features.shape[1] % out_channels:
        raise ValueError('Output channels must divide eight times the input channels')
    coarse = dense.reshape(len(parents), out_channels, 8*features.shape[1]//out_channels).mean(-1)
    return parents, coarse, occupancy


def _expanded(features, out_channels):
    if out_channels <= 0 or 8*out_channels % features.shape[1]:
        raise ValueError('Input channels must divide eight times the output channels')
    return features.repeat_interleave(8*out_channels//features.shape[1], dim=1).reshape(len(features), 8, out_channels)


def _unpack(coords, dense, mask):
    if mask.shape != (len(coords), 8) or mask.dtype != torch.bool or mask.device != coords.device:
        raise ValueError('Occupancy mask must be same-device boolean (P,8)')
    parents, octants = torch.where(mask)
    offsets = torch.stack((octants//4, octants//2 % 2, octants % 2), dim=1)
    children = coords[parents].clone()
    children[:, 1:] = children[:, 1:]*2 + offsets
    # Restore lexicographic order so masks from the next finer encoder level
    # remain aligned when chaining multiple decoder stages.
    _, inverse = torch.unique(children, dim=0, sorted=True, return_inverse=True)
    order = torch.argsort(inverse)
    return children[order], dense[parents, octants][order]


def upsample_shortcut(coords, features, out_channels, occupancy):
    _validate(coords, features)
    return _unpack(coords, _expanded(features, out_channels), occupancy)


class SparseDownsample(nn.Module):
    """Learned residual plus nonparametric space-to-channel shortcut."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or 8*in_channels % out_channels:
            raise ValueError('Incompatible shortcut channel dimensions')
        self.out_channels = out_channels
        self.residual = nn.Linear(8*in_channels, out_channels)

    def forward(self, coords, features):
        parents, dense, occupancy = pack_children(coords, features)
        flat = dense.flatten(1)
        shortcut = flat.reshape(len(parents), self.out_channels, flat.shape[1]//self.out_channels).mean(-1)
        return parents, self.residual(flat) + shortcut, occupancy


class SparseUpsample(nn.Module):
    """Learned child prediction with occupancy logits and pruning before output.

    Pass teacher occupancy during reconstruction training; omit it for predicted
    masks (sigmoid > threshold). Discrete mask selection has no gradient, so
    occupancy logits require their own supervised loss.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or 8*out_channels % in_channels:
            raise ValueError('Incompatible shortcut channel dimensions')
        self.out_channels = out_channels
        self.residual = nn.Linear(in_channels, 8*out_channels)
        self.occupancy_head = nn.Linear(in_channels, 8)

    def forward(self, coords, features, occupancy=None, *, threshold=.5):
        _validate(coords, features)
        if not 0 <= threshold <= 1:
            raise ValueError('Threshold must be in [0,1]')
        logits = self.occupancy_head(features)
        mask = logits.sigmoid() > threshold if occupancy is None else occupancy
        dense = self.residual(features).reshape(len(coords), 8, self.out_channels)
        dense = dense + _expanded(features, self.out_channels)
        children, values = _unpack(coords, dense, mask)
        return children, values, logits
