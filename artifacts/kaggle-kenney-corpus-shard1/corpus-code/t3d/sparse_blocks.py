"""PyTorch submanifold ConvNeXt reference; optimized Triton kernels pending."""
from itertools import product
import math
import torch
from torch import nn
from .sparse_compression import _validate


class SubmanifoldConv3d(nn.Module):
    """3x3x3 convolution evaluated only at supplied active coordinates.

    Missing neighbors contribute zero. Coordinate keys include the batch index.
    Uses sorted-key search on CPU/CUDA and retains input coordinate row order.
    """
    def __init__(self, channels, *, backend='reference'):
        super().__init__()
        if channels <= 0:
            raise ValueError('Channels must be positive')
        if backend not in ('reference', 'triton'):
            raise ValueError('Unknown sparse convolution backend')
        self.channels, self.backend = channels, backend
        self.weight = nn.Parameter(torch.empty(27, channels, channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.normal_(self.weight, std=(27*channels)**-.5)

    def forward(self, coords, features):
        if self.backend == 'triton':
            from .triton_sparse import triton_submanifold_conv
            return triton_submanifold_conv(coords, features, self.weight, self.bias)
        _validate(coords, features)
        if features.shape[1] != self.channels:
            raise ValueError('Convolution channel mismatch')
        if len(coords) == 0:
            return features + self.bias
        dimensions = [int(v)+1 for v in coords.max(dim=0).values.tolist()]
        if math.prod(dimensions) > torch.iinfo(torch.int64).max:
            raise ValueError('Coordinate range exceeds int64 key capacity')
        strides = coords.new_tensor([math.prod(dimensions[1:]), math.prod(dimensions[2:]), dimensions[3], 1])
        keys = (coords * strides).sum(dim=1)
        sorted_keys, order = keys.sort()
        output = features.new_zeros(features.shape)
        for i, offset in enumerate(product((-1, 0, 1), repeat=3)):
            queries = coords + coords.new_tensor([0, *offset])
            valid = ((queries >= 0) & (queries < coords.new_tensor(dimensions))).all(dim=1)
            query_keys = (queries * strides).sum(dim=1)
            positions = torch.searchsorted(sorted_keys, query_keys).clamp(max=len(coords)-1)
            valid = valid & (sorted_keys[positions] == query_keys)
            # Mask gathered inputs; no cross-boundary or cross-batch values leak.
            neighbors = features[order[positions]] * valid[:, None]
            output = output + neighbors @ self.weight[i]
        return output + self.bias


class SparseConvNeXtBlock(nn.Module):
    """One submanifold spatial convolution plus wide pointwise residual MLP."""
    def __init__(self, channels, expansion=4, layer_scale=1e-6, *, backend='reference'):
        super().__init__()
        if expansion <= 0:
            raise ValueError('Expansion must be positive')
        self.spatial = SubmanifoldConv3d(channels, backend=backend)
        self.norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels*expansion), nn.GELU(),
                                 nn.Linear(channels*expansion, channels))
        self.scale = nn.Parameter(torch.full((channels,), layer_scale))

    def forward(self, coords, features):
        return features + self.scale*self.mlp(self.norm(self.spatial(coords, features)))
