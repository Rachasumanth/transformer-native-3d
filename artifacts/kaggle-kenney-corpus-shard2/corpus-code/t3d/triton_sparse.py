"""Experimental indexed submanifold convolution; PyTorch remains the default.

The neighbor map and backward pass are PyTorch. Only the forward multiply and
sum run in Triton. Benchmark and gradient parity are required before wider use.
"""
from itertools import product
import math
import torch
from .sparse_compression import _validate

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _submanifold_forward(X, W, BIAS, MAP, OUT,
                             N: tl.constexpr, C: tl.constexpr,
                             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0., tl.float32)
        for offset in range(27):
            neighbor = tl.load(MAP + rows * 27 + offset, mask=rows < N, other=-1)
            for start in range(tl.cdiv(C, BK)):
                k = start * BK + kk
                a = tl.load(X + tl.maximum(neighbor[:, None], 0) * C + k[None, :],
                            mask=(rows[:, None] < N) & (neighbor[:, None] >= 0) & (k[None, :] < C), other=0.)
                b = tl.load(W + offset*C*C + k[:, None]*C + cols[None, :],
                            mask=(k[:, None] < C) & (cols[None, :] < C), other=0.)
                acc += tl.dot(a, b, input_precision='ieee')
        bias = tl.load(BIAS+cols, mask=cols<C, other=0.)
        tl.store(OUT + rows[:, None]*C + cols[None, :], acc+bias[None, :],
                 mask=(rows[:, None]<N)&(cols[None, :]<C))


def neighbor_map(coords):
    """Map each active coordinate to its 27 same-batch neighbors (-1 if absent)."""
    if coords.ndim != 2 or coords.shape[1] != 4 or coords.dtype != torch.int64 or len(coords)==0:
        raise ValueError('Expected nonempty int64 (N,4) coordinates')
    dimensions = [int(v)+1 for v in coords.max(dim=0).values.tolist()]
    if math.prod(dimensions)>torch.iinfo(torch.int64).max:
        raise ValueError('Coordinate range exceeds int64 key capacity')
    shape=coords.new_tensor(dimensions)
    strides=coords.new_tensor([math.prod(dimensions[1:]),math.prod(dimensions[2:]),dimensions[3],1])
    keys=(coords*strides).sum(dim=1);sorted_keys,order=keys.sort();result=[]
    for offset in product((-1,0,1),repeat=3):
        query=coords+coords.new_tensor([0,*offset]);valid=((query>=0)&(query<shape)).all(dim=1)
        key=(query*strides).sum(dim=1);position=torch.searchsorted(sorted_keys,key).clamp(max=len(coords)-1)
        valid=valid&(sorted_keys[position]==key)
        result.append(torch.where(valid,order[position],-1))
    return torch.stack(result,dim=1).contiguous()


class _IndexedConvolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weight, bias, neighbors):
        n,c=features.shape
        output=torch.empty_like(features)
        _submanifold_forward[(triton.cdiv(n,16),triton.cdiv(c,32))](features,weight,bias,neighbors,output,n,c,16,32,32)
        ctx.save_for_backward(features,weight,neighbors)
        return output

    @staticmethod
    def backward(ctx, grad):
        features,weight,neighbors=ctx.saved_tensors
        grad=grad.contiguous();grad_features=torch.zeros_like(features);grad_weight=torch.zeros_like(weight)
        for offset in range(27):
            ids=neighbors[:,offset];valid=ids>=0;src=ids[valid]
            if src.numel():
                go=grad[valid]
                grad_weight[offset]=features[src].T@go
                grad_features.index_add_(0,src,go@weight[offset].T)
        return grad_features,grad_weight,grad.sum(dim=0),None


def triton_submanifold_conv(coords, features, weight, bias):
    """Experimental float32 CUDA reference-compatible forward/backward."""
    _validate(coords,features)
    c=features.shape[1]
    if triton is None or not features.is_cuda:
        raise RuntimeError('Triton and CUDA are required')
    if (features.dtype!=torch.float32 or weight.dtype!=torch.float32 or bias.dtype!=torch.float32
            or weight.shape!=(27,c,c) or bias.shape!=(c,) or weight.device!=features.device or bias.device!=features.device):
        raise ValueError('Expected same-device float32 features, 27xCxC weights and C bias')
    if len(coords)==0:
        return features+bias
    return _IndexedConvolution.apply(features.contiguous(),weight.contiguous(),bias.contiguous(),neighbor_map(coords))
