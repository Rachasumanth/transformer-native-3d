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
    def __init__(self, mode='shape', channels=(32, 64, 128, 256, 512), latent_channels=16):
        super().__init__()
        if mode not in ('shape', 'material') or len(channels) != 5 or min(channels) <= 0 or latent_channels <= 0:
            raise ValueError('Specify shape/material mode, five positive widths, and positive latent channels')
        self.mode, self.channels = mode, tuple(channels)
        inputs = 7 if mode == 'shape' else 6
        self.input = nn.Linear(inputs, channels[0])
        self.encoder_blocks = nn.ModuleList([SparseConvNeXtBlock(c) for c in channels])
        self.down = nn.ModuleList([SparseDownsample(channels[i], channels[i+1]) for i in range(4)])
        self.moments = nn.Linear(channels[-1], 2*latent_channels)
        self.from_latent = nn.Linear(latent_channels, channels[-1])
        self.decoder_blocks = nn.ModuleList([SparseConvNeXtBlock(c) for c in reversed(channels)])
        self.up = nn.ModuleList([SparseUpsample(channels[i+1], channels[i]) for i in reversed(range(4))])
        self.output = nn.Linear(channels[0], inputs)

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

    def decode(self, coords, latent, *, teacher_occupancy=None, threshold=.5, max_active=1000000,trace=None):
        _validate(coords, latent)
        if teacher_occupancy is not None and len(teacher_occupancy) != 4:
            raise ValueError('Teacher occupancy must have four encoder levels')
        if trace is not None and not isinstance(trace,list):
            raise ValueError('trace must be a list')
        features = self.from_latent(latent)
        logits = []
        for i, up in enumerate(self.up):
            features = self.decoder_blocks[i](coords, features)
            teacher = None if teacher_occupancy is None else teacher_occupancy[3-i]
            # Bound potential child allocation before the upsampler executes.
            if len(coords)*8 > max_active:
                raise ValueError('Decoder child allocation exceeds max_active')
            parents=coords
            coords, features, prediction = up(coords, features, teacher, threshold=threshold)
            if trace is not None:
                selected=prediction.sigmoid()>threshold if teacher is None else teacher
                trace.append(dict(level=i,parents=parents.detach().clone(),logits=prediction.detach().clone(),
                                  selected=selected.detach().clone(),children=coords.detach().clone(),
                                  teacher_forced=teacher is not None))
            logits.append(prediction)
        features = self.decoder_blocks[4](coords, features)
        return coords, self.output(features), logits

    def forward(self, coords, features, *, stochastic=True):
        posterior = self.encode(coords, features)
        output = self.decode(posterior.coords, posterior.sample(stochastic),
                             teacher_occupancy=posterior.occupancy)
        return posterior, output


def reconstruction_loss(model, coords, features, *, stochastic=True, kl_weight=1e-6, occupancy_weight=1.,
                        edge_positive_weight=None,render_fn=None,render_weight=0.):
    """Shape: bounded dual xyz, edge BCE, split MSE; material: bounded PBR L1.

    Teacher masks reconstruct source support; they do not pass geometry features
    around the bottleneck. Decoder coordinates must exactly match sorted source
    coordinates. Values and masks are supervised independently of the KL term.
    """
    if kl_weight < 0 or occupancy_weight < 0 or render_weight < 0:
        raise ValueError('Loss weights must be nonnegative')
    posterior, (out_coords, raw, logits) = model(coords, features, stochastic=stochastic)
    sorted_coords, inverse = torch.unique(coords, dim=0, sorted=True, return_inverse=True)
    if not torch.equal(out_coords, sorted_coords):
        raise ValueError('Decoded support differs from source support')
    target = features[torch.argsort(inverse)]
    if model.mode == 'shape':
        if ((target[:, :6] < 0) | (target[:, :6] > 1)).any():
            raise ValueError('Dual coordinates and edge targets must be in [0,1]')
        weight=None if edge_positive_weight is None else torch.as_tensor(edge_positive_weight,device=raw.device,dtype=raw.dtype)
        if weight is not None and (weight.shape not in (torch.Size([]),torch.Size([3])) or not torch.isfinite(weight).all() or (weight<=0).any()):
            raise ValueError('edge_positive_weight must be finite positive scalar or three-vector')
        attribute = (F.mse_loss(raw[:, :3].sigmoid(), target[:, :3])
                     + F.binary_cross_entropy_with_logits(raw[:, 3:6], target[:, 3:6],pos_weight=weight)
                     + F.mse_loss(raw[:, 6:], target[:, 6:]))
    else:
        if ((target < 0) | (target > 1)).any():
            raise ValueError('PBR targets must be in [0,1]')
        attribute = F.l1_loss(raw.sigmoid(), target)
    occupancy = sum(F.binary_cross_entropy_with_logits(prediction, mask.to(prediction.dtype))
                    for prediction, mask in zip(logits, reversed(posterior.occupancy)))/4
    kl = posterior.kl()
    total = attribute + occupancy_weight*occupancy + kl_weight*kl
    parts=dict(attribute=attribute, occupancy=occupancy, kl=kl)
    if render_weight:
        if model.mode!='shape' or render_fn is None:raise ValueError('shape render callback required')
        render=render_fn(out_coords,target,raw)
        total=total+render_weight*render
        parts['render']=render
    return total, parts
