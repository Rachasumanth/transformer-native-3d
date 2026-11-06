"""Reference SkinAE foundation following AniGen section 3.3.1.

Widths, Fourier frequencies and temperature floor are local implementation
choices; this is not an upstream checkpoint-compatible reproduction.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SkinAE(nn.Module):
    def __init__(self, width=64, heads=4, layers=2, frequencies=4):
        super().__init__()
        if min(width, heads, layers, frequencies) <= 0 or width % heads:
            raise ValueError('Positive dimensions and width divisible by heads required')
        self.register_buffer('frequencies', 2.**torch.arange(frequencies)*math.pi)
        self.input = nn.Linear(2*(3+6*frequencies), width)
        layer = nn.TransformerEncoderLayer(width, heads, width*4, dropout=0., batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.to_joint = nn.Linear(width, 4)
        self.joint_decoder = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 64))
        self.vertex_decoder = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 64))
        self.temperature = nn.Linear(4, 1)

    def _pe(self, value):
        phase = value[..., None]*self.frequencies
        return torch.cat((value, phase.sin().flatten(-2), phase.cos().flatten(-2)), dim=-1)

    def encode_joints(self, joints, parents):
        if (joints.ndim != 2 or joints.shape[1] != 3 or not len(joints)
                or not joints.is_floating_point() or not torch.isfinite(joints).all()):
            raise ValueError('Finite floating joints (J,3) required')
        if (parents.shape != (len(joints),) or parents.dtype != torch.int64
                or parents.device != joints.device or (parents < -1).any() or (parents >= len(joints)).any()):
            raise ValueError('Int64 parents on joint device required; roots use -1')
        links = parents.detach().cpu().tolist()
        finished = set()
        for start in range(len(links)):
            current, seen = start, set()
            while current != -1 and current not in finished:
                if current in seen:
                    raise ValueError('Skeleton contains a parent cycle')
                seen.add(current); current = links[current]
            finished.update(seen)
        parent = torch.where(parents < 0, torch.arange(len(joints), device=joints.device), parents)
        edges = joints-joints[parent]
        # Parent edge features supply structure without joint-index positional IDs.
        features = torch.cat((self._pe(edges), self._pe(edges[parent])), dim=-1)
        return self.to_joint(self.encoder(self.input(features)[None])[0])

    @staticmethod
    def encode_vertices(weights, joint_embeddings):
        if (weights.ndim != 2 or joint_embeddings.ndim != 2 or joint_embeddings.shape[1] != 4
                or weights.shape[1] != len(joint_embeddings) or not len(joint_embeddings)
                or not weights.is_floating_point() or not torch.isfinite(weights).all()
                or (weights < 0).any() or not torch.allclose(weights.sum(-1), torch.ones_like(weights[:, 0]), atol=1e-6, rtol=0)):
            raise ValueError('Normalized nonnegative skin weights (N,J) and joint embeddings (J,4) required')
        return weights @ joint_embeddings

    def decode_logits(self, vertex_embeddings, joint_embeddings):
        for value in (vertex_embeddings, joint_embeddings):
            if value.ndim != 2 or value.shape[1] != 4 or not torch.isfinite(value).all():
                raise ValueError('Finite embeddings (N,4) required')
        if not len(joint_embeddings):
            raise ValueError('At least one joint required')
        temperature = F.softplus(self.temperature(vertex_embeddings))+1e-4
        return (self.vertex_decoder(vertex_embeddings) @ self.joint_decoder(joint_embeddings).T)/temperature

    def forward(self, joints, parents, weights):
        joint = self.encode_joints(joints, parents)
        vertex = self.encode_vertices(weights, joint)
        return joint, vertex, self.decode_logits(vertex, joint)

    def freeze(self):
        self.eval()
        self.requires_grad_(False)
        return self


def skin_reconstruction_loss(logits, target):
    """Mean per-vertex KL(target || predicted); zero target entries are supported."""
    if (target.shape != logits.shape or target.ndim != 2 or not len(target)
            or not torch.isfinite(target).all() or (target < 0).any()
            or not torch.allclose(target.sum(-1), torch.ones_like(target[:, 0]), atol=1e-6, rtol=0)):
        raise ValueError('Nonempty normalized skin targets matching logits required')
    return F.kl_div(F.log_softmax(logits, dim=-1), target, reduction='batchmean')
