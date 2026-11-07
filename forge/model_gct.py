"""FORGE-Net geometry completion transformer (GCT).

Grid-point dual-stream transformer:
  - 3D conv stem encodes raw 64^3 occupancy into 8^3 grid tokens (+ skips)
  - 512 surface keypoint tokens inject the raw mesh's fine detail
  - full self-attention with 3D axial RoPE over grid/point/register tokens
  - conv decoder upsamples to a 64^3 feature field
  - continuous SDF head + coarse occupancy head
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from t3d.flow import GramSchmidtRoPE

from .constants import GRID, FINE


class Fourier(nn.Module):
    def __init__(self, levels=4):
        super().__init__()
        self.register_buffer('freqs', 2.0 ** torch.arange(levels, dtype=torch.float32) * math.pi)

    def forward(self, x):
        projected = x[..., None] * self.freqs
        return torch.cat([torch.sin(projected).flatten(-2),
                          torch.cos(projected).flatten(-2)], dim=-1)


def _conv_block(cin, cout):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1, bias=False),
                         nn.GroupNorm(min(8, cout), cout), nn.SiLU())


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.norm2(self.conv2(self.act(self.norm1(self.conv1(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if (width // heads) % 6:
            raise ValueError('GCT head_dim must be divisible by 6 for axial RoPE')
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.rope = GramSchmidtRoPE(heads, width // heads)
        self.out = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))

    def forward(self, x, coords, pad_mask=None):
        batch, tokens, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, tokens, 3, self.heads,
                                              width // self.heads)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        q_rope, k_rope = [], []
        for index in range(batch):
            q_rope.append(self.rope(q[index], coords[index]))
            k_rope.append(self.rope(k[index], coords[index]))
        q = torch.stack(q_rope)
        k = torch.stack(k_rope)
        attention = F.scaled_dot_product_attention(q, k, v)
        attention = attention.transpose(1, 2).reshape(batch, tokens, width)
        x = x + self.out(attention)
        x = x + self.mlp(self.norm2(x))
        return x


class ConvStem(nn.Module):
    def __init__(self, widths=(32, 64, 96, 128)):
        super().__init__()
        self.entry = _conv_block(1, widths[0])
        self.res64 = ResBlock(widths[0])
        self.down1 = nn.Sequential(nn.Conv3d(widths[0], widths[1], 3, stride=2, padding=1,
                                             bias=False),
                                   nn.GroupNorm(8, widths[1]), nn.SiLU())
        self.res32 = ResBlock(widths[1])
        self.down2 = nn.Sequential(nn.Conv3d(widths[1], widths[2], 3, stride=2, padding=1,
                                             bias=False),
                                   nn.GroupNorm(8, widths[2]), nn.SiLU())
        self.res16 = ResBlock(widths[2])
        self.down3 = nn.Sequential(nn.Conv3d(widths[2], widths[3], 3, stride=2, padding=1,
                                             bias=False),
                                   nn.GroupNorm(8, widths[3]), nn.SiLU())
        self.res8 = ResBlock(widths[3])
        self.widths = widths

    def forward(self, occ):
        x64 = self.res64(self.entry(occ))
        x32 = self.res32(self.down1(x64))
        x16 = self.res16(self.down2(x32))
        x8 = self.res8(self.down3(x16))
        return x8, x16, x32, x64


class ConvDecoder(nn.Module):
    def __init__(self, width, stem_widths=(32, 64, 96, 128), feature=48):
        super().__init__()
        self.project = nn.Conv3d(width, 96, 1)
        self.up1 = nn.Conv3d(96 + stem_widths[2], 96, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, 96)
        self.res1 = ResBlock(96)
        self.up2 = nn.Conv3d(96 + stem_widths[1], 64, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, 64)
        self.res2 = ResBlock(64)
        self.up3 = nn.Conv3d(64 + stem_widths[0], feature, 3, padding=1, bias=False)
        self.norm3 = nn.GroupNorm(8, feature)
        self.res3 = ResBlock(feature)
        self.feature = feature

    def forward(self, tokens, skips):
        x8 = self.project(tokens)
        x16 = F.interpolate(x8, scale_factor=2, mode='nearest')
        x16 = self.res1(self.norm1(self.up1(torch.cat([x16, skips[0]], dim=1))))
        x32 = F.interpolate(x16, scale_factor=2, mode='nearest')
        x32 = self.res2(self.norm2(self.up2(torch.cat([x32, skips[1]], dim=1))))
        x64 = F.interpolate(x32, scale_factor=2, mode='nearest')
        x64 = self.res3(self.norm3(self.up3(torch.cat([x64, skips[2]], dim=1))))
        return x64


class GeometryCompletionTransformer(nn.Module):
    def __init__(self, width=384, heads=8, depth=8, point_tokens=512, registers=16,
                 stem_widths=(32, 64, 96, 128), feature=48, fourier_levels=4,
                 sdf_hidden=256):
        super().__init__()
        self.width = width
        self.stem = ConvStem(stem_widths)
        self.tokens = nn.Conv3d(stem_widths[3], width, 1)
        self.point_embed = nn.Sequential(
            nn.Linear(6 + 3 * 2 * fourier_levels, width), nn.GELU(),
            nn.Linear(width, width))
        self.point_fourier = Fourier(fourier_levels)
        self.grid_type = nn.Parameter(torch.zeros(1, 1, width))
        self.point_type = nn.Parameter(torch.zeros(1, 1, width))
        self.register_tokens = nn.Parameter(torch.zeros(1, registers, width))
        self.register_type = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList([TransformerBlock(width, heads) for _ in range(depth)])
        self.decoder = ConvDecoder(width, stem_widths, feature)
        self.occ_head = nn.Conv3d(feature, 1, 1)
        self.sdf_fourier = Fourier(fourier_levels)
        self.sdf_head = nn.Sequential(
            nn.Linear(feature + 3 * 2 * fourier_levels + 3, sdf_hidden), nn.SiLU(),
            nn.Linear(sdf_hidden, sdf_hidden), nn.SiLU(), nn.Linear(sdf_hidden, 1))
        self.fourier_levels = fourier_levels
        self.point_tokens = point_tokens
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        nn.init.zeros_(self.grid_type)
        nn.init.zeros_(self.point_type)
        nn.init.zeros_(self.register_type)

    def _grid_coords(self, device, dtype, dim=8):
        axis = (torch.arange(dim, device=device, dtype=dtype) + 0.5) / dim - 0.5
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), dim=-1)
        return grid.reshape(-1, 3)

    def encode(self, occ, points):
        """occ (B,1,64,64,64) float; points (B,P,6) xyz+normals. Returns feat, occ_logits."""
        batch = occ.shape[0]
        x8, x16, x32, x64 = self.stem(occ)
        grid = self.tokens(x8).flatten(2).transpose(1, 2)
        device, dtype = grid.device, grid.dtype
        grid_coords = self._grid_coords(device, dtype, 8) * 8.0
        selected = points[:, ::max(1, points.shape[1] // self.point_tokens)][:, :self.point_tokens]
        point_features = self.point_embed(
            torch.cat([selected[..., :3], selected[..., 3:], self.point_fourier(selected[..., :3])],
                      dim=-1))
        point_coords = selected[..., :3] * GRID
        registers = self.register_tokens.expand(batch, -1, -1)
        register_coords = torch.zeros(batch, registers.shape[1], 3, device=device, dtype=dtype)
        tokens = torch.cat([grid + self.grid_type, point_features + self.point_type,
                            registers + self.register_type], dim=1)
        coords = torch.cat([grid_coords.unsqueeze(0).expand(batch, -1, -1),
                            point_coords, register_coords], dim=1)
        for block in self.blocks:
            tokens = block(tokens, coords)
        grid_tokens = tokens[:, :8 ** 3].transpose(1, 2).reshape(batch, self.width, 8, 8, 8)
        feature_field = self.decoder(grid_tokens, (x16, x32, x64))
        occ_logits = self.occ_head(feature_field)
        return feature_field, occ_logits

    def sdf(self, feature_field, query):
        """query (B,N,3) in [-0.5,0.5]. Returns signed distance (B,N)."""
        batch, channels = feature_field.shape[:2]
        grid = 2.0 * (query + 0.5) - 1.0
        sampled = F.grid_sample(feature_field, grid.reshape(batch, -1, 1, 1, 3),
                                mode='bilinear', padding_mode='border', align_corners=False)
        sampled = sampled[..., 0, 0].transpose(1, 2)
        features = torch.cat([sampled, self.sdf_fourier(query), query], dim=-1)
        return self.sdf_head(features)[..., 0]

    def forward(self, occ, points, query):
        feature_field, occ_logits = self.encode(occ, points)
        return occ_logits, self.sdf(feature_field, query)
