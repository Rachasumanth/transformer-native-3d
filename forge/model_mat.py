"""FORGE-Net material transformer (MAT).

Predicts per-surface-point PBR from raw-input shape context and a degraded
texture hint. Local window attention follows a z-order sort of the query
points; learned global tokens pass information across the asset.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .constants import GRID


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


class ShapeContext(nn.Module):
    """Small 3D U-Net producing a 32-channel feature field at 64^3."""

    def __init__(self, feature=32):
        super().__init__()
        self.entry = _conv_block(1, 32)
        self.down1 = _conv_block(32, 64)
        self.down2 = _conv_block(64, 96)
        self.up1 = _conv_block(96 + 64, 64)
        self.up2 = _conv_block(64 + 32, feature)
        self.pool = nn.MaxPool3d(2)
        self.feature = feature

    def forward(self, occ):
        x64 = self.entry(occ)
        x32 = self.down1(self.pool(x64))
        x16 = self.down2(self.pool(x32))
        y32 = self.up1(torch.cat([F.interpolate(x16, scale_factor=2, mode='nearest'), x32], 1))
        y64 = self.up2(torch.cat([F.interpolate(y32, scale_factor=2, mode='nearest'), x64], 1))
        return y64


def _project(value, width):
    return nn.Sequential(nn.Linear(value, width), nn.GELU(), nn.Linear(width, width))


class CrossAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, width)

    def forward(self, query, context):
        batch = query.shape[0]
        q = self.q(query).reshape(batch, -1, self.heads, query.shape[-1] // self.heads)
        kv = self.kv(context).reshape(batch, -1, 2, self.heads,
                                      context.shape[-1] // self.heads)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        q = q.transpose(1, 2)
        attention = F.scaled_dot_product_attention(q, k, v)
        attention = attention.transpose(1, 2).reshape(batch, -1, query.shape[-1])
        return self.out(attention)


class WindowAttention(nn.Module):
    def __init__(self, width, heads, window=128):
        super().__init__()
        self.heads = heads
        self.window = window
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x, valid):
        batch, original, width = x.shape
        window = self.window
        has_pad = original % window
        if has_pad:
            pad = window - has_pad
            x = F.pad(x, (0, 0, 0, pad))
            valid = F.pad(valid, (0, pad), value=False)
        tokens = x.shape[1]
        windows = tokens // window
        qkv = self.qkv(x).reshape(batch, windows, window, 3, self.heads,
                                  width // self.heads)
        q, k, v = (qkv[..., i, :, :].permute(0, 1, 3, 2, 4) for i in range(3))
        q = q.reshape(batch * windows, self.heads, window, width // self.heads)
        k = k.reshape(batch * windows, self.heads, window, width // self.heads)
        v = v.reshape(batch * windows, self.heads, window, width // self.heads)
        mask = valid.reshape(batch * windows, 1, 1, window)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attention = attention.reshape(batch, windows, self.heads, window,
                                      width // self.heads)
        attention = attention.permute(0, 1, 3, 2, 4).reshape(batch, tokens, width)
        return self.out(attention)[:, :original]


class MATBlock(nn.Module):
    def __init__(self, width, heads, window=128, globals=8):
        super().__init__()
        self.local_norm = nn.LayerNorm(width)
        self.local = WindowAttention(width, heads, window)
        self.global_norm = nn.LayerNorm(width)
        self.token_norm = nn.LayerNorm(width)
        self.global_from_tokens = CrossAttention(width, heads)
        self.tokens_from_global = CrossAttention(width, heads)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))
        self.mlp_norm = nn.LayerNorm(width)
        self.global_mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                        nn.Linear(4 * width, width))
        self.global_mlp_norm = nn.LayerNorm(width)

    def forward(self, tokens, global_tokens, valid):
        tokens = tokens + self.local(self.local_norm(tokens), valid)
        global_tokens = global_tokens + self.global_from_tokens(
            self.global_norm(global_tokens), self.token_norm(tokens))
        tokens = tokens + self.tokens_from_global(self.token_norm(tokens),
                                                  self.global_norm(global_tokens))
        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        global_tokens = global_tokens + self.global_mlp(self.global_mlp_norm(global_tokens))
        return tokens, global_tokens


class MaterialTransformer(nn.Module):
    def __init__(self, width=256, heads=8, depth=6, window=128, globals=8,
                 context=32, fourier_levels=4):
        super().__init__()
        self.context = ShapeContext(context)
        self.fourier = Fourier(fourier_levels)
        self.embed = nn.Sequential(
            nn.Linear(context + 3 * 2 * fourier_levels + 3 + 3 + 3 + 1, width), nn.GELU(),
            nn.Linear(width, width))
        self.globals = nn.Parameter(torch.zeros(1, globals, width))
        self.blocks = nn.ModuleList([MATBlock(width, heads, window, globals)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 6)
        self.window = window
        nn.init.trunc_normal_(self.globals, std=0.02)

    def _sort_order(self, coords):
        index = torch.clamp(((coords + 0.5) * GRID).long(), 0, GRID - 1)
        key = (index[..., 0] << 12) | (index[..., 1] << 6) | index[..., 2]
        return torch.argsort(key, dim=1)

    def forward(self, occ, coords, normals, hint, has_tex):
        batch, tokens = coords.shape[:2]
        feature = self.context(occ)
        grid = 2.0 * (coords + 0.5) - 1.0
        sampled = F.grid_sample(feature, grid.reshape(batch, -1, 1, 1, 3),
                                mode='bilinear', padding_mode='border', align_corners=False)
        sampled = sampled[..., 0, 0].transpose(1, 2)
        has_tex = has_tex.reshape(batch, 1, 1).expand(batch, tokens, 1)
        embedded = self.embed(torch.cat([sampled, self.fourier(coords), coords, normals,
                                         hint, has_tex], dim=-1))
        order = self._sort_order(coords)
        inverse = torch.argsort(order, dim=1)
        gather = order[..., None].expand(-1, -1, embedded.shape[-1])
        tokens_sorted = torch.gather(embedded, 1, gather)
        valid = torch.ones(batch, tokens, dtype=torch.bool, device=coords.device)
        global_tokens = self.globals.expand(batch, -1, -1)
        for block in self.blocks:
            tokens_sorted, global_tokens = block(tokens_sorted, global_tokens, valid)
        output_sorted = self.head(self.norm(tokens_sorted))
        output = torch.gather(output_sorted, 1,
                              inverse[..., None].expand(-1, -1, output_sorted.shape[-1]))
        return torch.sigmoid(output)
