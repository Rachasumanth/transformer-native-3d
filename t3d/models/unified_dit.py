"""Unified Dense Diffusion Transformer (DiT) for 3D generation.

Unifies coarse occupancy prediction and continuous latent refinement
into a single joint state representation per voxel cell:
    x_i = [o_i, z_i] in R^(C+1)
where o_i in R is the occupancy logit/state and z_i in R^C is the SC-VAE
continuous latent (typically C=16).

Conditioned on raw mesh features:
    c_raw in R^(N x (C+1))
concatenated directly along the channel dimension.

Features 3D Rotary Position Embeddings (3D RoPE) decomposed across attention
head dimensions, adaptive layer normalization (AdaLN-Zero) conditioned on
diffusion timestep t in [0, 1], and Classifier-Free Guidance (CFG) / Interval Guidance.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def _unit(vector: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """Normalize vector along last dimension, returning fallback if zero-norm."""
    norm = vector.norm(dim=-1, keepdim=True)
    return torch.where(norm > 1e-6, vector / norm.clamp_min(1e-6), fallback)


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Compute continuous sinusoidal embeddings for timesteps in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / max(1, half - 1)
    )
    # Scale timesteps by 1000 to span the frequency spectrum
    args = (timesteps.float() * 1000.0)[:, None] * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(dtype=timesteps.dtype) if timesteps.is_floating_point() else embedding


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive scale and shift modulation."""
    return x * (1.0 + scale) + shift


class RoPE3D(nn.Module):
    """3D Rotary Position Embeddings decomposed across attention head dimensions.
    
    Rotates query and key token representations according to spatial 3D
    coordinates (x, y, z) using an orthonormal 3D frame per head.
    Preserves L2 norms and guarantees spatial translation invariance:
        Score(Q_rot(p + Delta), K_rot(p + Delta)) == Score(Q_rot(p), K_rot(p)).
    """
    def __init__(self, heads: int, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 6 != 0:
            raise ValueError(f"3D axial RoPE head dimension must be divisible by 6, got {head_dim}")
        self.heads = heads
        self.head_dim = head_dim
        # Initial canonical orthonormal axes: axis 0 -> [1, 0, 0], axis 1 -> [0, 1, 0]
        initial = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).repeat(heads, 1, 1)
        self.axes = nn.Parameter(initial)
        num_freqs = head_dim // 6
        self.register_buffer(
            "frequencies",
            base ** (-torch.arange(num_freqs, dtype=torch.float32) / max(1, num_freqs))
        )

    def rotation(self) -> torch.Tensor:
        """Compute orthonormal 3x3 rotation frame per head via Gram-Schmidt."""
        u = _unit(self.axes[:, 0], self.axes.new_tensor([1.0, 0.0, 0.0]).expand(len(self.axes), 3))
        raw = self.axes[:, 1] - (self.axes[:, 1] * u).sum(-1, keepdim=True) * u
        basis = F.one_hot(u.abs().argmin(-1), 3).to(u.dtype)
        fallback = _unit(basis - (basis * u).sum(-1, keepdim=True) * u, basis)
        v = _unit(raw, fallback)
        w = torch.cross(u, v, dim=-1)
        return torch.stack((u, v, w), dim=1)

    def forward(self, value: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Apply 3D rotary embeddings.
        
        Args:
            value: (B, heads, N, head_dim) or (heads, N, head_dim)
            coords: (B, N, 3) or (N, 3)
            
        Returns:
            Rotated value tensor with identical shape and preserved norm.
        """
        unbatched = (value.ndim == 3)
        if unbatched:
            value = value.unsqueeze(0)
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)

        B, H, N, D = value.shape
        rot = self.rotation().to(value.dtype)  # (H, 3, 3)

        # coords: (B_c, N, 3), rot: (H, 3, 3) -> angles: (B_c, H, N, 3)
        angles = torch.einsum("bni,hij->bhni", coords.to(value.dtype), rot)
        # angles: (B_c, H, N, 3, num_freqs)
        angles = angles.unsqueeze(-1) * self.frequencies.to(value.dtype)

        # Reshape for 2D complex rotations along each of the 3 spatial axes
        paired = value.reshape(B, H, N, 3, -1, 2)
        a, b = paired[..., 0], paired[..., 1]
        rot_a = a * angles.cos() - b * angles.sin()
        rot_b = a * angles.sin() + b * angles.cos()
        out = torch.stack((rot_a, rot_b), dim=-1).flatten(3)

        if unbatched:
            out = out.squeeze(0)
        return out


class SelfAttention3D(nn.Module):
    """Multi-head self-attention with 3D Rotary Position Embeddings (3D RoPE)."""
    def __init__(self, dim: int, heads: int, qkv_bias: bool = True):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        head_dim = dim // heads
        if head_dim % 6 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by 6 for 3D RoPE")
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.rope = RoPE3D(heads, head_dim)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each is (B, heads, N, head_dim)
        q = self.rope(q, coords)
        k = self.rope(k, coords)
        attn = F.scaled_dot_product_attention(q, k, v)
        out = attn.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)


class DiTBlock(nn.Module):
    """DiT block with Adaptive Layer Normalization (AdaLN-Zero) and 3D RoPE attention."""
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention3D(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        # Initialize AdaLN-Zero modulation to zero
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # t_emb: (B, dim)
        mod = self.adaLN_modulation(t_emb).unsqueeze(1)  # (B, 1, 6*dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        # Attention branch
        normed = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(normed, coords)
        # MLP branch
        normed2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(normed2)
        return x


class FinalLayer(nn.Module):
    """DiT final layer with AdaLN modulation and zero-initialized linear projection."""
    def __init__(self, dim: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(t_emb).unsqueeze(1)
        shift, scale = mod.chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class UnifiedDiT(nn.Module):
    """Single Unified Dense Diffusion Transformer (DiT) for 3D Generation.
    
    Unifies coarse occupancy prediction and continuous latent refinement
    into a single model with joint state representation per voxel cell:
        x_i = [o_i, z_i] in R^(C+1)
    conditioned on raw mesh prior:
        c_raw in R^(N x (C+1))
    concatenated directly along the channel dimension.
    
    Features 3D Rotary Position Embeddings (3D RoPE), AdaLN-Zero conditioning
    on diffusion timestep t in [0, 1], and Classifier-Free Guidance (CFG).
    
    Args:
        dim: Hidden dimension (default 288). Must satisfy dim % heads == 0 and (dim // heads) % 6 == 0.
        depth: Number of DiT blocks (default 4).
        heads: Number of attention heads (default 6).
        latent_channels: Dimensionality of SC-VAE continuous latent (default 16).
        coarse_res: Coarse voxel resolution per axis (default 4 -> N = 4^3 = 64).
        cond_drop_prob: Unconditional conditioning dropout probability for CFG (default 0.1).
        mlp_ratio: Ratio of MLP hidden dimension to dim (default 4.0).
    """
    def __init__(
        self,
        dim: int = 288,
        depth: int = 4,
        heads: int = 6,
        latent_channels: int = 16,
        coarse_res: int = 4,
        cond_drop_prob: float = 0.1,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        head_dim = dim // heads
        if head_dim % 6 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by 6 for 3D axial RoPE")
        if latent_channels <= 0:
            raise ValueError(f"latent_channels must be positive, got {latent_channels}")
        if coarse_res <= 0:
            raise ValueError(f"coarse_res must be positive, got {coarse_res}")

        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.latent_channels = latent_channels
        self.in_channels = latent_channels + 1  # 1 occupancy logit + C latent channels
        self.coarse_res = coarse_res
        self.cond_drop_prob = cond_drop_prob
        self.mlp_ratio = mlp_ratio

        # Input projection: [x_t, c_raw] concatenated along channels -> 2 * in_channels
        self.x_embed = nn.Linear(2 * self.in_channels, dim)

        # Learnable null conditioning token for unconditional CFG
        self.null_condition = nn.Parameter(torch.zeros(1, 1, self.in_channels))

        # Continuous diffusion timestep embedder
        self.t_embedder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Transformer backbone blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Final AdaLN layer + projection
        self.final_layer = FinalLayer(dim, self.in_channels)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights matching standard DiT conventions."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

        # Timestep embedder initialization
        nn.init.normal_(self.t_embedder[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder[2].weight, std=0.02)

        # Zero-out AdaLN modulation layers and final linear projection
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[1].weight)
            nn.init.zeros_(block.adaLN_modulation[1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def get_grid_coords(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        res: int | None = None,
    ) -> torch.Tensor:
        """Generate regular 3D grid coordinates of shape (res^3, 3)."""
        r = self.coarse_res if res is None else res
        axes = [torch.arange(r, device=device, dtype=dtype) for _ in range(3)]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        return grid

    @staticmethod
    def split_state(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split joint state x into (occupancy, latent) components.
        
        Args:
            state: Tensor of shape (..., C+1)
            
        Returns:
            occupancy: Tensor of shape (..., 1)
            latents: Tensor of shape (..., C)
        """
        return state[..., :1], state[..., 1:]

    @staticmethod
    def join_state(occupancy: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Concatenate occupancy and latent components into joint state tensor."""
        return torch.cat([occupancy, latent], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float | int,
        c_raw: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        cond_drop_prob: float | None = None,
        drop_condition: bool = False,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of UnifiedDiT.
        
        Args:
            x: Joint noisy state tensor x_t in R^(B x N x (C+1)) or R^(N x (C+1)).
               Channel 0 is coarse occupancy logit; channels 1..C are SC-VAE continuous latents.
            t: Continuous diffusion timestep in [0, 1]. Shape (B,) or scalar.
            c_raw: Optional raw mesh conditioning in R^(B x N x (C+1)) or R^(N x (C+1)).
                   If None or dropped, uses learnable null conditioning.
            coords: Optional 3D spatial grid coordinates in R^(B x N x 3) or R^(N x 3).
                    If None, defaults to regular 3D grid of shape (coarse_res^3, 3).
            cond_drop_prob: Optional override for condition dropout probability.
            drop_condition: If True, forces unconditional null conditioning (for CFG).
            cond_mask: Optional boolean conditioning mask (True = keep, False = mask to null condition).
                       Shape (B,), (B, 1, 1), or (B, N) / (B, N, 1).
            
        Returns:
            Predicted flow velocity v in R^(B x N x (C+1)) (or R^(N x (C+1)) if unbatched).
        """
        unbatched = (x.ndim == 2)
        if unbatched:
            x = x.unsqueeze(0)

        B, N, in_c = x.shape
        if in_c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels in x (1 occ + {self.latent_channels} latent), got {in_c}")

        # Process timestep t
        t = torch.as_tensor(t, device=x.device)
        if t.ndim == 0:
            t = t.expand(B)
        elif t.ndim == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        if t.shape != (B,):
            raise ValueError(f"Expected timestep t with shape ({B},), got {t.shape}")
        t = t.to(dtype=torch.float32)

        # Process raw mesh conditioning c_raw
        if drop_condition or c_raw is None:
            c_raw_effective = self.null_condition.expand(B, N, self.in_channels)
        else:
            if c_raw.ndim == 2:
                c_raw = c_raw.unsqueeze(0).expand(B, -1, -1)
            if c_raw.shape != (B, N, self.in_channels):
                raise ValueError(f"Expected c_raw shape ({B}, {N}, {self.in_channels}), got {c_raw.shape}")

            p_drop = self.cond_drop_prob if cond_drop_prob is None else cond_drop_prob
            if self.training and p_drop > 0.0 and cond_mask is None:
                # Per-sample Bernoulli condition dropout for CFG training
                sample_keep = torch.rand(B, 1, 1, device=x.device) >= p_drop
                c_raw_effective = torch.where(sample_keep, c_raw, self.null_condition.expand(B, N, self.in_channels))
            else:
                c_raw_effective = c_raw

            if cond_mask is not None:
                # Explicit condition mask (True = keep, False = mask to null condition)
                if cond_mask.ndim == 1:
                    cond_mask = cond_mask.view(-1, 1, 1)
                elif cond_mask.ndim == 2:
                    cond_mask = cond_mask.unsqueeze(-1)
                c_raw_effective = torch.where(cond_mask, c_raw_effective, self.null_condition.expand(B, N, self.in_channels))

        # Concatenate noisy state and raw mesh conditioning along channel dimension
        x_in = torch.cat([x, c_raw_effective], dim=-1)  # (B, N, 2 * in_channels)

        # Process spatial coordinates
        if coords is None:
            if N == self.coarse_res ** 3:
                coords = self.get_grid_coords(device=x.device, dtype=x.dtype)
            else:
                res = int(round(N ** (1.0 / 3.0)))
                if res ** 3 == N:
                    coords = self.get_grid_coords(device=x.device, dtype=x.dtype, res=res)
                else:
                    raise ValueError(f"coords must be provided when token count N={N} is not a perfect cube")

        # Linear projection
        h = self.x_embed(x_in)  # (B, N, dim)

        # Timestep embedding
        t_emb = self.t_embedder(sinusoidal_embedding(t, self.dim))  # (B, dim)

        # DiT transformer blocks
        for block in self.blocks:
            h = block(h, t_emb, coords)

        # Final AdaLN layer + projection
        v = self.final_layer(h, t_emb)  # (B, N, in_channels)

        if unbatched:
            v = v.squeeze(0)
        return v

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float | int,
        c_raw: torch.Tensor | None = None,
        cfg_scale: float = 3.0,
        coords: torch.Tensor | None = None,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Classifier-Free Guidance (CFG) and Interval Guidance.
        
        Args:
            x: Joint noisy state tensor in R^(B x N x (C+1)).
            t: Diffusion timestep in [0, 1].
            c_raw: Conditioning tensor.
            cfg_scale: Guidance weight (default 3.0).
            coords: Spatial 3D coordinates.
            cfg_interval: (t_min, t_max) interval where CFG is active. Outside
                          this interval, returns conditional prediction.
            cond_mask: Optional conditioning mask.
                          
        Returns:
            Guided velocity prediction in R^(B x N x (C+1)).
        """
        v_cond = self.forward(x, t, c_raw=c_raw, coords=coords, drop_condition=False, cond_mask=cond_mask)

        t_val = float(torch.as_tensor(t).mean().item())
        t_min, t_max = cfg_interval
        if cfg_scale == 1.0 or not (t_min <= t_val <= t_max):
            return v_cond

        v_uncond = self.forward(x, t, c_raw=None, coords=coords, drop_condition=True)
        return v_uncond + cfg_scale * (v_cond - v_uncond)


def flow_pair(
    clean: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate linear interpolation flow pair for rectified flow matching."""
    if clean.shape != noise.shape:
        raise ValueError(f"Clean and noise shapes must match: {clean.shape} vs {noise.shape}")
    t = time
    while t.ndim < clean.ndim:
        t = t.unsqueeze(-1)
    x_t = (1.0 - t) * noise + t * clean
    v_t = clean - noise
    return x_t, v_t


def flow_loss(
    model: UnifiedDiT,
    clean: torch.Tensor,
    c_raw: torch.Tensor | None = None,
    coords: torch.Tensor | None = None,
    cond_drop_prob: float = 0.1,
    cond_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute MSE velocity matching loss for UnifiedDiT."""
    unbatched = (clean.ndim == 2)
    if unbatched:
        clean = clean.unsqueeze(0)
    B = clean.shape[0]
    t = torch.rand(B, device=clean.device, dtype=clean.dtype)
    noise = torch.randn_like(clean)
    noisy, target_v = flow_pair(clean, noise, t)
    pred_v = model(noisy, t, c_raw=c_raw, coords=coords, cond_drop_prob=cond_drop_prob, cond_mask=cond_mask)
    return F.mse_loss(pred_v, target_v)


@torch.no_grad()
def sample_flow(
    model: UnifiedDiT,
    c_raw: torch.Tensor | None = None,
    coords: torch.Tensor | None = None,
    batch_size: int = 1,
    num_tokens: int | None = None,
    steps: int = 25,
    cfg_scale: float = 3.0,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    cond_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample joint state using Euler numerical integration of the learned flow."""
    model.eval()
    if num_tokens is None:
        num_tokens = model.coarse_res ** 3
    shape = (batch_size, num_tokens, model.in_channels)
    x = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    dt = 1.0 / steps
    for step in range(steps):
        t = torch.full((batch_size,), step / float(steps), device=device, dtype=dtype)
        v = model.forward_with_cfg(
            x, t, c_raw=c_raw, cfg_scale=cfg_scale, coords=coords, cfg_interval=cfg_interval, cond_mask=cond_mask
        )
        x = x + dt * v
    return x
