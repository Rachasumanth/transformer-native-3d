"""Surface-Grounded PBR Flow Network for UV Atlas Material Synthesis.

Predicts single UV atlas PBR texture maps (Basecolor, Metallic-Roughness, Normal map)
conditioned on 3D geometry and latent representations using continuous rectified flow.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple, Union
import numpy as np
import scipy.ndimage as ndimage
import torch
from torch import nn
from torch.nn import functional as F

# ITU-R BT.709 relative luminance weights
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def debake_luminance(
    basecolor: Union[torch.Tensor, np.ndarray],
    min_lum: float = 0.04,
    max_lum: float = 0.96,
) -> Union[torch.Tensor, np.ndarray]:
    """Clamp linear RGB basecolor luminance to [min_lum, max_lum] to strip baked shadows and specular blowouts.

    Preserves chromaticity and hue by scaling RGB channels proportionally to luminance.

    Args:
        basecolor: (B, 3, H, W) or (..., 3) array/tensor of RGB float values in [0, 1].
        min_lum: Minimum luminance threshold (default: 0.04).
        max_lum: Maximum luminance threshold (default: 0.96).

    Returns:
        Array/tensor of identical shape and type with luminance clamped to [min_lum, max_lum].
    """
    if min_lum < 0.0 or max_lum > 1.0 or min_lum > max_lum:
        raise ValueError(f"Invalid luminance bounds [{min_lum}, {max_lum}]")

    is_tensor = isinstance(basecolor, torch.Tensor)
    if is_tensor:
        rgb = basecolor.clone()
        if rgb.dim() >= 3 and rgb.shape[1] == 3 and (rgb.dim() == 4 or rgb.dim() == 3 and rgb.shape[-1] != 3):
            # Channels-first: (B, 3, H, W) or (3, H, W)
            weights = rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1) if rgb.dim() == 4 else rgb.new_tensor([0.2126, 0.7152, 0.0722]).view(3, 1, 1)
            ch_dim = 1 if rgb.dim() == 4 else 0
            lum = (rgb * weights).sum(dim=ch_dim, keepdim=True)
            scale = torch.where(
                lum > 1e-6,
                torch.clamp(lum, min_lum, max_lum) / torch.clamp_min(lum, 1e-6),
                torch.ones_like(lum),
            )
            adjusted = rgb * scale
            zero_mask = lum <= 1e-6
            adjusted = torch.where(zero_mask, rgb.new_full(adjusted.shape, min_lum), adjusted)
            return torch.clamp(adjusted, 0.0, 1.0)
        else:
            # Channels-last: (..., 3)
            if rgb.shape[-1] < 3:
                raise ValueError(f"Expected last dimension >= 3, got {rgb.shape}")
            weights = rgb.new_tensor([0.2126, 0.7152, 0.0722])
            lum = (rgb[..., :3] * weights).sum(dim=-1, keepdim=True)
            scale = torch.where(
                lum > 1e-6,
                torch.clamp(lum, min_lum, max_lum) / torch.clamp_min(lum, 1e-6),
                torch.ones_like(lum),
            )
            adjusted = rgb[..., :3] * scale
            zero_mask = lum <= 1e-6
            adjusted = torch.where(zero_mask, rgb.new_full(adjusted.shape, min_lum), adjusted)
            return torch.clamp(adjusted, 0.0, 1.0)
    else:
        arr = np.asarray(basecolor, dtype=np.float32).copy()
        if arr.ndim >= 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            # (3, H, W)
            weights = LUMINANCE_WEIGHTS[:, None, None]
            lum = np.sum(arr[:3] * weights, axis=0, keepdims=True)
            scale = np.where(
                lum > 1e-6,
                np.clip(lum, min_lum, max_lum) / np.maximum(lum, 1e-6),
                1.0,
            )
            adjusted = arr[:3] * scale
            zero_mask = lum <= 1e-6
            adjusted = np.where(zero_mask, min_lum, adjusted)
            return np.clip(adjusted, 0.0, 1.0)
        else:
            # (..., 3)
            if arr.shape[-1] < 3:
                raise ValueError(f"Expected last dimension >= 3, got {arr.shape}")
            weights = LUMINANCE_WEIGHTS
            lum = np.sum(arr[..., :3] * weights, axis=-1, keepdims=True)
            scale = np.where(
                lum > 1e-6,
                np.clip(lum, min_lum, max_lum) / np.maximum(lum, 1e-6),
                1.0,
            )
            adjusted = arr[..., :3] * scale
            zero_mask = lum <= 1e-6
            adjusted = np.where(zero_mask, min_lum, adjusted)
            return np.clip(adjusted, 0.0, 1.0)


def pad_texture_dilation(
    image: Union[torch.Tensor, np.ndarray],
    valid_mask: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """Pad uncovered atlas texels using exact nearest-neighbor Euclidean distance transform.

    Preserves all covered texel values exactly while extending chart borders
    into atlas background gaps to eliminate bilinear mipmap bleeding.

    Args:
        image: (B, C, H, W), (C, H, W), (H, W, C), or (H, W) array/tensor.
        valid_mask: (B, 1, H, W), (B, H, W), or (H, W) boolean mask of covered texels.

    Returns:
        Padded texture array/tensor of identical shape and type.
    """
    is_tensor = isinstance(image, torch.Tensor)
    if is_tensor:
        if image.dim() == 4:
            # (B, C, H, W)
            b, c, h, w = image.shape
            mask = valid_mask.squeeze(1) if valid_mask.dim() == 4 else valid_mask
            if mask.shape != (b, h, w):
                raise ValueError(f"Mask shape {mask.shape} does not match image spatial shape {(b, h, w)}")
            padded_list = []
            for i in range(b):
                img_i = image[i]
                mask_i = mask[i].detach().cpu().numpy().astype(bool)
                if mask_i.all() or not mask_i.any():
                    padded_list.append(img_i)
                    continue
                indices = ndimage.distance_transform_edt(~mask_i, return_distances=False, return_indices=True)
                row_idx = torch.as_tensor(indices[0], dtype=torch.long, device=image.device)
                col_idx = torch.as_tensor(indices[1], dtype=torch.long, device=image.device)
                padded_list.append(img_i[:, row_idx, col_idx])
            return torch.stack(padded_list, dim=0)
        elif image.dim() == 3:
            # (C, H, W)
            c, h, w = image.shape
            mask = valid_mask.squeeze(0) if valid_mask.dim() == 3 else valid_mask
            if mask.shape != (h, w):
                raise ValueError(f"Mask shape {mask.shape} does not match image spatial shape {(h, w)}")
            mask_np = mask.detach().cpu().numpy().astype(bool)
            if mask_np.all() or not mask_np.any():
                return image.clone()
            indices = ndimage.distance_transform_edt(~mask_np, return_distances=False, return_indices=True)
            row_idx = torch.as_tensor(indices[0], dtype=torch.long, device=image.device)
            col_idx = torch.as_tensor(indices[1], dtype=torch.long, device=image.device)
            return image[:, row_idx, col_idx]
        else:
            raise ValueError(f"Unsupported tensor shape {image.shape}")
    else:
        arr = np.asarray(image)
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if arr.ndim == 3 and arr.shape[:2] == mask.shape:
            # (H, W, C)
            if mask.all() or not mask.any():
                return arr.copy()
            indices = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
            return arr[indices[0], indices[1]]
        elif arr.ndim == 3 and arr.shape[1:] == mask.shape:
            # (C, H, W)
            if mask.all() or not mask.any():
                return arr.copy()
            indices = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
            return arr[:, indices[0], indices[1]]
        elif arr.ndim == 2 and arr.shape == mask.shape:
            # (H, W)
            if mask.all() or not mask.any():
                return arr.copy()
            indices = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
            return arr[indices[0], indices[1]]
        else:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {arr.shape}")


dilate_uv_seams = pad_texture_dilation


@dataclass
class PBRFlowOutput:
    """Output container for Surface-Grounded PBR Flow network."""
    velocity: torch.Tensor
    basecolor: torch.Tensor
    metallic_roughness: torch.Tensor
    normal_map: torch.Tensor
    combined: Optional[torch.Tensor] = None

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return (self.velocity, self.basecolor, self.metallic_roughness, self.normal_map)[key]

    def __len__(self):
        return 4


def format_pbr_textures(
    raw_pbr: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    pad_seams: bool = False,
) -> PBRFlowOutput:
    """Format an 8-channel PBR tensor into individual validated texture maps.

    Channels:
    - 0..3: Basecolor linear RGB in [0, 1] debaked to [0.04, 0.96]
    - 3..5: Metallic-Roughness (Roughness G, Metallic B) in [0, 1]
    - 5..8: Tangent-space Normal map RGB in [0, 1] (neutral [0.5, 0.5, 1.0])

    Args:
        raw_pbr: (B, 8, H, W) float tensor.
        valid_mask: Optional (B, 1, H, W) or (B, H, W) boolean mask of valid UV texels.
        pad_seams: Whether to apply nearest-neighbor dilation padding.

    Returns:
        PBRFlowOutput container with formatted maps.
    """
    if raw_pbr.shape[1] != 8:
        raise ValueError(f"Expected 8 channels for raw_pbr, got {raw_pbr.shape[1]}")

    # Basecolor: clamped to [0, 1], then debaked to [0.04, 0.96]
    bc_raw = torch.clamp(raw_pbr[:, 0:3], 0.0, 1.0)
    basecolor = debake_luminance(bc_raw, min_lum=0.04, max_lum=0.96)

    # Metallic-Roughness: 2-channel float32 in [0, 1] (channel 0: Roughness G, channel 1: Metallic B)
    metallic_roughness = torch.clamp(raw_pbr[:, 3:5], 0.0, 1.0)

    # Normal map: tangent-space RGB in [0, 1], neutral [0.5, 0.5, 1.0]
    n_raw = raw_pbr[:, 5:8]
    n_len = torch.linalg.norm(n_raw, ord=2, dim=1, keepdim=True)
    # Neutral normal is pointing along +Z in tangent space: [0, 0, 1]
    neutral_n = torch.tensor([0.0, 0.0, 1.0], device=raw_pbr.device, dtype=raw_pbr.dtype).view(1, 3, 1, 1)
    n_unit = torch.where(n_len > 1e-6, n_raw / (n_len + 1e-6), neutral_n)
    # Tangent vector in [-1, 1] -> RGB in [0, 1]: 0.5 * n + 0.5
    normal_rgb = 0.5 * n_unit + 0.5
    normal_map = torch.clamp(normal_rgb, 0.0, 1.0)

    if pad_seams and valid_mask is not None:
        basecolor = pad_texture_dilation(basecolor, valid_mask)
        metallic_roughness = pad_texture_dilation(metallic_roughness, valid_mask)
        normal_map = pad_texture_dilation(normal_map, valid_mask)

    combined = torch.cat([basecolor, metallic_roughness, normal_map], dim=1)
    return PBRFlowOutput(
        velocity=raw_pbr,
        basecolor=basecolor,
        metallic_roughness=metallic_roughness,
        normal_map=normal_map,
        combined=combined,
    )


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for continuous flow timestep."""
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(timesteps.dtype)


class AdaGNResBlock2D(nn.Module):
    """Residual convolutional block with Adaptive Group Normalization (AdaGN / FiLM)."""
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, groups: int = 8):
        super().__init__()
        g1 = min(groups, in_channels)
        g2 = min(groups, out_channels)
        self.norm1 = nn.GroupNorm(g1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(g2, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.mod = nn.Linear(cond_dim, 2 * in_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.mod(self.act(cond)).chunk(2, dim=-1)
        h = self.norm1(x) * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        h = self.conv1(self.act(h))
        h = self.conv2(self.act(self.norm2(h)))
        return self.skip(x) + h


class SurfaceGroundedPBRFlow(nn.Module):
    """Surface-Grounded Rectified Flow Network for Single UV Atlas PBR Texture Synthesis.

    Predicts PBR texture flow velocities and decoded texture maps (Basecolor,
    Metallic-Roughness, Normal map) conditioned on 3D surface geometry and latent
    representations.
    """
    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        geom_channels: int = 3,
        latent_channels: int = 16,
        time_dim: int = 64,
        cond_dim: int = 128,
        channels: Tuple[int, ...] = (32, 64, 128),
    ):
        super().__init__()
        if len(channels) != 3 or min(channels) <= 0:
            raise ValueError("channels must contain three positive widths")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.geom_channels = geom_channels
        self.latent_channels = latent_channels
        self.cond_dim = cond_dim

        # Timestep conditioning branch
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Latent conditioning branch (from SC-VAE bottleneck or prompt embeddings)
        self.latent_mlp = nn.Sequential(
            nn.Linear(latent_channels, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        ) if latent_channels > 0 else None

        # 3D Geometry conditioning branch
        self.geom_spatial = nn.Conv2d(geom_channels, channels[0], kernel_size=3, padding=1) if geom_channels > 0 else None
        self.geom_global = nn.Linear(geom_channels, cond_dim) if geom_channels > 0 else None

        # Initial texture projection
        self.in_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        # Downsampling stages
        self.down0 = AdaGNResBlock2D(channels[0], channels[0], cond_dim)
        self.downsample0 = nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1)

        self.down1 = AdaGNResBlock2D(channels[1], channels[1], cond_dim)
        self.downsample1 = nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=2, padding=1)

        # Mid bottleneck stage
        self.mid = AdaGNResBlock2D(channels[2], channels[2], cond_dim)

        # Upsampling stages
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels[2], channels[1], kernel_size=3, padding=1),
        )
        self.up1 = AdaGNResBlock2D(channels[1] * 2, channels[1], cond_dim)

        self.upsample0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels[1], channels[0], kernel_size=3, padding=1),
        )
        self.up0 = AdaGNResBlock2D(channels[0] * 2, channels[0], cond_dim)

        # Output projection (zero-initialized for flow stability)
        self.out_norm = nn.GroupNorm(min(8, channels[0]), channels[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _build_context(
        self,
        time: Union[torch.Tensor, float],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        geom_cond: Optional[torch.Tensor] = None,
        latent_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Construct the combined (B, cond_dim) conditioning context vector."""
        t_tensor = torch.as_tensor(time, device=device, dtype=dtype)
        if t_tensor.dim() == 0:
            t_tensor = t_tensor.expand(batch_size)
        elif t_tensor.dim() == 1 and len(t_tensor) == 1:
            t_tensor = t_tensor.expand(batch_size)

        cond = self.time_mlp(self.time_embed(t_tensor))

        if self.latent_mlp is not None and latent_cond is not None:
            l_cond = latent_cond
            if l_cond.dim() == 3:
                # Sequence of tokens: average pool
                l_cond = l_cond.mean(dim=1)
            elif l_cond.dim() == 4:
                # Spatial latents: global average pool
                l_cond = l_cond.mean(dim=[-2, -1])
            cond = cond + self.latent_mlp(l_cond.to(dtype))

        if self.geom_global is not None and geom_cond is not None and geom_cond.dim() == 2:
            cond = cond + self.geom_global(geom_cond.to(dtype))

        return cond

    def forward(
        self,
        x: torch.Tensor,
        time: Union[torch.Tensor, float],
        geom_cond: Optional[torch.Tensor] = None,
        latent_cond: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        pad_seams: bool = False,
    ) -> Union[PBRFlowOutput, torch.Tensor]:
        """Forward pass predicting PBR flow velocity and decoded material maps.

        Args:
            x: Current/noisy texture atlas state, shape (B, in_channels, H, W).
            time: Scalar or (B,) float timestep in [0, 1].
            geom_cond: 3D geometry condition, either spatial (B, C_geom, H, W) or global (B, C_geom).
            latent_cond: Latent condition (B, C_latent) or (B, L, C_latent).
            valid_mask: Optional boolean mask (B, 1, H, W) or (B, H, W) of active UV texels.
            return_dict: If True, returns a PBRFlowOutput container with decoded texture maps.
            pad_seams: If True and valid_mask is provided, dilates textures across UV seams.

        Returns:
            PBRFlowOutput container or velocity tensor (B, out_channels, H, W).
        """
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected input shape (B, {self.in_channels}, H, W), got {x.shape}")

        b, c, h, w = x.shape
        cond = self._build_context(time, b, x.device, x.dtype, geom_cond=geom_cond, latent_cond=latent_cond)

        # Initial features
        feat = self.in_conv(x)
        if self.geom_spatial is not None and geom_cond is not None and geom_cond.dim() == 4:
            feat = feat + self.geom_spatial(geom_cond.to(x.dtype))

        # Downsample path with skip connections
        skip0 = self.down0(feat, cond)
        d0 = self.downsample0(skip0)

        skip1 = self.down1(d0, cond)
        d1 = self.downsample1(skip1)

        # Mid stage
        m = self.mid(d1, cond)

        # Upsample path
        u1 = self.upsample1(m)
        if u1.shape[-2:] != skip1.shape[-2:]:
            u1 = F.interpolate(u1, size=skip1.shape[-2:], mode="nearest")
        u1 = self.up1(torch.cat([u1, skip1], dim=1), cond)

        u0 = self.upsample0(u1)
        if u0.shape[-2:] != skip0.shape[-2:]:
            u0 = F.interpolate(u0, size=skip0.shape[-2:], mode="nearest")
        u0 = self.up0(torch.cat([u0, skip0], dim=1), cond)

        # Final velocity prediction
        velocity = self.out_conv(self.out_act(self.out_norm(u0)))

        if not return_dict:
            return velocity

        # Rectified flow clean state integration: x_1 = x_t + (1 - t) * v_t
        t_val = torch.as_tensor(time, device=x.device, dtype=x.dtype)
        if t_val.dim() == 0:
            t_val = t_val.view(1, 1, 1, 1)
        elif t_val.dim() == 1:
            t_val = t_val.view(-1, 1, 1, 1)

        clean_est = x + (1.0 - t_val) * velocity

        out = format_pbr_textures(clean_est, valid_mask=valid_mask, pad_seams=pad_seams)
        out.velocity = velocity
        return out

    def predict_textures(
        self,
        x: torch.Tensor,
        time: Union[torch.Tensor, float] = 1.0,
        geom_cond: Optional[torch.Tensor] = None,
        latent_cond: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        pad_seams: bool = True,
    ) -> PBRFlowOutput:
        """Convenience method to predict clean PBR textures and apply UV seam dilation."""
        return self.forward(
            x,
            time=time,
            geom_cond=geom_cond,
            latent_cond=latent_cond,
            valid_mask=valid_mask,
            return_dict=True,
            pad_seams=pad_seams,
        )


PBRFlowNetwork = SurfaceGroundedPBRFlow
PBRFlow = SurfaceGroundedPBRFlow


def pbr_flow_pair(
    clean: torch.Tensor,
    noise: torch.Tensor,
    time: Union[torch.Tensor, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct straight-line rectified flow interpolation and target velocity.

    Args:
        clean: Clean target texture tensor (B, C, H, W).
        noise: Standard Gaussian noise tensor (B, C, H, W).
        time: Scalar or (B,) timestep in [0, 1].

    Returns:
        (x_t, v_target): Interpolated state and target velocity.
    """
    if clean.shape != noise.shape:
        raise ValueError(f"Clean {clean.shape} and noise {noise.shape} shapes must match")
    t = torch.as_tensor(time, device=clean.device, dtype=clean.dtype)
    if t.dim() == 0:
        t = t.view(1, 1, 1, 1)
    elif t.dim() == 1:
        t = t.view(-1, 1, 1, 1)
    x_t = (1.0 - t) * noise + t * clean
    v_target = clean - noise
    return x_t, v_target


def pbr_flow_loss(
    model: nn.Module,
    clean_textures: torch.Tensor,
    geom_cond: Optional[torch.Tensor] = None,
    latent_cond: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    time: Optional[Union[torch.Tensor, float]] = None,
) -> torch.Tensor:
    """Compute rectified flow velocity matching MSE loss.

    Args:
        model: SurfaceGroundedPBRFlow instance.
        clean_textures: Ground-truth PBR texture atlas (B, 8, H, W).
        geom_cond: Optional 3D geometry conditioning.
        latent_cond: Optional latent conditioning.
        valid_mask: Optional (B, 1, H, W) or (B, H, W) mask of covered texels.
        time: Optional timestep; if None, sampled uniformly in [0, 1].

    Returns:
        Scalar MSE loss.
    """
    b = len(clean_textures)
    device = clean_textures.device
    dtype = clean_textures.dtype

    if time is None:
        t = torch.rand(b, device=device, dtype=dtype)
    else:
        t = torch.as_tensor(time, device=device, dtype=dtype)

    noise = torch.randn_like(clean_textures)
    x_t, v_target = pbr_flow_pair(clean_textures, noise, t)

    v_pred = model(x_t, t, geom_cond=geom_cond, latent_cond=latent_cond, return_dict=False)

    if valid_mask is not None:
        mask = valid_mask
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.expand(-1, clean_textures.shape[1], -1, -1).bool()
        if mask.sum() > 0:
            return F.mse_loss(v_pred[mask], v_target[mask])
    return F.mse_loss(v_pred, v_target)


@torch.no_grad()
def sample_pbr_flow(
    model: nn.Module,
    shape: Tuple[int, int, int, int],
    geom_cond: Optional[torch.Tensor] = None,
    latent_cond: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    steps: int = 20,
    pad_seams: bool = True,
    device: str = "cpu",
) -> PBRFlowOutput:
    """Sample PBR textures from noise via Euler integration of the learned velocity field."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = i / steps
        v = model(x, t, geom_cond=geom_cond, latent_cond=latent_cond, return_dict=False)
        x = x + dt * v

    return format_pbr_textures(x, valid_mask=valid_mask, pad_seams=pad_seams)
