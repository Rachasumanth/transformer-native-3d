"""Multimodal Front-End Adapter for Transformer-Native 3D.

Maps multimodal conditioning tokens (text embeddings / CLIP tokens and/or image
embeddings / DINOv2 tokens) to coarse occupancy and conditioning latents:
    c_raw in R^(B x N x (C + 1))
where N = coarse_res^3 is the number of coarse voxel tokens, and C is the SC-VAE
continuous latent dimensionality (typically 16).
"""
import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CoordinatePositionalEmbedding(nn.Module):
    """3D continuous sinusoidal positional embedding with trainable projection."""

    def __init__(self, embed_dim: int, num_freqs: int = 16, max_period: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_freqs = num_freqs
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(num_freqs, dtype=torch.float32) / max(1, num_freqs - 1)
        )
        self.register_buffer("freqs", freqs)
        in_dim = 3 * 2 * num_freqs
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode 3D coordinates (..., 3) into positional embeddings (..., embed_dim)."""
        # coords: (..., 3)
        args = coords.unsqueeze(-1) * self.freqs  # (..., 3, num_freqs)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (..., 3, 2 * num_freqs)
        emb = emb.reshape(*coords.shape[:-1], -1)  # (..., 3 * 2 * num_freqs)
        return self.proj(emb)


class AdapterTransformerBlock(nn.Module):
    """Transformer block with cross-attention to condition tokens, self-attention, and MLP."""

    def __init__(self, embed_dim: int, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm_cross = nn.LayerNorm(embed_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm_self = nn.LayerNorm(embed_dim)

        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )
        self.norm_mlp = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Cross-attention to condition tokens
        cross_out, _ = self.cross_attn(query=x, key=cond, value=cond)
        x = self.norm_cross(x + cross_out)

        # Self-attention among spatial voxel tokens
        self_out, _ = self.self_attn(query=x, key=x, value=x)
        x = self.norm_self(x + self_out)

        # Feedforward MLP
        x = self.norm_mlp(x + self.mlp(x))
        return x


class MultimodalFrontEndAdapter(nn.Module):
    """Multimodal Front-End Adapter projecting text and/or image inputs to coarse latents c_raw.

    Args:
        text_dim: Feature dimension of input text tokens (default: 512).
        image_dim: Feature dimension of input image tokens (default: 768).
        embed_dim: Intermediate transformer hidden dimension (default: 256).
        latent_channels: SC-VAE continuous latent dimensionality C (default: 16).
        coarse_res: Spatial resolution of coarse grid (default: 4 -> N = 64).
        num_layers: Number of transformer blocks (default: 4).
        num_heads: Number of attention heads (default: 4).
    """

    def __init__(
        self,
        text_dim: int = 512,
        image_dim: int = 768,
        embed_dim: int = 256,
        latent_channels: int = 16,
        coarse_res: int = 4,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        if latent_channels <= 0:
            raise ValueError(f"latent_channels must be positive, got {latent_channels}")
        if coarse_res <= 0:
            raise ValueError(f"coarse_res must be positive, got {coarse_res}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.text_dim = text_dim
        self.image_dim = image_dim
        self.embed_dim = embed_dim
        self.latent_channels = latent_channels
        self.coarse_res = coarse_res
        self.num_tokens = coarse_res ** 3
        self.num_layers = num_layers
        self.num_heads = num_heads

        # 3D Canonical grid coordinates in [-0.45, 0.45]^3 (10% empty margin within [-0.5, 0.5]^3)
        coords = self._build_canonical_grid(coarse_res)
        self.register_buffer("coords", coords)

        # 3D coordinate positional embedding
        self.pos_embed = CoordinatePositionalEmbedding(embed_dim)

        # Learnable voxel queries
        self.voxel_queries = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim) * 0.02)

        # Modality projectors
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.image_projector = nn.Sequential(
            nn.Linear(image_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Modality type indicator embeddings
        self.text_type_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.image_type_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Unconditional learnable null context token
        self.null_context = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            AdapterTransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        # Output prediction heads
        self.occ_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.latent_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, latent_channels),
        )

    @staticmethod
    def _build_canonical_grid(res: int) -> torch.Tensor:
        """Construct normalized 3D grid coordinates strictly in [-0.45, 0.45]^3."""
        coords_1d = torch.linspace(-0.45, 0.45, res)
        grid = torch.stack(torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij"), dim=-1)
        return grid.reshape(-1, 3)

    def forward(
        self,
        text_tokens: Optional[torch.Tensor] = None,
        image_tokens: Optional[torch.Tensor] = None,
        text_pooled: Optional[torch.Tensor] = None,
        image_pooled: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        coords: Optional[torch.Tensor] = None,
        num_tokens: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass generating coarse occupancy logits, continuous latents, and c_raw."""
        # Determine batch size and validate modality inputs
        inputs = [
            (text_tokens, "text_tokens", 3, self.text_dim),
            (image_tokens, "image_tokens", 3, self.image_dim),
            (text_pooled, "text_pooled", 2, self.text_dim),
            (image_pooled, "image_pooled", 2, self.image_dim),
        ]

        active_batch_sizes = []
        for tensor, name, expected_ndim, expected_dim in inputs:
            if tensor is not None:
                if tensor.shape[-1] != expected_dim:
                    raise ValueError(f"Expected {name} feature dimension {expected_dim}, got {tensor.shape[-1]}")
                # Unbatched (1D or 2D) -> batched
                if expected_ndim == 3:
                    if tensor.ndim == 2:
                        b = 1
                    else:
                        b = tensor.shape[0]
                else:
                    if tensor.ndim == 1:
                        b = 1
                    else:
                        b = tensor.shape[0]
                active_batch_sizes.append(b)

        if len(set(active_batch_sizes)) > 1:
            raise ValueError(f"Mismatched batch sizes across multimodal inputs: {active_batch_sizes}")

        if active_batch_sizes:
            B = active_batch_sizes[0]
        else:
            B = batch_size if batch_size is not None else 1

        device = next(self.parameters()).device

        # Process condition tokens
        cond_tokens_list = []

        if text_tokens is not None:
            t = text_tokens
            if t.ndim == 2:
                t = t.unsqueeze(0)
            if t.device != device:
                t = t.to(device)
            h_text = self.text_projector(t) + self.text_type_embed
            cond_tokens_list.append(h_text)

        if text_pooled is not None:
            t = text_pooled
            if t.ndim == 1:
                t = t.unsqueeze(0).unsqueeze(1)
            elif t.ndim == 2:
                t = t.unsqueeze(1)
            if t.device != device:
                t = t.to(device)
            h_text_p = self.text_projector(t) + self.text_type_embed
            cond_tokens_list.append(h_text_p)

        if image_tokens is not None:
            img = image_tokens
            if img.ndim == 2:
                img = img.unsqueeze(0)
            if img.device != device:
                img = img.to(device)
            h_img = self.image_projector(img) + self.image_type_embed
            cond_tokens_list.append(h_img)

        if image_pooled is not None:
            img = image_pooled
            if img.ndim == 1:
                img = img.unsqueeze(0).unsqueeze(1)
            elif img.ndim == 2:
                img = img.unsqueeze(1)
            if img.device != device:
                img = img.to(device)
            h_img_p = self.image_projector(img) + self.image_type_embed
            cond_tokens_list.append(h_img_p)

        if cond_tokens_list:
            cond = torch.cat(cond_tokens_list, dim=1)
        else:
            # Unconditional mode with learnable null context token
            cond = self.null_context.expand(B, 1, -1)

        # Prepare voxel spatial query tokens
        c = self.coords if coords is None else coords
        if c.device != device:
            c = c.to(device)
        pos = self.pos_embed(c)
        if pos.ndim == 2:
            pos = pos.unsqueeze(0)

        x = (self.voxel_queries + pos).expand(B, -1, -1)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, cond)

        # Output predictions
        occ_logits = self.occ_head(x).squeeze(-1)  # (B, N)
        latents = self.latent_head(x)  # (B, N, C)
        c_raw = torch.cat([occ_logits.unsqueeze(-1), latents], dim=-1)  # (B, N, C+1)

        return {
            "occ_logits": occ_logits,
            "latents": latents,
            "c_raw": c_raw,
        }


class LossWithComponents(torch.Tensor):
    """Subclass of torch.Tensor holding component loss attributes."""
    loss_occ: float
    loss_lat: float


def adapter_loss(
    out_or_pred: Union[Dict[str, torch.Tensor], torch.Tensor],
    target_occ: torch.Tensor,
    target_lat: Optional[torch.Tensor] = None,
    occ_weight: float = 1.0,
    lat_weight: float = 1.0,
    pos_weight: Optional[float] = None,
    return_dict: bool = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute combined loss for front-end adapter predictions.

    Args:
        out_or_pred: Output dict from adapter, containing 'occ_logits' and 'latents'.
        target_occ: Target binary occupancy tensor in R^(B x N) or (N,).
        target_lat: Optional target continuous latents in R^(B x N x C).
        occ_weight: Weight for binary cross entropy on occupancy.
        lat_weight: Weight for MSE loss on continuous latents.
        pos_weight: Optional positive class weighting for BCE.
        return_dict: Whether to return a dictionary of individual loss components.

    Returns:
        Scalar loss tensor (with loss_occ and loss_lat attributes) or dictionary.
    """
    if isinstance(out_or_pred, dict):
        pred_occ = out_or_pred["occ_logits"]
        pred_lat = out_or_pred.get("latents")
    else:
        pred_occ = out_or_pred[..., 0]
        pred_lat = out_or_pred[..., 1:]

    target_occ_f = target_occ.float()
    if pos_weight is not None:
        pw = torch.tensor(pos_weight, device=pred_occ.device, dtype=pred_occ.dtype)
        loss_occ = F.binary_cross_entropy_with_logits(pred_occ, target_occ_f, pos_weight=pw)
    else:
        loss_occ = F.binary_cross_entropy_with_logits(pred_occ, target_occ_f)

    if pred_lat is not None and target_lat is not None:
        # Latent loss is masked to active target cells only
        active_mask = (target_occ_f > 0.0).unsqueeze(-1)
        diff = (pred_lat - target_lat) * active_mask
        if active_mask.sum() > 0:
            loss_lat = (diff ** 2).sum() / (active_mask.sum() * target_lat.shape[-1]).clamp_min(1.0)
        else:
            loss_lat = torch.tensor(0.0, device=pred_occ.device, dtype=pred_occ.dtype, requires_grad=True)
    else:
        loss_lat = torch.tensor(0.0, device=pred_occ.device, dtype=pred_occ.dtype)

    total_loss = occ_weight * loss_occ + lat_weight * loss_lat

    if return_dict:
        return {
            "loss": total_loss,
            "loss_occ": loss_occ,
            "loss_lat": loss_lat,
        }

    # Attach component attributes to tensor
    total_loss.loss_occ = float(loss_occ.item())
    total_loss.loss_lat = float(loss_lat.item())
    return total_loss


def compute_occupancy_metrics(
    pred_logits: Union[torch.Tensor, np.ndarray],
    gt_occupancy: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.0,
) -> Dict[str, float]:
    """Compute binary classification metrics: recall, precision, f1, accuracy, iou."""
    if isinstance(pred_logits, torch.Tensor):
        p = (pred_logits.detach().cpu().numpy() > threshold).astype(bool).ravel()
    else:
        p = (np.asarray(pred_logits) > threshold).astype(bool).ravel()

    if isinstance(gt_occupancy, torch.Tensor):
        g = (gt_occupancy.detach().cpu().numpy() > 0.0).astype(bool).ravel()
    else:
        g = (np.asarray(gt_occupancy) > 0.0).astype(bool).ravel()

    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    tn = np.logical_and(~p, ~g).sum()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    if precision + recall > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0
    acc = float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 1.0

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "iou": iou,
        "accuracy": acc,
    }
