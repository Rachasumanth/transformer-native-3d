"""Model architectures for Transformer-Native 3D."""
from .unified_dit import (
    UnifiedDiT,
    RoPE3D,
    SelfAttention3D,
    DiTBlock,
    flow_pair,
    flow_loss,
    sample_flow,
)
from .sc_vae import (
    SparseCompressionVAE,
    HermiteNormalHead,
    HermiteHead,
    hermite_normal_loss,
    reconstruction_loss,
)
from .pbr_flow import (
    SurfaceGroundedPBRFlow,
    PBRFlowNetwork,
    PBRFlow,
    PBRFlowOutput,
    debake_luminance,
    pad_texture_dilation,
    dilate_uv_seams,
    format_pbr_textures,
    pbr_flow_pair,
    pbr_flow_loss,
    sample_pbr_flow,
)
from . import rigging

__all__ = [
    "UnifiedDiT",
    "RoPE3D",
    "SelfAttention3D",
    "DiTBlock",
    "flow_pair",
    "flow_loss",
    "sample_flow",
    "SparseCompressionVAE",
    "HermiteNormalHead",
    "HermiteHead",
    "hermite_normal_loss",
    "reconstruction_loss",
    "SurfaceGroundedPBRFlow",
    "PBRFlowNetwork",
    "PBRFlow",
    "PBRFlowOutput",
    "debake_luminance",
    "pad_texture_dilation",
    "dilate_uv_seams",
    "format_pbr_textures",
    "pbr_flow_pair",
    "pbr_flow_loss",
    "sample_pbr_flow",
    "rigging",
]
