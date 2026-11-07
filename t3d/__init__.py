"""Transformer-native 3D format implementation."""
from .asset_validator import validate_production_asset
from .decimation import bake_tangent_normal_map, decimate_mesh
from .models.unified_dit import UnifiedDiT
from .pipeline import T3DOutput, T3DPipeline

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "UnifiedDiT",
    "T3DPipeline",
    "T3DOutput",
    "decimate_mesh",
    "bake_tangent_normal_map",
    "validate_production_asset",
]
