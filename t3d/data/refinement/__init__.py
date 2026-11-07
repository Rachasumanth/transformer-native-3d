"""Refinement pipeline data processing and pre-encoding utilities."""
from .canonicalize_geom import (
    canonicalize_geometry,
    normalize_coordinates,
    compute_udf_and_flood_fill,
    extract_manifold_isosurface,
    decimate_crease_preserving,
)
from .corrupt_mesh import (
    synthesize_corrupted_mesh,
)
from .encode_latents import (
    encode_paired_asset,
    encode_micro_batch,
    extract_voxel_hermite_normals,
    load_mesh_or_asset,
)
from .canonicalize_rig import (
    canonicalize_rig,
    parse_partnet_urdf,
    prune_dead_bones,
    dfs_order_skeleton,
    barycentric_weight_transfer,
    laplacian_smooth_weights,
    truncate_top4_weights,
    pack_gltf_skin_accessors,
    RigValidationError,
)
from .canonicalize_pbr import (
    unwrap_single_atlas,
    debake_luminance,
    pad_texture,
    bake_single_atlas_pbr,
    compute_tangent_frames,
    rasterize_uv_atlas,
)

__all__ = [
    'canonicalize_geometry',
    'normalize_coordinates',
    'compute_udf_and_flood_fill',
    'extract_manifold_isosurface',
    'decimate_crease_preserving',
    'encode_paired_asset',
    'encode_micro_batch',
    'extract_voxel_hermite_normals',
    'load_mesh_or_asset',
    'synthesize_corrupted_mesh',
    'canonicalize_rig',
    'parse_partnet_urdf',
    'prune_dead_bones',
    'dfs_order_skeleton',
    'barycentric_weight_transfer',
    'laplacian_smooth_weights',
    'truncate_top4_weights',
    'pack_gltf_skin_accessors',
    'RigValidationError',
    'unwrap_single_atlas',
    'debake_luminance',
    'pad_texture',
    'bake_single_atlas_pbr',
    'compute_tangent_frames',
    'rasterize_uv_atlas',
]

