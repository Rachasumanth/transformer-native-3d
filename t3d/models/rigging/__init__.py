"""Rigging and Articulation Flow Module.

Provides:
- Skeleton Flow: DFS relative joint parameterization, Kinematic Tree forward kinematics decoder,
  and containment loss inside mesh bounding volume.
- Skinning Module: Surface-grounded Point-to-Bone cross-attention skinning, manifold geodesic
  fallback (heat / Dijkstra), top-4 bone truncation, and glTF JOINTS_0/WEIGHTS_0 buffer packing.
"""

from .skeleton_flow import (
    build_dfs_order,
    reorder_to_dfs,
    compute_ancestor_matrix,
    global_to_relative_joints,
    relative_to_global_joints,
    KinematicTreeDecoder,
    containment_loss,
    SkeletonFlowModel,
)

from .skinning_module import (
    point_to_bone_distances,
    compute_manifold_geodesic_weights,
    SurfacePointToBoneSkinning,
    truncate_top4_weights,
    pack_gltf_skin_buffers,
)

__all__ = [
    'build_dfs_order',
    'reorder_to_dfs',
    'compute_ancestor_matrix',
    'global_to_relative_joints',
    'relative_to_global_joints',
    'KinematicTreeDecoder',
    'containment_loss',
    'SkeletonFlowModel',
    'point_to_bone_distances',
    'compute_manifold_geodesic_weights',
    'SurfacePointToBoneSkinning',
    'truncate_top4_weights',
    'pack_gltf_skin_buffers',
]
