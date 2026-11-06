"""Optional upstream CPU O-Voxel bridge with explicit dual-vertex units.

Reviewed against TRELLIS.2 commit 75fbf0183001ed9876c8dbb35de6b68552ee08bd.
The upstream package still requires a CUDA-enabled build; computation here is CPU.
"""
import numpy as np
from .schema import VoxelAsset
from .sampling import _mesh

UPSTREAM_COMMIT='75fbf0183001ed9876c8dbb35de6b68552ee08bd'


def from_native_grid(coords,positions,flags,*,grid_size,aabb,diagnostics=None):
    """Adapt AABB-relative physical native positions into bounded cell offsets."""
    coords=np.asarray(coords)
    positions=np.asarray(positions,dtype=np.float64)
    flags=np.asarray(flags)
    metadata=VoxelAsset(np.empty((0,3),dtype=int),np.empty((0,3)),np.empty((0,3),bool),grid_size,aabb)
    if positions.shape!=coords.shape or positions.ndim!=2 or positions.shape[1]!=3 or not np.isfinite(positions).all():
        raise ValueError('native dual positions must be finite (L,3)')
    # The reviewed wrapper computes these quantities as float32, not float64.
    bounds=np.asarray(aabb,dtype=np.float32)
    voxel_size=(bounds[1]-bounds[0])/np.asarray(grid_size,dtype=np.float32)
    local=positions/voxel_size.astype(np.float64)-coords.astype(np.float64)
    error=np.maximum(np.maximum(-local,local-1),0)
    maximum=float(error.max()) if error.size else 0.
    tolerance=float(4*np.finfo(np.float32).eps*max(grid_size))
    if maximum>tolerance:
        raise ValueError('native positions are not inside their declared voxels; check position units')
    result=VoxelAsset(coords,np.clip(local,0,1),flags,metadata.grid_size,metadata.aabb).canonical()
    if diagnostics is not None:
        diagnostics.update(upstream_commit=UPSTREAM_COMMIT,native_position_units='AABB-relative physical',
                           maximum_local_roundoff=maximum,clamped_components=int((error>0).sum()),
                           local_roundoff_tolerance=tolerance)
    return result


def mesh_to_voxels_native(vertices,faces,*,grid_size,aabb,face_weight=1.,boundary_weight=1.,
                          regularization_weight=.1,diagnostics=None):
    """Run the optional upstream converter on explicit CPU inputs.

    No topology/performance guarantee follows from using upstream. Pin the reviewed
    revision and measure reconstruction against source assets before training.
    """
    v,f,_,_=_mesh(vertices,faces)
    metadata=VoxelAsset(np.empty((0,3),dtype=int),np.empty((0,3)),np.empty((0,3),bool),grid_size,aabb)
    if (v<metadata.aabb[0]).any() or (v>metadata.aabb[1]).any():raise ValueError('vertices outside aabb')
    weights=np.asarray([face_weight,boundary_weight,regularization_weight],dtype=float)
    if not np.isfinite(weights).all() or (weights<0).any():raise ValueError('QEF weights must be finite nonnegative')
    import torch
    from o_voxel.convert import mesh_to_flexible_dual_grid
    coords,physical,flags=mesh_to_flexible_dual_grid(
        torch.tensor(v,dtype=torch.float32,device='cpu'),torch.tensor(f,dtype=torch.int32,device='cpu'),
        grid_size=metadata.grid_size,aabb=metadata.aabb,
        face_weight=face_weight,boundary_weight=boundary_weight,regularization_weight=regularization_weight)
    return from_native_grid(coords.cpu().numpy(),physical.cpu().numpy(),flags.cpu().numpy(),
                            grid_size=metadata.grid_size,aabb=metadata.aabb,diagnostics=diagnostics)
