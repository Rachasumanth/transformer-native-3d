"""Teacher-connectivity rendering supervision for a single raw shape asset."""
import torch
from .extraction_torch import voxels_to_mesh
from .render_loss import shape_render_loss


def shape_supervision(asset,coords,target,raw,*,resolution=64):
    """Render predicted and target dual vertices with source edge connectivity.

    The source support and flags define triangles; edge logits remain supervised
    by BCE. This provides gradients to dual vertices, not differentiable topology.
    """
    if (coords[:,0]!=0).any():raise ValueError('render supervision supports one asset per batch')
    grid=tuple(asset.grid_size)
    device=coords.device
    bounds=torch.tensor(asset.aabb.copy(),device=device,dtype=raw.dtype)
    with torch.no_grad():
        target_v,faces=voxels_to_mesh(coords[:,1:].long(),target[:,:3],target[:,3:6].bool(),grid,bounds,
                                     split_weight=target[:,6:7])
    if len(faces)==0:raise ValueError('source contains no renderable faces')
    voxel_size=(bounds[1]-bounds[0])/torch.tensor(grid,device=device,dtype=raw.dtype)
    predicted=bounds[0]+(coords[:,1:].to(raw.dtype)+raw[:,:3].sigmoid())*voxel_size
    center=bounds.mean(0);scale=1.8/(bounds[1]-bounds[0]).max()
    # Three orthographic views, normalized to fit the canonical AABB.
    rotations=raw.new_tensor([[[1,0,0],[0,1,0],[0,0,1]],
                              [[1,0,0],[0,0,1],[0,-1,0]],
                              [[0,0,1],[0,1,0],[-1,0,0]]])
    mvp=torch.eye(4,device=device,dtype=raw.dtype).repeat(3,1,1)
    mvp[:,:3,:3]=rotations*scale
    mvp[:,:3,3]=-torch.matmul(rotations,center)*scale
    terms=shape_render_loss(predicted,target_v,faces,mvp,resolution=resolution)
    return terms['silhouette']+terms['depth']+terms['normal']
