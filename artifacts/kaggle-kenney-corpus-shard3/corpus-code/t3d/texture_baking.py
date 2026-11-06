"""CUDA UV rasterization, BVH reprojection and sparse six-channel PBR baking."""
import numpy as np
import torch


@torch.no_grad()
def bake_pbr(prepared, coords, pbr, grid_size, aabb, *, texture_size=2048,
             raster_chunk_size=100000):
    """Return CPU H×W×6 linear PBR and an original valid-texel mask.

    Uses the plan's nvdiffrast, CuMesh BVH and FlexGEMM sparse sampler. Sparse
    samples live at cell centers (coord + 0.5) in FlexGEMM's voxel units;
    queries are (world - aabb_min) / voxel_size. Missing support fails explicitly.
    """
    import nvdiffrast.torch as dr
    from flex_gemm.ops.grid_sample import grid_sample_3d
    device=prepared.vertices.device
    if not prepared.vertices.is_cuda or coords.device!=device or pbr.device!=device:
        raise ValueError('prepared geometry, coordinates and PBR must share CUDA device')
    if coords.ndim!=2 or coords.shape[1]!=3 or coords.dtype not in (torch.int32,torch.int64):
        raise ValueError('coords must be integer (L,3)')
    if pbr.shape!=(len(coords),6) or not pbr.is_floating_point() or not len(coords):
        raise ValueError('PBR must be floating nonempty (L,6)')
    if not torch.isfinite(pbr).all() or not ((pbr>=0)&(pbr<=1)).all():
        raise ValueError('PBR values must be finite in [0,1]')
    if type(texture_size) is not int or not 16<=texture_size<=8192:
        raise ValueError('texture_size must be an integer in [16,8192]')
    if type(raster_chunk_size) is not int or not 1<=raster_chunk_size<=100000:
        raise ValueError('raster_chunk_size must be in [1,100000]')
    if len(grid_size)!=3 or any(type(x) is not int or x<=0 for x in grid_size):
        raise ValueError('invalid grid dimensions')
    grid=torch.tensor(grid_size,device=device)
    if not ((coords>=0)&(coords<grid)).all() or len(torch.unique(coords,dim=0))!=len(coords):
        raise ValueError('invalid or duplicate coordinates')
    bounds=torch.as_tensor(aabb,device=device,dtype=torch.float32)
    if bounds.shape!=(2,3) or not torch.isfinite(bounds).all() or not (bounds[1]>bounds[0]).all():
        raise ValueError('invalid aabb')
    with torch.cuda.device(device):
        uv=prepared.uv.float()
        clip=torch.cat([uv*2-1,torch.zeros_like(uv[:,:1]),torch.ones_like(uv[:,:1])],-1)[None].contiguous()
        faces=prepared.faces.int().contiguous()
        raster=torch.zeros((1,texture_size,texture_size,4),device=device)
        context=dr.RasterizeCudaContext(device=device)
        for first in range(0,len(faces),raster_chunk_size):
            block,_=dr.rasterize(context,clip,faces[first:first+raster_chunk_size],resolution=[texture_size]*2)
            valid=block[...,3:4]>0
            block[...,3:4]+=first
            raster=torch.where(valid,block,raster)
        mask=raster[0,...,3]>0
        if not mask.any():raise ValueError('UV atlas rasterized no texels')
        positions=dr.interpolate(prepared.vertices.float()[None].contiguous(),raster,faces)[0][0][mask]
        features=torch.cat([pbr.float(),torch.ones((len(pbr),1),device=device)],-1).contiguous()
        batch_coords=torch.cat([torch.zeros_like(coords[:,:1]),coords],-1).int().contiguous()
        samples=[]
        for first in range(0,len(positions),100000):
            points=positions[first:first+100000].contiguous()
            _,ids,bary=prepared.reference_bvh.unsigned_distance(points,return_uvw=True)
            if (ids<0).any() or (ids>=len(prepared.reference_faces)).any():
                raise ValueError('BVH returned invalid projection face')
            tri=prepared.reference_vertices[prepared.reference_faces[ids.long()].long()]
            projected=(tri*bary[...,None]).sum(1)
            query=((projected-bounds[0])/((bounds[1]-bounds[0])/grid)).float()[None].contiguous()
            value=grid_sample_3d(features,batch_coords,shape=torch.Size([1,7,*grid_size]),grid=query,mode='trilinear').reshape(-1,7)
            if not torch.isfinite(value).all() or (value[:,6]<.5).any():
                raise ValueError('baked texel has no valid sparse material support')
            material=value[:,:6]
            # Sparse trilinear kernels may round constant boundary channels a
            # few ulps outside their convex [0,1] range. Clamp only that numeric
            # drift; larger excursions indicate invalid sampling or attributes.
            if (material < -1e-5).any() or (material > 1+1e-5).any():
                raise ValueError('interpolated PBR values exceed their valid [0,1] range')
            samples.append(material.clamp(0,1))
        image=torch.zeros((texture_size,texture_size,6),device=device)
        image[mask]=torch.cat(samples)
        image=image.cpu().numpy()
        valid=mask.cpu().numpy()
    from .texture_padding import pad_pbr
    return pad_pbr(image,valid),valid
