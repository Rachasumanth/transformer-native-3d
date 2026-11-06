"""Differentiable multi-view shape supervision using nvdiffrast on CUDA."""
import torch
import torch.nn.functional as F


def shape_render_loss(predicted, target, faces, mvp, *, resolution=128):
    """Compare silhouette, depth and interpolated normals for aligned meshes.

    Both surfaces share triangle connectivity (e.g. source edge flags on teacher
    support); predicted vertex positions remain differentiable. Edge selection
    itself is discrete and receives no gradient. MVP is (B,4,4), clip-space w
    must be positive for all vertices. Returned masks use antialiasing for gradients.
    """
    import nvdiffrast.torch as dr
    if not predicted.is_cuda or predicted.device!=target.device or predicted.device!=faces.device or predicted.device!=mvp.device:
        raise ValueError('inputs must share a CUDA device')
    if predicted.shape!=target.shape or predicted.ndim!=2 or predicted.shape[1]!=3 or not len(predicted):
        raise ValueError('vertices must share nonempty (V,3) shape')
    if faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or faces.dtype not in (torch.int32,torch.int64):
        raise ValueError('faces must be nonempty integer (F,3)')
    if (faces<0).any() or (faces>=len(predicted)).any():raise ValueError('face index out of range')
    if mvp.ndim!=3 or mvp.shape[1:]!=(4,4) or not len(mvp):raise ValueError('mvp must be (B,4,4)')
    if type(resolution) is not int or resolution<8:raise ValueError('resolution must be >=8')
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all() or not torch.isfinite(mvp).all():
        raise ValueError('inputs must be finite')
    faces=faces.int().contiguous()
    ctx=dr.RasterizeCudaContext(device=predicted.device)
    def render(vertices):
        ones=torch.ones_like(vertices[:,:1])
        clip=torch.matmul(torch.cat([vertices,ones],1)[None],mvp.transpose(1,2)).contiguous()
        if not (clip[...,3]>0).all():raise ValueError('all clip-space w values must be positive')
        rast,_=dr.rasterize(ctx,clip,faces,resolution=[resolution,resolution])
        mask=(rast[...,3:4]>0).to(vertices.dtype)
        silhouette=dr.antialias(mask,rast,clip,faces)
        tri=vertices[faces.long()]
        face_normals=torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=1)
        normals=torch.zeros_like(vertices)
        for i in range(3):normals=normals.index_add(0,faces[:,i].long(),face_normals)
        normals=F.normalize(normals,dim=1,eps=1e-12)
        normal=dr.interpolate(normals[None].expand(len(mvp),-1,-1).contiguous(),rast,faces)[0]
        normal=F.normalize(normal,dim=-1,eps=1e-12)
        return silhouette,mask,rast[...,2:3],normal
    p,pm,pd,pn=render(predicted)
    with torch.no_grad():t,tm,td,tn=render(target)
    overlap=pm*tm
    denom=overlap.sum().clamp_min(1)
    silhouette=(p-t).abs().mean()
    depth=((pd-td).abs()*overlap).sum()/denom
    normal=((1-(pn*tn).sum(-1,keepdim=True).clamp(-1,1))*overlap).sum()/denom
    return dict(silhouette=silhouette,depth=depth,normal=normal,overlap_pixels=overlap.sum())
