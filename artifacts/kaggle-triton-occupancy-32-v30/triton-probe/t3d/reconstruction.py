"""Decode posterior-mean VAE outputs into raw geometry/PBR voxel assets."""
import numpy as np
import torch
from .dataset import batch_arrays
from .schema import VoxelAsset
from .sparse_compression import pack_children


@torch.no_grad()
def reconstruct_asset(asset, shape_model, material_model=None, *, support='predicted', threshold=.5,
                      edge_threshold=None):
    """Autoencoder reconstruction, not generative inference.

    teacher support exposes the source occupancy masks as an explicit diagnostic.
    predicted support uses decoder occupancy heads. Material decoding follows the
    reconstructed shape support; no source PBR values are copied into the result.
    """
    if support not in ('teacher','predicted') or shape_model.mode!='shape':
        raise ValueError('Expected shape model and teacher/predicted support')
    if edge_threshold is None:edge_threshold=threshold
    if not np.isfinite(threshold) or not 0<=threshold<=1 or not np.isfinite(edge_threshold) or not 0<=edge_threshold<=1:
        raise ValueError('Thresholds must be finite and in [0,1]')
    if asset.skeleton_field is not None or asset.skin_embedding is not None:
        raise ValueError('Rig latent decoding is not implemented')
    device=next(shape_model.parameters()).device
    coords,features=batch_arrays([asset],'shape')
    coords=torch.from_numpy(coords).to(device);features=torch.from_numpy(features).to(device)
    shape_model.eval()
    posterior=shape_model.encode(coords,features)
    out,raw,_=shape_model.decode(posterior.coords,posterior.mean,
                teacher_occupancy=posterior.occupancy if support=='teacher' else None,threshold=threshold)
    valid=(out[:,1:] < out.new_tensor(asset.grid_size)).all(dim=1)
    out,raw=out[valid],raw[valid]
    pbr=None
    if material_model is not None:
        if material_model.mode!='material' or next(material_model.parameters()).device!=device:
            raise ValueError('Material model must use the same device')
        if len(out)==0:
            pbr=np.empty((0,6))
        else:
            mc,mf=batch_arrays([asset],'material')
            material_model.eval()
            mp=material_model.encode(torch.from_numpy(mc).to(device),torch.from_numpy(mf).to(device))
            # Derive teacher masks from decoded shape; keep only surviving coarse parents.
            parent=out; dummy=raw[:,:1]; masks=[]
            for _ in range(4):
                parent,dense,mask=pack_children(parent,dummy)
                masks.append(mask);dummy=dense.mean(dim=1)
            lookup={tuple(row):i for i,row in enumerate(mp.coords.cpu().tolist())}
            ids=[lookup[tuple(row)] for row in parent.cpu().tolist()]
            material_coords,material_raw,_=material_model.decode(parent,mp.mean[ids],teacher_occupancy=masks)
            if not torch.equal(material_coords,out):
                raise ValueError('Material and shape decoder support mismatch')
            pbr=material_raw.sigmoid().cpu().numpy()
    result=VoxelAsset(out[:,1:].cpu().numpy(),raw[:,:3].sigmoid().cpu().numpy(),
                       (raw[:,3:6].sigmoid()>edge_threshold).cpu().numpy(),asset.grid_size,asset.aabb,
                       split_weight=raw[:,6:].cpu().numpy(),pbr=pbr)
    return result.canonical()


@torch.no_grad()
def encode_latents(asset,shape_model,material_model=None,*,shape_model_sha256,material_model_sha256=None):
    """Encode posterior means; caller identities must hash the actual decoder checkpoints."""
    from .latent_container import LatentAsset
    if shape_model.mode!='shape' or asset.skeleton_field is not None or asset.skin_embedding is not None:
        raise ValueError('Expected shape/material asset; rig latent encoding is pending')
    device=next(shape_model.parameters()).device
    c,f=batch_arrays([asset],'shape')
    shape_model.eval()
    posterior=shape_model.encode(torch.from_numpy(c).to(device),torch.from_numpy(f).to(device))
    material=None
    if material_model is not None:
        if material_model.mode!='material' or next(material_model.parameters()).device!=device:
            raise ValueError('Expected same-device material model')
        c,f=batch_arrays([asset],'material');material_model.eval()
        mp=material_model.encode(torch.from_numpy(c).to(device),torch.from_numpy(f).to(device))
        if not torch.equal(mp.coords,posterior.coords):raise ValueError('Latent supports differ')
        material=mp.mean.cpu().numpy()
    return LatentAsset(posterior.coords[:,1:].cpu().numpy(),posterior.mean.cpu().numpy(),
                       asset.grid_size,asset.aabb,shape_model_sha256,
                       material=material,material_model_sha256=material_model_sha256)
