"""Raw T3DV Track A -> CUDA extraction/cleanup/baking -> binary glTF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from .schema import VoxelAsset
from .articulation import infer_skeleton_from_field, sample_sparse_embeddings


def _decode_export_skin(asset, prepared, checkpoint_path, device, joint_tolerance, max_skin_influences):
    import torch
    from .skin_ae import SkinAE
    path=Path(checkpoint_path)
    if not path.is_file():raise ValueError(f'SkinAE checkpoint does not exist: {path}')
    if asset.skeleton_field is None or asset.skin_embedding is None:
        raise ValueError('Rig export requires both skeleton_field and skin_embedding')
    if joint_tolerance is None:joint_tolerance=max(float(np.min(asset.voxel_size))*.25,1e-6)
    centers=asset.aabb[0]+(asset.coords.astype(float)+.5)*asset.voxel_size
    skeleton=infer_skeleton_from_field(centers,asset.skeleton_field,tolerance=joint_tolerance)
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    if not isinstance(checkpoint,dict) or checkpoint.get('version')!=1:
        raise ValueError('Unsupported SkinAE checkpoint format')
    config=checkpoint.get('config');state=checkpoint.get('model')
    if not isinstance(config,dict) or not isinstance(state,dict):
        raise ValueError('SkinAE checkpoint is missing config or model state')
    model=SkinAE(**config).to(device)
    model.load_state_dict(state,strict=True);model.freeze()
    with torch.inference_mode():
        joints=torch.as_tensor(skeleton.joints,device=device,dtype=torch.float32)
        parents=torch.as_tensor(skeleton.parents,device=device,dtype=torch.int64)
        joint_embeddings=model.encode_joints(joints,parents)
        vertex_embeddings,supported=sample_sparse_embeddings(prepared.vertices,
            asset.coords,asset.skin_embedding,grid_size=asset.grid_size,aabb=asset.aabb)
        if not bool(supported.all()):
            missing=int((~supported).sum().item())
            raise ValueError(f'Skin embedding sparse support missing at {missing} exported vertices')
        logits=model.decode_logits(vertex_embeddings,joint_embeddings)
        weights=torch.softmax(logits,dim=-1)
        if not torch.isfinite(weights).all():raise ValueError('SkinAE produced nonfinite weights')
        result=weights.detach().cpu().numpy()
    from .glb import limit_skin_influences
    result,influence_audit=limit_skin_influences(result,max_influences=max_skin_influences)
    return skeleton,result,hashlib.sha256(path.read_bytes()).hexdigest(),joint_tolerance,influence_audit


def export_raw(asset, *, device='cuda:0', target_faces=1000000, texture_size=2048,
               source_up='z', alpha_mode='auto', alpha_cutoff=.5, remesh=False,
               skinae_checkpoint=None, joint_tolerance=None, max_skin_influences=16):
    """Return (GLB bytes, execution audit). Requires optional CUDA export libraries.

    Accepts decoded raw geometry/materials. Latent profile 2 must first be decoded
    with matching trained checkpoints; rig fields require a frozen SkinAE checkpoint.
    """
    if not isinstance(asset,VoxelAsset):raise ValueError('expected a raw VoxelAsset')
    if asset.pbr is None:raise ValueError('raw PBR attributes are required')
    has_rig=asset.skeleton_field is not None or asset.skin_embedding is not None
    if (asset.skeleton_field is None)!=(asset.skin_embedding is None):
        raise ValueError('rig export requires both skeleton_field and skin_embedding')
    if has_rig and skinae_checkpoint is None:
        raise ValueError('rig export requires --skinae-checkpoint for SkinAE decoding')
    if not has_rig and skinae_checkpoint is not None:
        raise ValueError('SkinAE checkpoint supplied for an asset without rig fields')
    if joint_tolerance is not None and (not np.isfinite(joint_tolerance) or joint_tolerance<=0):
        raise ValueError('joint_tolerance must be finite and positive')
    if type(max_skin_influences) is not int or not 1<=max_skin_influences<=16:
        raise ValueError('max_skin_influences must be an integer in [1,16]')
    if source_up not in ('y','z'):raise ValueError('source_up must be y or z')
    if alpha_mode not in ('auto','OPAQUE','MASK','BLEND'):raise ValueError('invalid alpha mode')
    if type(target_faces) is not int or not 1<=target_faces<=1000000:raise ValueError('invalid face target')
    if type(texture_size) is not int or not 16<=texture_size<=8192:raise ValueError('invalid texture size')
    if not np.isfinite(alpha_cutoff) or not 0<=alpha_cutoff<=1:raise ValueError('invalid alpha cutoff')
    if type(remesh) is not bool:raise ValueError('remesh must be boolean')
    if alpha_mode=='auto':alpha_mode='BLEND' if (asset.pbr[:,5]<1).any() else 'OPAQUE'
    import torch
    from .extraction_torch import voxels_to_mesh
    from .mesh_postprocess import clean_and_unwrap
    from .texture_baking import bake_pbr
    from .glb import pack_glb
    destination=torch.device(device)
    if destination.type!='cuda' or not torch.cuda.is_available():raise ValueError('CUDA export device required')
    start=time.perf_counter()
    with torch.cuda.device(destination),torch.inference_mode():
        tensor=lambda x:torch.tensor(np.array(x,copy=True),device=destination)
        coords=tensor(asset.coords.astype(np.int64))
        vertices,faces=voxels_to_mesh(coords,tensor(asset.dual_vertices).float(),
            tensor(asset.intersected_flag),asset.grid_size,asset.aabb,
            None if asset.split_weight is None else tensor(asset.split_weight).float())
        prepared=clean_and_unwrap(vertices,faces,target_faces=target_faces,remesh=remesh,
                                  aabb=asset.aabb,grid_size=asset.grid_size)
        texture,mask=bake_pbr(prepared,coords,tensor(asset.pbr).float(),asset.grid_size,
                              asset.aabb,texture_size=texture_size)
        skeleton=skin_weights=None;skin_checkpoint_sha256=None
        resolved_joint_tolerance=None;skin_influence_audit=None
        if has_rig:
            skeleton,skin_weights,skin_checkpoint_sha256,resolved_joint_tolerance,skin_influence_audit=_decode_export_skin(
                asset,prepared,skinae_checkpoint,destination,joint_tolerance,max_skin_influences)
        data=pack_glb(prepared.vertices.cpu().numpy(),prepared.faces.cpu().numpy(),
            prepared.uv.cpu().numpy(),prepared.normals.cpu().numpy(),texture,
            source_up=source_up,alpha_mode=alpha_mode,alpha_cutoff=alpha_cutoff,
            skin_weights=skin_weights,skeleton=skeleton)
        torch.cuda.synchronize(destination)
    audit=dict(device=str(destination),device_name=torch.cuda.get_device_name(destination),
        active_voxels=len(coords),extracted_faces=len(faces),uv_faces=len(prepared.faces),
        stages=prepared.audit,texture_size=texture_size,valid_texels=int(mask.sum()),
        source_up=source_up,alpha_mode=alpha_mode,remesh=remesh,
        skinned=has_rig,joint_count=0 if skeleton is None else len(skeleton.joints),
        skinae_checkpoint_sha256=skin_checkpoint_sha256,
        skin_influence_audit=skin_influence_audit,
        joint_tolerance=resolved_joint_tolerance,
        total_seconds=time.perf_counter()-start,glb_bytes=len(data),
        glb_sha256=hashlib.sha256(data).hexdigest())
    return data,audit


def _atomic_write(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name)
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--target-faces',type=int,default=1000000)
    parser.add_argument('--texture-size',type=int,default=2048)
    parser.add_argument('--source-up',choices=['y','z'],default='z')
    parser.add_argument('--alpha-mode',choices=['auto','OPAQUE','MASK','BLEND'],default='auto')
    parser.add_argument('--alpha-cutoff',type=float,default=.5)
    parser.add_argument('--remesh',action='store_true')
    parser.add_argument('--skinae-checkpoint',type=Path)
    parser.add_argument('--joint-tolerance',type=float)
    parser.add_argument('--max-skin-influences',type=int,default=16)
    args=parser.parse_args()
    audit_path=args.output.with_suffix(args.output.suffix+'.json')
    if args.input.resolve() in (args.output.resolve(),audit_path.resolve()):
        parser.error('input must differ from output and audit paths')
    from .container import loads
    source=args.input.read_bytes()
    container=loads(source)
    data,audit=export_raw(container.asset,device=args.device,target_faces=args.target_faces,
        texture_size=args.texture_size,source_up=args.source_up,alpha_mode=args.alpha_mode,
        alpha_cutoff=args.alpha_cutoff,remesh=args.remesh,
        skinae_checkpoint=args.skinae_checkpoint,joint_tolerance=args.joint_tolerance,
        max_skin_influences=args.max_skin_influences)
    audit.update(input_sha256=hashlib.sha256(source).hexdigest(),
                 bpt_token_count=len(container.bpt_tokens),geometry_track='raw Track A')
    _atomic_write(args.output,data)
    _atomic_write(audit_path,(json.dumps(audit,indent=2)+'\n').encode())
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
