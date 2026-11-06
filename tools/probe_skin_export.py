"""Exercise voxel-field recovery and SkinAE decoding on a CUDA device."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from t3d.articulation import (Skeleton, attach_articulation, sample_sparse_embeddings,
                              skeleton_displacements)
from t3d.export import _decode_export_skin
from t3d.schema import VoxelAsset
from t3d.skin_ae import SkinAE


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('/kaggle/working/skin-export-probe.json'))
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for this probe')
    device=torch.device('cuda:0')
    checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=True)
    model=SkinAE(**checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'],strict=True);model.freeze()
    skeleton=Skeleton([[.25,.25,.25],[.75,.25,.25],[.75,.75,.25]],[-1,0,1])
    grid=(8,8,8);aabb=np.array([[0.,0.,0.],[1.,1.,1.]])
    coords=np.asarray([(x,y,z) for x in range(8) for y in range(8) for z in range(8)],dtype=np.int32)
    points=aabb[0]+(coords+.5)/8
    _,labels=skeleton_displacements(points,skeleton)
    weights=np.eye(3,dtype=np.float32)[labels]
    with torch.inference_mode():
        joint_embeddings=model.encode_joints(
            torch.as_tensor(skeleton.joints,device=device,dtype=torch.float32),
            torch.as_tensor(skeleton.parents,device=device,dtype=torch.int64))
    asset=VoxelAsset(coords,np.full((len(coords),3),.5),np.ones((len(coords),3),dtype=bool),
        grid,aabb,pbr=np.tile([.5,.5,.5,0,.5,1.],(len(coords),1)))
    asset=attach_articulation(asset,skeleton,voxel_weights=weights,
        joint_embeddings=joint_embeddings.detach().cpu().numpy())
    prepared=SimpleNamespace(vertices=torch.as_tensor(points,device=device,dtype=torch.float32))
    recovered,predicted,digest,tolerance=_decode_export_skin(
        asset,prepared,args.checkpoint,device,1e-6)
    if not np.array_equal(recovered.parents,skeleton.parents):
        raise AssertionError('Recovered hierarchy differs from the source field')
    if not np.allclose(recovered.joints,skeleton.joints,atol=1e-6,rtol=0):
        raise AssertionError('Recovered joint positions differ from the source field')
    sums=predicted.sum(axis=1)
    if not np.isfinite(predicted).all() or np.any(predicted<0) or not np.allclose(sums,1,atol=1e-6):
        raise AssertionError('Decoded skin weights are not finite normalized distributions')
    lattice_features=np.column_stack((coords,coords[:,0]+2*coords[:,1]+3*coords[:,2])).astype(np.float32)
    query_lattice=np.array([[2.25,3.5,1.75]],dtype=np.float32)
    query_world=aabb[0]+(query_lattice+.5)/8
    interpolated,supported=sample_sparse_embeddings(
        torch.as_tensor(query_world,device=device),coords,lattice_features,
        grid_size=grid,aabb=aabb)
    expected=np.array([[2.25,3.5,1.75,2.25+2*3.5+3*1.75]],dtype=np.float32)
    interpolation_error=float(np.max(np.abs(interpolated.cpu().numpy()-expected)))
    if not bool(supported.all()) or interpolation_error>1e-5:
        raise AssertionError('Sparse trilinear interpolation failed its affine fixture')
    _,unsupported=sample_sparse_embeddings(
        torch.as_tensor((aabb[0]+.5/8)[None],device=device,dtype=torch.float32),
        np.array([[7,7,7]],dtype=np.int32),np.ones((1,4),dtype=np.float32),
        grid_size=grid,aabb=aabb)
    if bool(unsupported.any()):raise AssertionError('Missing sparse support was not reported')
    report=dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,
        checkpoint_sha256=digest,joint_count=len(recovered.joints),vertex_count=len(predicted),
        joint_tolerance=tolerance,support_count=int(len(predicted)),
        interpolation_max_error=interpolation_error,unsupported_corner_handling='reported',
        minimum_weight_sum=float(sums.min()),maximum_weight_sum=float(sums.max()),
        minimum_weight=float(predicted.min()),maximum_weight=float(predicted.max()),
        mean_max_joint_probability=float(predicted.max(axis=1).mean()),
        scope='Synthetic CUDA probe of field clustering, sparse feature sampling, and frozen SkinAE decoding; no mesh extraction or quality claim')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
