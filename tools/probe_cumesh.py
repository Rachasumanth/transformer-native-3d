"""Compile pinned CuMesh on Kaggle and exercise cleanup/UV on extracted geometry."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

COMMIT='12289e1062f0603f2f0d0771b02e1395d247f26f'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-sha256',required=True)
    args=parser.parse_args()
    log=Path('/kaggle/working/cumesh-build.log')
    from tools.build_cuda_dependency import build_and_install
    with log.open('w') as stream:
        build_and_install('CuMesh','https://github.com/JeffreyXiang/CuMesh.git',COMMIT,stream)
    import numpy as np
    import torch
    from t3d.voxelize import mesh_to_voxels
    from t3d.extraction_torch import voxels_to_mesh
    from t3d.mesh_postprocess import clean_and_unwrap
    from t3d.topology import orient_faces
    v=np.array([[x,y,z] for x in (.2,.8) for y in (.2,.8) for z in (.2,.8)])
    faces=[]
    for axis in range(3):
        for side in (.2,.8):
            a,b,c,d=np.flatnonzero(v[:,axis]==side)
            faces.extend([[a,b,d],[a,d,c]])
    asset=mesh_to_voxels(v,faces,grid_size=(8,)*3,aabb=[[0]*3,[1]*3])
    tensor=lambda a:torch.tensor(np.array(a,copy=True),device='cuda')
    vertices,triangles=voxels_to_mesh(tensor(asset.coords.astype(np.int64)),tensor(asset.dual_vertices).float(),
        tensor(asset.intersected_flag),(8,)*3,[[0]*3,[1]*3])
    # Boolean crossings carry no sign; establish consistent seed geometry first.
    oriented,audit=orient_faces(vertices.cpu().numpy(),triangles.cpu().numpy(),outward=True)
    results=[]
    for remesh in (False,True):
        prepared=clean_and_unwrap(vertices,tensor(oriented),target_faces=200,
            remesh=remesh,aabb=[[0]*3,[1]*3],grid_size=(8,)*3)
        assert torch.isfinite(prepared.uv).all() and torch.isfinite(prepared.normals).all()
        assert ((prepared.uv>=0)&(prepared.uv<=1)).all()
        assert len(prepared.faces)>0 and prepared.faces.min()>=0 and prepared.faces.max()<len(prepared.vertices)
        assert prepared.uv.shape==(len(prepared.vertices),2)
        positions=prepared.vertices.cpu().numpy()
        unique,inverse=np.unique(positions,axis=0,return_inverse=True)
        welded=inverse[prepared.faces.cpu().numpy()]
        _,topology=orient_faces(unique,welded,outward=True)
        assert topology['components'] and all(c['closed'] for c in topology['components'])
        torch.save(dict(vertices=prepared.vertices.cpu(),faces=prepared.faces.cpu(),uv=prepared.uv.cpu(),
                        normals=prepared.normals.cpu()),f'/kaggle/working/prepared-cube-{remesh}.pt')
        results.append(dict(remesh=remesh,stages=prepared.audit,uv_vertices=len(positions),uv_faces=len(welded),
            closed_after_position_weld=True,finite_uv_normals=True,topology=topology,
            source_component_count_preserved=len(topology['components'])==1,
            outward_orientation_preserved=topology['flipped_faces']==0))
    report=dict(source_sha256=args.source_sha256,cumesh_commit=COMMIT,device=torch.cuda.get_device_name(0),
                results=results,scope='analytic extracted cube; no texture baking or GLB export')
    Path('/kaggle/working/cumesh-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
