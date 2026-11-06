"""Create a tiny prompt-labeled occupancy pilot from known CC0/CC-BY rigs."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from t3d.import_rig import load_rig_glb
from t3d.voxelize import mesh_to_voxels
from t3d.coupled_occupancy import closed_shell_solid_mask,shape_bounding_box_mask

torch.set_num_threads(2);n=16;root=Path('artifacts/prompt-occupancy-pilot');root.mkdir(exist_ok=True);rows=[]
items=[('Corgi','train','a 3D model of a corgi dog'),('Bat','train','a 3D model of a bat'),('Shark','train','a 3D model of a shark'),('RiggedFigure','validation','a 3D model of a humanoid figure')]
for name,split,prompt in items:
 p=Path('artifacts/rigged-candidates')/name/(name+'.glb');rig,tr=load_rig_glb(p).normalized();vertices=np.clip(rig.vertices,-.5,.5);start=time.monotonic();voxel=mesh_to_voxels(vertices,rig.faces,grid_size=(n,n,n),aabb=[[-.5]*3,[.5]*3]);surface=np.zeros((n,n,n),np.float32);c=np.asarray(voxel.coords,dtype=int);surface[c[:,0],c[:,1],c[:,2]]=1
 joints=np.asarray(rig.skeleton.joints);samples=[]
 for i,parent in enumerate(rig.skeleton.parents):
  if parent<0:continue
  a=joints[i];b=joints[parent];count=max(2,int(np.ceil(np.linalg.norm(a-b)*n*4))+1);t=np.linspace(0,1,count)[:,None];samples.extend(a[None,:]*(1-t)+b[None,:]*t)
 points=np.asarray(samples);xyz=np.floor((points+.5)*n).astype(int);outside=((xyz<0)|(xyz>=n)).any(1);xyz=xyz[~outside];skel=np.zeros_like(surface)
 if len(xyz):skel[xyz[:,0],xyz[:,1],xyz[:,2]]=1
 coords=torch.cartesian_prod(*(torch.arange(n) for _ in range(3)));shape=torch.from_numpy(surface.reshape(-1));bones=torch.from_numpy(skel.reshape(-1)).bool();box=shape_bounding_box_mask(coords,shape.bool());solid=closed_shell_solid_mask(coords,shape.bool());target=root/(name+'.npz');np.savez_compressed(target,shape=surface,skeleton=skel)
 rows.append(dict(name=name,split=split,prompt=prompt,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),data_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),file=target.name,normalization=dict(center=tr.center,scale=tr.scale),surface_count=int(surface.sum()),bone_count=int(skel.sum()),bone_samples_outside_grid=int(outside.sum()),bone_outside_bbox=int((bones&~box).sum()),bone_outside_enclosed=int((bones&~solid).sum()),voxelization_seconds=time.monotonic()-start))
 print(name,'surface',rows[-1]['surface_count'],'bones',rows[-1]['bone_count'],'outside shell',rows[-1]['bone_outside_enclosed'],flush=True)
(root/'manifest.json').write_text(json.dumps(dict(scope='Four small source assets, hand-written class prompts, animal training family and RiggedFigure validation; not a caption corpus or held-out benchmark',grid=n,assets=rows),indent=2)+'\n')
