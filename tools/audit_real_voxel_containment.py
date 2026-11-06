"""Audit surface, AABB and filled-shell bone support on a real input rig."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from t3d.container import loads
from t3d.import_rig import load_rig_glb
from t3d.coupled_occupancy import shape_bounding_box_mask,closed_shell_solid_mask

torch.set_num_threads(2);root=Path('artifacts/kaggle-textured-rig-export-v14');p=root/'export-real-rigged-input.t3d';asset=loads(p.read_bytes()).asset;source=Path('artifacts/rigged-inputs/CesiumMan/CesiumMan.glb');rig,_=load_rig_glb(source).normalized();rows=[]
for n in (16,32,64):
 if any(x%n for x in asset.grid_size):raise ValueError('Expected integer downsample')
 scaled=np.asarray(asset.coords,dtype=np.int64)//(np.array(asset.grid_size)//n);surface=np.zeros((n,n,n),bool);surface[scaled[:,0],scaled[:,1],scaled[:,2]]=True
 bone=[];joints=np.asarray(rig.skeleton.joints)
 for i,parent in enumerate(rig.skeleton.parents):
  if parent<0:continue
  a=joints[i];b=joints[parent];count=max(2,int(np.ceil(np.linalg.norm(a-b)*n*4))+1);bone.extend(a[None,:]*(1-np.linspace(0,1,count))[:,None]+b[None,:]*np.linspace(0,1,count)[:,None])
 bone=np.asarray(bone);voxel=np.floor((bone-asset.aabb[0])/(asset.aabb[1]-asset.aabb[0])*n).astype(int);out=((voxel<0)|(voxel>=n)).any(1);voxel=np.clip(voxel,0,n-1);vox=np.unique(voxel[~out],axis=0)
 coords=torch.cartesian_prod(*(torch.arange(n) for _ in range(3)));sm=torch.from_numpy(surface.reshape(-1));bbox=shape_bounding_box_mask(coords,sm).numpy().reshape(n,n,n);solid=closed_shell_solid_mask(coords,sm).numpy().reshape(n,n,n);ix=tuple(vox.T)
 rows.append(dict(grid=n,surface_voxels=int(surface.sum()),solid_voxels=int(solid.sum()),interior_voxels=int((solid&~surface).sum()),bone_sample_out_of_aabb=int(out.sum()),bone_voxels=len(vox),bone_on_surface=int(surface[ix].sum()),bone_in_bounding_box=int(bbox[ix].sum()),bone_in_filled_shell=int(solid[ix].sum())))
r=dict(scope='CesiumMan bind skeleton segments sampled and rasterized at coarse resolution; downsampled 128-grid surface and 6-connected fill, not a watertight ground-truth occupancy or bone containment benchmark',container_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows);Path('artifacts/real-voxel-containment-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(rows,indent=2))
