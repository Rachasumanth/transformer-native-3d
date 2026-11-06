"""Build a single-asset coupled occupancy pilot from the archived v14 field."""
import hashlib,json
from pathlib import Path
import numpy as np
from t3d.container import loads
from t3d.import_rig import load_rig_glb
root=Path('artifacts/kaggle-textured-rig-export-v14');src=root/'export-real-rigged-input.t3d';source=Path('artifacts/rigged-inputs/CesiumMan/CesiumMan.glb');asset=loads(src.read_bytes()).asset;rig,_=load_rig_glb(source).normalized();n=16
if any(x%n for x in asset.grid_size):raise ValueError('Expected integer downsample')
coords=np.asarray(asset.coords,dtype=np.int64)//(np.asarray(asset.grid_size)//n);surface=np.zeros((n,n,n),dtype=np.float32);surface[coords[:,0],coords[:,1],coords[:,2]]=1
j=np.asarray(rig.skeleton.joints);bone=[]
for i,parent in enumerate(rig.skeleton.parents):
 if parent<0:continue
 a=j[i];b=j[parent];count=max(2,int(np.ceil(np.linalg.norm(a-b)*n*4))+1);t=np.linspace(0,1,count)[:,None];bone.extend(a[None,:]*(1-t)+b[None,:]*t)
bone=np.asarray(bone);vox=np.floor((bone-asset.aabb[0])/(asset.aabb[1]-asset.aabb[0])*n).astype(int)
if ((vox<0)|(vox>=n)).any():raise ValueError('Bone samples outside grid')
skeleton=np.zeros_like(surface);skeleton[vox[:,0],vox[:,1],vox[:,2]]=1
out=Path('artifacts/coupled-real-pilot');out.mkdir(exist_ok=True);data=out/'cesiumman-16.npz';np.savez_compressed(data,shape=surface,skeleton=skeleton)
manifest=dict(scope='One CesiumMan training asset; 128-to-16 surface occupancy and sampled bind skeleton segments; no held-out data and no image/text conditioning',asset_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),shape_count=int(surface.sum()),skeleton_count=int(skeleton.sum()),grid=n)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
