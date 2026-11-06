"""Build area-weighted vertex samples with exact source LBS pose targets."""
import hashlib,json
from pathlib import Path
import numpy as np
from t3d.import_rig import load_rig_glb,rig_joint_world
from t3d.articulation import joint_deformation_matrices,linear_blend_skinning

src=Path('artifacts/skin-family-pilot');out=Path('artifacts/skin-family-vertex-pilot');out.mkdir(exist_ok=True)
m=json.loads((src/'manifest.json').read_text());rows=[]
for row in m['assets']:
 p=src/row['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 source=Path('artifacts/rigged-inputs')/row['name']/(row['name']+'.glb');assert hashlib.sha256(source.read_bytes()).hexdigest()==row['source_sha256']
 rig,tr=load_rig_glb(source).normalized();_,bind=rig_joint_world(source)
 with np.load(p,allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
 tri=rig.vertices[rig.faces];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
 rng=np.random.default_rng(1700+sum(row['name'].encode()));face=rng.choice(len(area),size=1024,p=area/area.sum());corner=rng.integers(0,3,size=1024);vertex=rig.faces[face,corner]
 rest=rig.vertices[vertex];weights=rig.weights[vertex];c=np.eye(4);c[:3,:3]*=tr.scale;c[:3,3]=-tr.scale*np.asarray(tr.center);points=[];mats=[];maxerr=0.
 for clip,time in zip(arrays['animation_index'],arrays['animation_time']):
  clip=int(clip);time=float(time);posed=load_rig_glb(source,animation_index=clip if clip>=0 else None,time=time if clip>=0 else None)
  _,world=rig_joint_world(source,animation_index=clip if clip>=0 else None,time=time if clip>=0 else None)
  matrix=c@joint_deformation_matrices(bind,world)@np.linalg.inv(c);target=np.asarray(tr.apply(posed.vertices))[vertex]
  maxerr=max(maxerr,float(np.linalg.norm(linear_blend_skinning(rest,weights,matrix)-target,axis=1).max()))
  points.append(target.astype(np.float32));mats.append(matrix.astype(np.float32))
 arrays.update(points=np.stack(points),weights=weights.astype(np.float32),bind_points=rest.astype(np.float32),deformations=np.stack(mats),source_faces=face.astype(np.int64),barycentric=np.eye(3,dtype=np.float32)[corner],source_vertices=vertex.astype(np.int64))
 dest=out/p.name;np.savez_compressed(dest,**arrays);rows.append({**row,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'max_lbs_target_error':maxerr})
m['assets']=rows;m['scope']='Three original training rigs, Gobkit animal family held out for validation; area-weighted vertex samples, not a representative corpus';m['sampling']='1024 vertices sampled with face area weighting; shared normalization and exact original vertex skinning';(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ('name','max_lbs_target_error')} for r in rows],indent=2))
