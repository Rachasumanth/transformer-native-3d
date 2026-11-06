"""Add canonical bind samples and per-pose joint deformations to pose pilot data."""
import hashlib,json
from pathlib import Path
import numpy as np
from t3d.import_rig import load_rig_glb,rig_joint_world
from t3d.articulation import joint_deformation_matrices,linear_blend_skinning

src=Path('artifacts/skin-pose-pilot');out=Path('artifacts/skin-deformation-pilot');out.mkdir(exist_ok=True)
m=json.loads((src/'manifest.json').read_text()); rows=[]
for row in m['assets']:
 p=src/row['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 source=Path('artifacts/rigged-inputs')/row['name']/(row['name']+'.glb')
 assert hashlib.sha256(source.read_bytes()).hexdigest()==row['source_sha256']
 rig,transform=load_rig_glb(source).normalized();_,bind=rig_joint_world(source)
 c=np.eye(4);c[:3,:3]*=transform.scale;c[:3,3]=-transform.scale*np.asarray(transform.center)
 with np.load(p,allow_pickle=False) as data: arrays={k:data[k] for k in data.files}
 faces=rig.faces[arrays['source_faces']];bary=arrays['barycentric']
 rest=np.einsum('ni,nij->nj',bary,rig.vertices[faces]);deformations=[];max_error=0.
 for clip,time,points in zip(arrays['animation_index'],arrays['animation_time'],arrays['points']):
  _,world=rig_joint_world(source,animation_index=int(clip) if clip>=0 else None,time=float(time) if clip>=0 else None)
  matrix=c@joint_deformation_matrices(bind,world)@np.linalg.inv(c)
  # Barycentric interpolation of posed triangle vertices need not equal LBS of
  # interpolated positions and weights; retain that discrepancy as evidence.
  predicted=linear_blend_skinning(rest,arrays['weights'],matrix)
  max_error=max(max_error,float(np.linalg.norm(predicted-points,axis=1).max()))
  deformations.append(matrix.astype(np.float32))
 arrays['bind_points']=rest.astype(np.float32);arrays['deformations']=np.stack(deformations)
 dest=out/p.name;np.savez_compressed(dest,**arrays)
 rows.append({**row,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
              'max_interpolated_lbs_vs_posed_triangle':max_error})
m['assets']=rows;m['scope']+='; canonical bind samples and per-pose LBS transforms added'
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ('name','max_interpolated_lbs_vs_posed_triangle')} for r in rows],indent=2))
