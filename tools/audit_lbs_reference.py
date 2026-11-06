"""Check matrix-based LBS reference against directly sampled source glTF poses."""
import json
from pathlib import Path
import numpy as np
from t3d.import_rig import load_rig_glb,rig_joint_world
from t3d.articulation import joint_deformation_matrices,linear_blend_skinning
from t3d.gltf_accessors import GLBAccessors

rows=[]
for asset in ('RiggedSimple','CesiumMan','Fox'):
 path=Path('artifacts/rigged-inputs')/asset/f'{asset}.glb'
 rig,transform=load_rig_glb(path).normalized();_,bind=rig_joint_world(path)
 c=np.eye(4);c[:3,:3]*=transform.scale;c[:3,3]=-transform.scale*np.asarray(transform.center)
 glb=GLBAccessors(path)
 for clip,animation in enumerate(glb.document.get('animations',[])):
  keys=[glb.read(s['input']).reshape(-1) for s in animation['samplers']]
  for time in np.linspace(min(k.min() for k in keys),max(k.max() for k in keys),5):
   posed=load_rig_glb(path,animation_index=clip,time=float(time))
   _,world=rig_joint_world(path,animation_index=clip,time=float(time))
   matrix=c@joint_deformation_matrices(bind,world)@np.linalg.inv(c)
   reference=transform.apply(posed.vertices)
   predicted=linear_blend_skinning(rig.vertices,rig.weights,matrix)
   error=np.linalg.norm(predicted-reference,axis=1)
   rows.append(dict(asset=asset,clip=clip,time=float(time),mean=float(error.mean()),max=float(error.max())))
out=Path('artifacts/kaggle-skin-position-ablation-v3/lbs-reference-audit.json')
out.write_text(json.dumps(rows,indent=2)+'\n')
for asset in ('RiggedSimple','CesiumMan','Fox'):
 subset=[r for r in rows if r['asset']==asset]
 print(asset,'poses',len(subset),'max error',max(r['max'] for r in subset))
