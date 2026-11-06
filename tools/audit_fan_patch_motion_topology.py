"""Inspect welded topology and triangle area at sampled retargeted poses."""
from pathlib import Path
import collections,hashlib,json
import numpy as np
from t3d.import_rig import load_rig_glb
from t3d.gltf_accessors import GLBAccessors
p=Path('artifacts/kaggle-textured-rig-export-v14/fan-patch-posed/candidate-animated.glb');g=GLBAccessors(p);anim=g.document['animations'][0];keys=[g.read(s['input']).reshape(-1) for s in anim['samplers']];rows=[]
for time in np.linspace(min(k.min() for k in keys),max(k.max() for k in keys),25):
 rig=load_rig_glb(p,animation_index=0,time=float(time));v=np.asarray(rig.vertices);f=np.asarray(rig.faces);tri=v[f];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2;_,inv=np.unique(v,axis=0,return_inverse=True);wf=inv[f];edges=collections.Counter(tuple(sorted((int(a),int(b)))) for t in wf for a,b in zip(t,np.roll(t,-1)));rows.append(dict(time=float(time),degenerate_faces=int(np.sum(areas<=1e-12)),min_area=float(areas.min()),boundary_edges=sum(x==1 for x in edges.values()),nonmanifold_edges=sum(x>2 for x in edges.values())))
r={'scope':'25 sampled times, exact-position weld, topology only; no continuous-motion or visual guarantee','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'poses':rows};out=p.parent/'motion-topology-25frames.json';out.write_text(json.dumps(r,indent=2)+'\n');print('poses',len(rows),'max boundary',max(x['boundary_edges'] for x in rows),'max nonmanifold',max(x['nonmanifold_edges'] for x in rows),'max degenerates',max(x['degenerate_faces'] for x in rows),'min area',min(x['min_area'] for x in rows))
