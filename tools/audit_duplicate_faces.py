"""Compare winding, texture coordinates, and skin weights on coincident faces."""
import collections,hashlib,json
from pathlib import Path
import numpy as np
from t3d.import_rig import load_rig_glb
from t3d.gltf_accessors import GLBAccessors
p=Path('artifacts/kaggle-textured-rig-export-v14/export-real-rigged.glb');rig=load_rig_glb(p);doc=GLBAccessors(p);prim=doc.document['meshes'][0]['primitives'][0];uv=doc.read(prim['attributes']['TEXCOORD_0']);v=np.asarray(rig.vertices);f=np.asarray(rig.faces);weights=np.asarray(rig.weights);unique,inv=np.unique(v,axis=0,return_inverse=True);w=inv[f];groups=collections.defaultdict(list)
for i,t in enumerate(w):groups[tuple(sorted(map(int,t)))].append(i)
rows=[]
for key,ids in groups.items():
 if len(ids)<2:continue
 for j in ids[1:]:
  i=ids[0];a=f[i];b=f[j];mapping=[int(np.where(inv[b]==inv[x])[0][0]) for x in a];b=b[mapping]
  ta=v[f[i]];tb=v[f[j]];na=np.cross(ta[1]-ta[0],ta[2]-ta[0]);nb=np.cross(tb[1]-tb[0],tb[2]-tb[0]);dot=float(np.dot(na,nb)/(np.linalg.norm(na)*np.linalg.norm(nb)))
  rows.append(dict(faces=[i,j],normal_dot=dot,max_uv_difference=float(np.abs(uv[a]-uv[b]).max()),max_weight_difference=float(np.abs(weights[a]-weights[b]).max()),vertices_a=a.tolist(),vertices_b=b.tolist()))
r=dict(source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='Coincidence at exact bind position; attribute differences and winding compared at position-matched corners',pairs=rows)
out=Path('artifacts/kaggle-textured-rig-export-v14/duplicate-face-analysis.json');out.write_text(json.dumps(r,indent=2)+'\n');print('pairs',len(rows),'same normals',sum(x['normal_dot']>.99 for x in rows),'opposite',sum(x['normal_dot']<-.99 for x in rows),'different UV',sum(x['max_uv_difference']>1e-6 for x in rows),'different weights',sum(x['max_weight_difference']>1e-6 for x in rows),'max weight diff',max(x['max_weight_difference'] for x in rows))
