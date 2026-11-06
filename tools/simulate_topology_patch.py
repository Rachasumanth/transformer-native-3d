"""Prototype planar fan closures on one-ring defects; geometry only."""
import collections,json
from pathlib import Path
import numpy as np
from t3d.import_rig import load_rig_glb
from t3d.metrics import surface_comparison
root=Path('artifacts/kaggle-textured-rig-export-v14');rig=load_rig_glb(root/'duplicate-removal-candidate.glb');v=np.asarray(rig.vertices);f=np.asarray(rig.faces);u,inv=np.unique(v,axis=0,return_inverse=True);wf=inv[f];inc=collections.defaultdict(set)
for i,t in enumerate(wf):
 for a,b in zip(t,np.roll(t,-1)):inc[tuple(sorted((int(a),int(b))))].add(i)
bad=[ids for ids in inc.values() if len(ids)>2];region=set.union(*bad);affected=set(int(x) for i in region for x in wf[i]);region|={i for i,t in enumerate(wf) if any(int(x) in affected for x in t)}
kept=np.delete(wf,sorted(region),axis=0);counts=collections.defaultdict(list)
for i,t in enumerate(kept):
 for a,b in zip(t,np.roll(t,-1)):counts[tuple(sorted((int(a),int(b))))].append((int(a),int(b)))
boundary=[x[0] for x in counts.values() if len(x)==1];adj=collections.defaultdict(set)
for a,b in boundary:adj[a].add(b);adj[b].add(a)
if any(len(n)!=2 for n in adj.values()):raise ValueError('Boundary is not a set of loops')
seen=set();loops=[]
for start in adj:
 if start in seen:continue
 loop=[start];prev=None;cur=start
 while True:
  seen.add(cur);nxt=next(x for x in adj[cur] if x!=prev)
  if nxt==start:break
  if nxt in seen:raise ValueError('Loop intersection')
  loop.append(nxt);prev,cur=cur,nxt
 loops.append(loop)
newv=list(u);newf=[]
for loop in loops:
 center=len(newv);newv.append(np.mean(u[loop],axis=0))
 for i,a in enumerate(loop):
  b=loop[(i+1)%len(loop)];directed=counts[tuple(sorted((a,b)))][0]
  newf.append((b,a,center) if directed==(a,b) else (a,b,center))
newv=np.asarray(newv);newf=np.vstack((kept,np.asarray(newf)));source,_=load_rig_glb('artifacts/rigged-inputs/CesiumMan/CesiumMan.glb').normalized()
metric=surface_comparison(source.vertices,source.faces,newv,newf,samples=512,tolerance=.01)
from t3d.proximity import TriangleBVH
patch=newv[np.asarray(newf[-sum(map(len,loops)):])]; probes=np.concatenate((patch.reshape(-1,3),patch.mean(axis=1)),axis=0)
ids,bary=TriangleBVH(source.vertices,source.faces).query(probes)
nearest=np.einsum('ni,nij->nj',bary,np.asarray(source.vertices)[np.asarray(source.faces)[ids]])
patch_distances=np.linalg.norm(probes-nearest,axis=1)
r=dict(scope='Prototype one-ring fan closure in memory only; no UV, skin, normals or self-intersection guarantees',removed_faces=len(region),loop_lengths=[len(x) for x in loops],added_faces=sum(map(len,loops)),patch_probe_count=len(probes),patch_to_source_mean=float(patch_distances.mean()),patch_to_source_max=float(patch_distances.max()),surface_comparison=metric)
(root/'fan-patch-simulation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
