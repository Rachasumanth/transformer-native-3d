"""Inspect k-ring removal neighborhoods around position-welded nonmanifold edges."""
import collections,hashlib,json
from pathlib import Path
import numpy as np
from t3d.import_rig import load_rig_glb
p=Path('artifacts/kaggle-textured-rig-export-v14/duplicate-removal-candidate.glb');rig=load_rig_glb(p);v=np.asarray(rig.vertices);f=np.asarray(rig.faces);u,inv=np.unique(v,axis=0,return_inverse=True);wf=inv[f];inc=collections.defaultdict(set)
for i,t in enumerate(wf):
 for a,b in zip(t,np.roll(t,-1)):inc[tuple(sorted((int(a),int(b))))].add(i)
bad={e for e,x in inc.items() if len(x)>2};region=set.union(*(inc[e] for e in bad));rows=[]
for ring in range(5):
 removed=set(region);boundary=[];counts=collections.Counter()
 for edge,ids in inc.items():
  n=len(ids-removed)
  if n==1:boundary.append(edge)
  if n>2:counts['nonmanifold_edges']+=1
 degree=collections.Counter(x for edge in boundary for x in edge)
 seen=set();components=[];adj=collections.defaultdict(set)
 for a,b in boundary:adj[a].add(b);adj[b].add(a)
 for vertex in adj:
  if vertex in seen:continue
  stack=[vertex];seen.add(vertex);part=[]
  while stack:
   n=stack.pop();part.append(n)
   for q in adj[n]-seen:seen.add(q);stack.append(q)
  components.append(len(part))
 rows.append(dict(ring=ring,removed_faces=len(removed),remaining_nonmanifold_edges=counts['nonmanifold_edges'],boundary_edges=len(boundary),boundary_vertex_degrees=dict(collections.Counter(degree.values())),boundary_components=components))
 vertices=set(int(x) for i in region for x in wf[i]);region|={i for i,t in enumerate(wf) if any(int(x) in vertices for x in t)}
r={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'Exact-weld k-ring deletion diagnostics; no patch or GLB changes','rings':rows}
out=p.parent/'nonmanifold-patch-neighborhoods.json';out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(rows,indent=2))
