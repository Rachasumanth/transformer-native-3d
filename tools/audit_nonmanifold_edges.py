"""Classify exact-position-welded nonmanifold edges in a skinned GLB."""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from t3d.import_rig import load_rig_glb

def main():
 p=argparse.ArgumentParser();p.add_argument('glb',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 rig=load_rig_glb(a.glb);v=np.asarray(rig.vertices);f=np.asarray(rig.faces);unique,inv=np.unique(v,axis=0,return_inverse=True);wf=inv[f]
 incidence=collections.defaultdict(list)
 for i,tri in enumerate(wf):
  for x,y in zip(tri,np.roll(tri,-1)):incidence[tuple(sorted((int(x),int(y))))].append(i)
 duplicate=collections.Counter(tuple(sorted(map(int,t))) for t in wf)
 rows=[]
 for (x,y),ids in incidence.items():
  if len(ids)<=2:continue
  actual=collections.Counter(tuple(sorted((int(f[i,j]),int(f[i,(j+1)%3])))) for i in ids for j in range(3) if tuple(sorted((int(wf[i,j]),int(wf[i,(j+1)%3]))))==(x,y))
  triangles=v[f[ids]];norm=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);norm/=np.linalg.norm(norm,axis=1)[:,None]
  rows.append(dict(endpoints=[unique[x].tolist(),unique[y].tolist()],length=float(np.linalg.norm(unique[x]-unique[y])),face_ids=ids,incidence=len(ids),distinct_unwelded_edges=len(actual),duplicate_face_incidents=sum(duplicate[tuple(sorted(map(int,wf[i])))]>1 for i in ids),normal_dot_products=np.round((norm@norm.T),5).tolist()))
 r=dict(source=str(a.glb),sha256=hashlib.sha256(a.glb.read_bytes()).hexdigest(),scope='Exact positional weld; coincident disconnected surfaces can appear nonmanifold',vertices=len(v),welded_vertices=len(unique),faces=len(f),duplicate_welded_faces=sum(n-1 for n in duplicate.values()),nonmanifold_edges=len(rows),edges=rows)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ('source','vertices','welded_vertices','faces','duplicate_welded_faces','nonmanifold_edges')}))
if __name__=='__main__':main()
