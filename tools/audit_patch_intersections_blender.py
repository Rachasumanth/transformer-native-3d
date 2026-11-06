"""Broad-phase Blender BVH overlap check for prototype patch triangles."""
import json
from pathlib import Path
import numpy as np
from mathutils.bvhtree import BVHTree
root=Path('artifacts/kaggle-textured-rig-export-v14')
with np.load(root/'fan-projected-candidate-geometry.npz') as z:
 v=z['vertices'];f=z['faces'];start=int(z['patch_start'])
alltree=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True,epsilon=0)
patchtree=BVHTree.FromPolygons(v.tolist(),f[start:].tolist(),all_triangles=True,epsilon=0)
pairs=[]
for local,other in patchtree.overlap(alltree):
 pi=start+local
 if pi==other or set(f[pi]) & set(f[other]):continue
 pairs.append([int(pi),int(other)])
r={'scope':'Blender BVH triangle overlap excluding pairs sharing welded vertices; not a robust exhaustive self-intersection proof','patch_faces':int(len(f)-start),'pairs':pairs,'overlap_count':len(pairs)}
(root/'fan-patch-bvh-overlap.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
