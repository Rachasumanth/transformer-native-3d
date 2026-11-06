"""Measure a small pilot's reconstructed surfaces against original source GLBs."""
import hashlib,json
from pathlib import Path
import numpy as np
from t3d.container import load
from t3d.extraction import voxels_to_mesh
from t3d.import_glb import load_glb
from t3d.metrics import surface_comparison
from t3d.normalization import normalize_asset


def main():
 root=Path.cwd();manifest=json.loads((root/'artifacts/pilot-dataset-32/manifest.json').read_text());results={}
 for row in manifest['assets']:
  split=row['split'];name='avocado' if split=='train' else 'waterbottle';filename='Avocado.glb' if split=='train' else 'WaterBottle.glb';source=load_glb(root/'artifacts/import'/name/filename)
  if source.source_sha256!=row['source_sha256']:raise ValueError('Original GLB source hash mismatch')
  vertices,_,_=normalize_asset(source.vertices);paths={'prepared':root/'artifacts/pilot-dataset-32'/row['path'],'predicted':root/'artifacts/kaggle-triton-occupancy-32-v30'/(split+'-predicted.t3d')};cases={}
  for label,path in paths.items():
   voxels=load(path).asset
   if not np.allclose(voxels.aabb,[[-.5]*3,[.5]*3],rtol=0,atol=1e-6):raise ValueError('Incompatible normalization')
   verts,faces=voxels_to_mesh(voxels);runs=[surface_comparison(vertices,source.faces,verts,faces,samples=128,seed=seed,tolerance=.01) for seed in (0,1,2)]
   cases[label]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),measurements=runs)
   print(split,label,[(round(m.get('mean_distance',0),4),round(m.get('fscore',0),4)) for m in runs],flush=True)
  results[split]=dict(source_sha256=source.source_sha256,source_faces=len(source.faces),cases=cases)
 out=dict(scope='Three small sampled surface comparisons against normalized original GLBs; two previously inspected assets; not a research benchmark',assets=results)
 (root/'artifacts/kaggle-triton-occupancy-32-v30/original-surface-audit.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
