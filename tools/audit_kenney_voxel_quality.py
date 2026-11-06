"""Sample prepared voxelization fidelity against original normalized source meshes."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from t3d.container import load
from t3d.extraction import voxels_to_mesh
from t3d.import_glb import load_glb_geometry
from t3d.metrics import surface_comparison
from t3d.normalization import normalize_asset


def main():
 raw=Path('artifacts/kenney-raw-pilot');prepared=Path('artifacts/kenney-shape-pilot-32');source=json.loads((raw/'acquisition.json').read_text());manifest=json.loads((prepared/'manifest.json').read_text());by_id={'kenney-'+r['id'].replace('/','-'):r for r in source['records']};rows=[]
 for row in manifest['assets']:
  start=time.monotonic();origin=by_id[row['asset_id']];source_path=raw/origin['path'];data=source_path.read_bytes()
  if hashlib.sha256(data).hexdigest()!=row['source_sha256']:raise ValueError('Source hash mismatch')
  source_v,source_f,_=load_glb_geometry(source_path);source_v=np.asarray(normalize_asset(source_v)[0]);triangles=source_v[source_f];norm=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1);source_f=source_f[np.isfinite(norm)&(norm>0)]
  path=prepared/row['path']
  if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Prepared hash mismatch')
  asset=load(path).asset;vertices,faces=voxels_to_mesh(asset)
  result=surface_comparison(source_v,source_f,vertices,faces,samples=64,seed=91,tolerance=.01)
  rows.append(dict(asset_id=row['asset_id'],split=row['split'],group=row['source_group'],source_sha256=row['source_sha256'],prepared_sha256=row['sha256'],metrics=result,seconds=time.monotonic()-start));print(row['asset_id'],result.get('valid'),result.get('fscore'),flush=True)
 result=dict(scope='64 surface samples per asset vs original GLB; exploratory conversion fidelity screening, not a VLM filter or validated quality benchmark',manifest_sha256=hashlib.sha256((prepared/'manifest.json').read_bytes()).hexdigest(),rows=rows)
 (prepared/'voxel-quality-audit.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
