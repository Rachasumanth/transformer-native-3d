"""Prepare a small, provenance-bearing shape-only dataset from existing assets."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from t3d.container import dumps,loads
from t3d.import_rig import load_rig_glb
from t3d.voxelize import mesh_to_voxels


def main():
 root=Path('artifacts/mixed-shape-pilot-32');root.mkdir(parents=True,exist_ok=True)
 previous=Path('artifacts/pilot-dataset-32');old=json.loads((previous/'manifest.json').read_text());records=[];conversion=[]
 for row in old['assets']:
  raw=(previous/row['path']).read_bytes()
  if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Prepared asset hash mismatch')
  (root/row['path']).write_bytes(raw);row['source_group']=row['asset_id'];records.append(row)
 entries=json.loads(Path('artifacts/prompt-occupancy-pilot/manifest.json').read_text())['assets']
 for row in entries:
  name=row['name'];source=Path('artifacts/rigged-candidates')/name/(name+'.glb');raw=source.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=row['source_sha256']:raise ValueError('Original source hash mismatch')
  start=time.monotonic();rig,transform=load_rig_glb(source).normalized();vertices=np.clip(rig.vertices,-.5,.5)
  asset=mesh_to_voxels(vertices,rig.faces,grid_size=(32,32,32),aabb=[[-.5]*3,[.5]*3]);data=dumps(asset);digest=hashlib.sha256(data).hexdigest();out=name.lower()+'.t3d';(root/out).write_bytes(data)
  records.append(dict(asset_id=name.lower(),path=out,sha256=digest,split=row['split'],source=row['source'],license=row['license']+('; '+row['attribution'] if row.get('attribution') else ''),source_sha256=row['source_sha256'],source_group='gobkit-animal' if row['split']=='train' else 'khronos-riggedfigure'))
  conversion.append(dict(asset=name,voxels=len(asset.coords),faces=len(rig.faces),prepared_sha256=digest,source_sha256=row['source_sha256'],center=transform.center,scale=transform.scale,seconds=time.monotonic()-start));print(name,len(asset.coords),round(conversion[-1]['seconds'],2),flush=True)
 (root/'manifest.json').write_text(json.dumps(dict(version=1,assets=records),indent=2)+'\n')
 (root/'conversion-report.json').write_text(json.dumps(dict(scope='Small shape-only pilot; rigs and materials omitted; prior inspected assets, not a curated corpus',grid=32,assets=conversion),indent=2)+'\n')
if __name__=='__main__':main()
