"""Convert the pinned raw Kenney pilot to shape-only 32-cubed .t3d assets."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from t3d.container import dumps
from t3d.extraction import voxels_to_mesh
from t3d.import_glb import load_glb_geometry
from t3d.normalization import normalize_asset
from t3d.voxelize import mesh_to_voxels


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--retry-rejected',action='store_true');args=parser.parse_args()
 src=Path('artifacts/kenney-raw-pilot');out=Path('artifacts/kenney-shape-pilot-32');out.mkdir(parents=True,exist_ok=True);source_bytes=(src/'acquisition.json').read_bytes();manifest=json.loads(source_bytes);accepted=[];rejected=[];conversion=[];seen={};retry=None
 if args.retry_rejected:
  previous=json.loads((out/'conversion-report.json').read_text())
  if previous['source_manifest_sha256']!=hashlib.sha256(source_bytes).hexdigest():raise ValueError('Source acquisition changed')
  accepted=json.loads((out/'manifest.json').read_text())['assets'];conversion=previous['accepted'];retry={r['id'] for r in previous['rejected']}
  for old in accepted:
   target=out/old['path']
   if hashlib.sha256(target.read_bytes()).hexdigest()!=old['sha256']:raise ValueError('Existing prepared asset changed')
   seen[old['sha256']]=old['asset_id']
  for old in conversion:old.setdefault('cleanup_policy','normalized_zero_area')
 for row in manifest['records']:
  if retry is not None and row['id'] not in retry:continue
  start=time.monotonic()
  try:
   p=(src/row['path']).resolve();root=src.resolve()
   if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Source hash or path mismatch')
   vertices,source_faces,source_sha=load_glb_geometry(p)
   if source_sha!=row['sha256']:raise ValueError('Importer source hash mismatch')
   vertices,_,transform=normalize_asset(vertices);vertices=np.clip(np.asarray(vertices),-.5,.5)
   triangles=vertices[source_faces];areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
   grid_triangles=(vertices+.5)[source_faces]*32
   grid_areas=np.linalg.norm(np.cross(grid_triangles[:,1]-grid_triangles[:,0],grid_triangles[:,2]-grid_triangles[:,0]),axis=1)
   if not np.isfinite(areas).all() or not np.isfinite(grid_areas).all():raise ValueError('Nonfinite face geometry')
   clean_faces=source_faces[(areas>0)&(grid_areas>0)];removed=len(source_faces)-len(clean_faces)
   if not len(clean_faces):raise ValueError('All source faces are degenerate')
   voxels=mesh_to_voxels(vertices,clean_faces,grid_size=(32,32,32),aabb=[[-.5]*3,[.5]*3]);_,faces=voxels_to_mesh(voxels)
   if not len(voxels.coords) or not len(faces):raise ValueError('No extractable surface')
   data=dumps(voxels);digest=hashlib.sha256(data).hexdigest()
   if digest in seen:raise ValueError('Duplicate prepared output of '+seen[digest])
   seen[digest]=row['id'];asset_id='kenney-'+row['id'].replace('/','-');target=asset_id+'.t3d';(out/target).write_bytes(data)
   accepted.append(dict(asset_id=asset_id,path=target,sha256=digest,split=row['split'],source=row['source_url'],license='CC0-1.0; Kenney; '+row['creator_pack_url'],source_sha256=row['sha256'],source_group='kenney-'+row['group']))
   conversion.append(dict(cleanup_policy='normalized_and_grid_zero_area',asset_id=asset_id,voxels=len(voxels.coords),source_faces=len(source_faces),removed_zero_area_or_grid_collapsed_faces=removed,extracted_triangles=len(faces),center=transform.center,scale=transform.scale,seconds=time.monotonic()-start));print('accepted',asset_id,len(voxels.coords),flush=True)
  except (ValueError,RuntimeError,IndexError,KeyError,NotImplementedError) as exc:
   rejected.append(dict(id=row['id'],reason=str(exc)));print('rejected',row['id'],str(exc),flush=True)
 (out/'manifest.json').write_text(json.dumps(dict(version=1,assets=accepted),indent=2)+'\n')
 (out/'conversion-report.json').write_text(json.dumps(dict(scope='Shape-only conversion pilot; not curated, semantically deduplicated, or validated for quality',source_manifest_sha256=hashlib.sha256((src/'acquisition.json').read_bytes()).hexdigest(),grid=32,accepted=conversion,rejected=rejected),indent=2)+'\n')
 print('complete',len(accepted),'accepted',len(rejected),'rejected',flush=True)
if __name__=='__main__':main()
