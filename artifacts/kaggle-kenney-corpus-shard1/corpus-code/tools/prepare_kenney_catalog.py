"""Convert a bounded shard of pinned raw catalog GLBs into shape-only voxels."""
import argparse,concurrent.futures,hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
from t3d.container import dumps
from t3d.extraction import voxels_to_mesh
from t3d.import_glb import load_glb_geometry
from t3d.normalization import normalize_asset
from t3d.voxelize import mesh_to_voxels


def convert(task):
 row,root=task;start=time.monotonic();pack,name=row['id'].split('/',1);source=Path(root)/pack/(name+'.glb');metadata=source.with_suffix('.json');meta=json.loads(metadata.read_text());data=source.read_bytes()
 if meta['id']!=row['id'] or meta['source_url']!=row['source_url'] or hashlib.sha256(data).hexdigest()!=meta['sha256'] or len(data)!=row['expected_bytes']:raise ValueError('Raw provenance mismatch')
 vertices,faces,digest=load_glb_geometry(source)
 if digest!=meta['sha256']:raise ValueError('Importer source mismatch')
 vertices=np.clip(np.asarray(normalize_asset(vertices)[0]),-.5,.5);tri=vertices[faces];cross=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1);grid=(vertices+.5)[faces]*32;grid_cross=np.linalg.norm(np.cross(grid[:,1]-grid[:,0],grid[:,2]-grid[:,0]),axis=1)
 if not np.isfinite(cross).all() or not np.isfinite(grid_cross).all():raise ValueError('Nonfinite geometry')
 cleaned=faces[(cross>0)&(grid_cross>0)];removed=len(faces)-len(cleaned)
 if not len(cleaned):raise ValueError('No nondegenerate source faces')
 asset=mesh_to_voxels(vertices,cleaned,grid_size=(32,32,32),aabb=[[-.5]*3,[.5]*3]);_,mesh_faces=voxels_to_mesh(asset)
 if not len(asset.coords) or not len(mesh_faces):raise ValueError('No extractable surface')
 content=dumps(asset);asset_id='kenney-'+row['id'].replace('/','-');name_out=asset_id+'.t3d'
 record=dict(asset_id=asset_id,path=name_out,sha256=hashlib.sha256(content).hexdigest(),split=row['split'],source=row['source_url'],license='CC0-1.0; Kenney; '+row['creator_pack_url'],source_sha256=digest,source_group='kenney-'+row['group'])
 stats=dict(asset_id=asset_id,source_faces=len(faces),removed_zero_or_grid_collapsed=removed,active_voxels=len(asset.coords),triangles=len(mesh_faces),seconds=time.monotonic()-start)
 return record,stats,content


def main():
 p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,required=True);p.add_argument('--stride',type=int,default=1);p.add_argument('--workers',type=int,default=2);args=p.parse_args()
 if args.start<0 or args.stop<=args.start or args.stride<=0 or not 1<=args.workers<=4:raise ValueError('Invalid range or worker count')
 index_raw=args.index.read_bytes();index=json.loads(index_raw);candidates=index['candidates'][args.start:args.stop:args.stride]
 if not candidates:raise ValueError('Empty candidate shard')
 args.output.mkdir(parents=True,exist_ok=True);success=[];failures=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
  futures={pool.submit(convert,(row,str(args.raw_root))):row for row in candidates}
  for future in concurrent.futures.as_completed(futures):
   row=futures[future]
   try:
    record,stats,data=future.result();(args.output/record['path']).write_bytes(data);success.append((record,stats))
   except Exception as exc:
    failures.append(dict(id=row['id'],error_type=type(exc).__name__,reason=str(exc)[:500]))
   total=len(success)+len(failures)
   if total%20==0:print(json.dumps(dict(processed=total,total=len(candidates),accepted=len(success),rejected=len(failures))),flush=True)
 success.sort(key=lambda pair:pair[0]['asset_id'])
 # Group exact source or prepared-content duplicates, including transitive links.
 parents=list(range(len(success)))
 def find(i):
  while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
  return i
 seen={}
 for i,(record,_) in enumerate(success):
  for key in (('source',record['source_sha256']),('prepared',record['sha256'])):
   if key in seen:parents[find(i)]=find(seen[key])
   else:seen[key]=i
 groups={}
 for i in range(len(success)):groups.setdefault(find(i),[]).append(i)
 keep=set()
 for indexes in groups.values():
  splits={success[i][0]['split'] for i in indexes}
  selected=indexes[:1] if len(splits)==1 else []
  keep.update(selected)
  for i in indexes:
   if i not in selected:failures.append(dict(id=success[i][0]['asset_id'],error_type='ExactDuplicate',reason='Across split boundary' if len(splits)>1 else 'Repeated within split'))
 records=[x[0] for i,x in enumerate(success) if i in keep];stats=[x[1] for i,x in enumerate(success) if i in keep]
 report=dict(scope='Uncurated shape-only shard; only within-shard exact duplicates checked; cross-shard and semantic deduplication and quality filtering remain pending',index_sha256=hashlib.sha256(index_raw).hexdigest(),start=args.start,stop=args.stop,stride=args.stride,accepted=stats,rejected=sorted(failures,key=lambda x:x['id']))
 (args.output/'manifest.json').write_text(json.dumps(dict(version=1,assets=records),indent=2)+'\n');(args.output/'conversion-report.json').write_text(json.dumps(report,indent=2)+'\n')
 archive=args.output.parent/(args.output.name+'.zip')
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for path in [args.output/'manifest.json',args.output/'conversion-report.json',*(args.output/r['path'] for r in records)]:z.write(path,arcname=path.name)
 print(json.dumps(dict(complete=True,accepted=len(records),rejected=len(failures),archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())),flush=True)
if __name__=='__main__':main()
