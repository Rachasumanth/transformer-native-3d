"""Verify and merge five corpus shards, removing exact cross-shard duplicates."""
import hashlib,json,re,zipfile
from pathlib import Path
from t3d.dataset import AssetDataset

SOURCE='7506aaf2adde39e39ae751c00a748b7b5088d94e88ee2ab757db9ae83ff782a8'


def exact_deduplicate(records):
 records=sorted(records,key=lambda r:r['asset_id']);parents=list(range(len(records)));seen={}
 def find(i):
  while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
  return i
 for i,r in enumerate(records):
  for key in (('source',r['source_sha256']),('prepared',r['sha256'])):
   if key in seen:parents[find(i)]=find(seen[key])
   else:seen[key]=i
 groups={}
 for i in range(len(records)):groups.setdefault(find(i),[]).append(i)
 keep=set();duplicates=[]
 for ids in groups.values():
  splits={records[i]['split'] for i in ids};selected=ids[:1] if len(splits)==1 else [];keep.update(selected)
  for i in ids:
   if i not in selected:duplicates.append(dict(asset_id=records[i]['asset_id'],reason='cross-split exact duplicate' if len(splits)>1 else 'within-split exact duplicate'))
 return [r for i,r in enumerate(records) if i in keep],duplicates


def main():
 index_path=Path('artifacts/kenney-raw-pilot/candidate-index.json');index_raw=index_path.read_bytes();index=json.loads(index_raw);candidates={r['id']:r for r in index['candidates']};records=[];shard_failures=[];data={};seen_ids=set()
 for shard in range(5):
  root=Path(f'artifacts/kaggle-kenney-corpus-shard{shard}');report=json.loads((root/f'kenney-shard-{shard}-report.json').read_text());archive=root/f'kenney-shard-{shard}.zip'
  if report['source_sha256']!=SOURCE or report['index_sha256']!=hashlib.sha256(index_raw).hexdigest() or (report['start'],report['stop'],report['stride'])!=(shard,944,5):raise ValueError('Shard provenance mismatch')
  if len(report['accepted'])+len(report['rejected'])!=len(index['candidates'][shard:944:5]):raise ValueError('Incomplete shard accounting')
  shard_failures.extend(dict(shard=shard,**r) for r in report['rejected'])
  with zipfile.ZipFile(archive) as z:
   manifest=json.loads(z.read('manifest.json'));inside=json.loads(z.read('conversion-report.json'))
   if inside!= {k:v for k,v in report.items() if k!='source_sha256'}:raise ValueError('Inner and outer reports differ')
   if manifest['version']!=1 or len(manifest['assets'])!=len(report['accepted']):raise ValueError('Manifest/report count mismatch')
   for rec in manifest['assets']:
    name=rec['path'];asset_id=rec['asset_id']
    if not re.fullmatch(r'kenney-[a-z0-9-]+\.t3d',name) or name!=asset_id+'.t3d' or asset_id in seen_ids:raise ValueError('Invalid or repeated asset path')
    seen_ids.add(asset_id);source_id=asset_id[len('kenney-'):];matches=[r for r in candidates.values() if r['id'].replace('/','-')==source_id]
    if len(matches)!=1:raise ValueError('Unrecognized source id')
    origin=matches[0]
    if rec['source']!=origin['source_url'] or rec['split']!=origin['split'] or rec['source_group']!='kenney-'+origin['group']:raise ValueError('Source/split mismatch')
    metadata=Path('artifacts/kenney-catalog-download')/(origin['id']+'.json');meta=json.loads(metadata.read_text())
    if rec['source_sha256']!=meta['sha256']:raise ValueError('Raw source hash mismatch')
    blob=z.read(name)
    if hashlib.sha256(blob).hexdigest()!=rec['sha256']:raise ValueError('Prepared hash mismatch')
    records.append(rec);data[name]=blob
 selected,duplicates=exact_deduplicate(records)
 out=Path('artifacts/kenney-corpus-32');out.mkdir(parents=True,exist_ok=True)
 for r in selected:
  path=out/r['path'];blob=data[r['path']]
  if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Existing output differs')
  path.write_bytes(blob)
 (out/'manifest.json').write_text(json.dumps(dict(version=1,assets=selected),indent=2)+'\n')
 audit=dict(scope='Exact source/prepared dedup only; semantic dedup and quality filtering remain pending',index_sha256=hashlib.sha256(index_raw).hexdigest(),source_sha256=SOURCE,candidates=len(index['candidates']),accepted=len(selected),shard_rejections=shard_failures,cross_shard_duplicates=duplicates)
 (out/'merge-report.json').write_text(json.dumps(audit,indent=2)+'\n')
 for split in ('train','validation'):
  ds=AssetDataset(out/'manifest.json',split=split);print(split,len(ds),flush=True)
 print('rejections',len(shard_failures),'duplicates',len(duplicates),'manifest_sha256',hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),flush=True)
if __name__=='__main__':main()
