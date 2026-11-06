"""Resumable, bounded acquisition of pinned catalog candidates."""
import concurrent.futures,hashlib,json,os,re,threading,time
from pathlib import Path
import requests


def main():
 index_path=Path('artifacts/kenney-raw-pilot/candidate-index.json');index=json.loads(index_path.read_text());root=Path('artifacts/kenney-catalog-download');root.mkdir(parents=True,exist_ok=True);local=threading.local();pilot=Path('artifacts/kenney-raw-pilot');counts={'downloaded':0,'reused':0,'failed':0};lock=threading.Lock()
 def fetch(row):
  if not re.fullmatch(r'[a-z0-9-]+/[a-z0-9-]+',row['id']):raise ValueError('Unsafe candidate ID')
  pack,name=row['id'].split('/',1);expected_prefix='https://raw.githubusercontent.com/Hidencod/tge-assets/'+index['revision']+'/packs/'+pack+'/'
  if not row['source_url'].startswith(expected_prefix) or row['source_url']!=expected_prefix+name+'.glb':raise ValueError('Unexpected source URL')
  path=root/pack/(name+'.glb');meta=path.with_suffix('.json');path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists() and meta.exists():
   old=json.loads(meta.read_text());raw=path.read_bytes()
   if old['source_url']==row['source_url'] and len(raw)==row['expected_bytes'] and hashlib.sha256(raw).hexdigest()==old['sha256']:return dict(id=row['id'],status='reused',sha256=old['sha256'])
  cached=pilot/pack/(name+'.glb');data=None
  if row['downloaded_sha256'] and cached.exists():
   candidate=cached.read_bytes()
   if hashlib.sha256(candidate).hexdigest()==row['downloaded_sha256']:data=candidate
  if data is None:
   if not hasattr(local,'session'):local.session=requests.Session()
   for attempt in range(3):
    try:
     response=local.session.get(row['source_url'],timeout=30);response.raise_for_status();data=response.content;break
    except requests.RequestException:
     if attempt==2:raise
     time.sleep(1+attempt)
  if len(data)!=row['expected_bytes'] or data[:4]!=b'glTF':raise ValueError('Unexpected size or GLB signature')
  digest=hashlib.sha256(data).hexdigest();temporary=path.with_suffix('.glb.tmp');temporary.write_bytes(data);os.replace(temporary,path)
  metadata=dict(id=row['id'],source_url=row['source_url'],sha256=digest,bytes=len(data),group=row['group'],split=row['split'],license=row['license'],creator_pack_url=row['creator_pack_url'])
  temporary=meta.with_suffix('.json.tmp');temporary.write_text(json.dumps(metadata,indent=2)+'\n');os.replace(temporary,meta);time.sleep(.05)
  return dict(id=row['id'],status='downloaded',sha256=digest)
 results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(fetch,row):row for row in index['candidates']}
  for future in concurrent.futures.as_completed(futures):
   row=futures[future]
   try:result=future.result()
   except (requests.RequestException,ValueError,OSError,KeyError) as exc:result=dict(id=row['id'],status='failed',error=str(exc)[:300])
   results.append(result);counts[result['status']]+=1
   if len(results)%50==0:print('processed',len(results),'of',len(futures),'failed',counts['failed'],flush=True)
 results.sort(key=lambda x:x['id']);report=dict(scope='Pinned raw downloads, not parsed, quality-filtered, deduplicated or training-ready',index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),revision=index['revision'],counts=counts,results=results)
 (root/'acquisition-report.json').write_text(json.dumps(report,indent=2)+'\n');print('complete',counts,flush=True)
if __name__=='__main__':main()
