"""Acquire a tiny, pack-split Kenney GLB pilot from an immutable mirror revision."""
import hashlib,json
from pathlib import Path,PurePosixPath
from urllib.parse import quote
import requests

COMMIT='ea2f2aa1ca8da0898330c97df8a5489976b051d0'
BASE=f'https://raw.githubusercontent.com/Hidencod/tge-assets/{COMMIT}/'
SPLITS={'car-kit':'train','food-kit':'train','furniture-kit':'train','nature-kit':'train','space-kit':'validation','pirate-kit':'validation'}


def main():
 root=Path('artifacts/kenney-raw-pilot');root.mkdir(parents=True,exist_ok=True);session=requests.Session()
 response=session.get(BASE+'catalog.json',timeout=30);response.raise_for_status();raw=response.content;catalog=response.json();(root/'catalog.json').write_bytes(raw);records=[]
 packs={p['id']:p for p in catalog['packs']}
 for pack_id,split in SPLITS.items():
  pack=packs[pack_id]
  if pack['license']!='CC0-1.0' or pack['author']!='Kenney' or pack['url']!='https://kenney.nl/assets/'+pack_id:raise ValueError('Unexpected pack provenance')
  models=sorted(pack['models'],key=lambda m:m['id']);indices=[round((len(models)-1)*i/3) for i in range(4)]
  for index in indices:
   model=models[index];source=PurePosixPath(model['file'])
   if source.is_absolute() or '..' in source.parts or source.suffix!='.glb' or source.parts[:2]!=('packs',pack_id):raise ValueError('Unsafe model path')
   if not 12<=model['size']<=10_000_000:raise ValueError('Model outside size limit')
   url=BASE+quote(str(source),safe='/');response=session.get(url,timeout=30);response.raise_for_status();data=response.content
   if len(data)!=model['size'] or data[:4]!=b'glTF':raise ValueError('Unexpected GLB length or signature')
   destination=root/pack_id/source.name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
   records.append(dict(id=model['id'],name=model['name'],split=split,group=pack_id,source_url=url,creator='Kenney',creator_pack_url=pack['url'],license='CC0-1.0',path=str(destination.relative_to(root)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
   print(model['id'],len(data),flush=True)
 manifest=dict(scope='24-model raw acquisition pilot; pack-level split; no geometry filtering, deduplication or quality labels',mirror_revision=COMMIT,catalog_sha256=hashlib.sha256(raw).hexdigest(),records=records)
 (root/'acquisition.json').write_text(json.dumps(manifest,indent=2)+'\n');print('downloaded',len(records),flush=True)
if __name__=='__main__':main()
