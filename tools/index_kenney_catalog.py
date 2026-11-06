"""Index candidate Kenney models without downloading or approving their geometry."""
import hashlib,json
from pathlib import Path,PurePosixPath
from urllib.parse import quote

PACK_SPLITS={'car-kit':'train','food-kit':'train','furniture-kit':'train','nature-kit':'train','space-kit':'validation','pirate-kit':'validation'}


def main():
 root=Path('artifacts/kenney-raw-pilot');acquisition=json.loads((root/'acquisition.json').read_text());catalog_raw=(root/'catalog.json').read_bytes()
 if hashlib.sha256(catalog_raw).hexdigest()!=acquisition['catalog_sha256']:raise ValueError('Catalog hash mismatch')
 catalog=json.loads(catalog_raw);revision=acquisition['mirror_revision'];base=f'https://raw.githubusercontent.com/Hidencod/tge-assets/{revision}/';selected={r['id']:r for r in acquisition['records']};rows=[];ids=set();paths=set()
 for pack in catalog['packs']:
  if pack['id'] not in PACK_SPLITS:continue
  if pack['license']!='CC0-1.0' or pack['author']!='Kenney' or pack['url']!='https://kenney.nl/assets/'+pack['id']:raise ValueError('Unexpected provenance')
  for model in pack['models']:
   path=PurePosixPath(model['file']);model_id=model['id']
   if path.is_absolute() or '..' in path.parts or path.suffix!='.glb' or path.parts[:2]!=('packs',pack['id']) or model_id != pack['id']+'/'+path.stem or len(path.parts)!=3 or model_id in ids or str(path) in paths:raise ValueError('Invalid or duplicated catalog entry')
   ids.add(model_id);paths.add(str(path));saved=selected.get(model_id)
   rows.append(dict(id=model_id,name=model['name'],group=pack['id'],split=PACK_SPLITS[pack['id']],source_url=base+quote(str(path),safe='/'),creator_pack_url=pack['url'],license=pack['license'],expected_bytes=model['size'],downloaded_sha256=saved['sha256'] if saved else None))
 rows.sort(key=lambda x:x['id'])
 out=dict(scope='Catalog index only: not downloaded, quality-filtered, deduplicated, licensed independently, or training-ready',revision=revision,catalog_sha256=acquisition['catalog_sha256'],candidates=rows)
 target=root/'candidate-index.json';target.write_text(json.dumps(out,indent=2)+'\n');print('candidates',len(rows),'already downloaded',sum(r['downloaded_sha256'] is not None for r in rows),'expected bytes',sum(r['expected_bytes'] for r in rows))
if __name__=='__main__':main()
