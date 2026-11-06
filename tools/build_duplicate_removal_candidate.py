"""Apply a previously inspected face-removal simulation to a separate GLB."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
from t3d.glb import _glb_chunks
from t3d.gltf_accessors import GLBAccessors
root=Path('artifacts/kaggle-textured-rig-export-v14');source=root/'export-real-rigged.glb';out=root/'duplicate-removal-candidate.glb'
analysis=json.loads((root/'duplicate-face-analysis.json').read_text());simulation=json.loads((root/'duplicate-removal-simulation.json').read_text())
if hashlib.sha256(source.read_bytes()).hexdigest()!=analysis['sha256']:raise ValueError('Simulation source hash mismatch')
chunks=_glb_chunks(source.read_bytes());doc=json.loads(chunks[0][1]);binary=bytearray(chunks[1][1]);access=GLBAccessors(source);prim=doc['meshes'][0]['primitives'][0];aid=prim['indices'];old=doc['accessors'][aid]
indices=access.read(aid).reshape(-1,3);remove=simulation['removed_face_ids'];kept=np.delete(indices,remove,axis=0).reshape(-1)
if old['componentType'] not in (5121,5123,5125):raise ValueError('Unexpected index type')
dtype={5121:'u1',5123:'<u2',5125:'<u4'}[old['componentType']];view=doc['bufferViews'][old['bufferView']];offset=view.get('byteOffset',0)+old.get('byteOffset',0);payload=kept.astype(dtype).tobytes();binary[offset:offset+len(payload)]=payload
old['count']=int(len(kept));old['min']=[int(kept.min())];old['max']=[int(kept.max())]
j=json.dumps(doc,separators=(',',':')).encode();j+=b' '*((-len(j))%4);binary+=b'\0'*((-len(binary))%4);total=12+8+len(j)+8+len(binary)
out.write_bytes(struct.pack('<4sII',b'glTF',2,total)+struct.pack('<I4s',len(j),b'JSON')+j+struct.pack('<I4s',len(binary),b'BIN\0')+binary)
restored=GLBAccessors(out);new=restored.read(restored.document['meshes'][0]['primitives'][0]['indices']).reshape(-1,3);np.testing.assert_array_equal(new,np.delete(indices,remove,axis=0));print('candidate',out,'faces',len(new),'sha256',hashlib.sha256(out.read_bytes()).hexdigest())
