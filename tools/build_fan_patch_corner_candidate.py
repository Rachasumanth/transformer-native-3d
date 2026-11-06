"""Build a research-only GLB from the saved one-ring fan prototype.

Patch corners sample UVs and skin weights from a single nearest original
triangle per patch face. This can smear textures at seams; inspect results.
"""
import hashlib,json,struct
from pathlib import Path
import numpy as np
from t3d.glb import _glb_chunks,limit_skin_influences
from t3d.gltf_accessors import GLBAccessors
from t3d.import_rig import load_rig_glb
from t3d.proximity import TriangleBVH
from t3d.materials import closest_surface
root=Path('artifacts/kaggle-textured-rig-export-v14');source=root/'duplicate-removal-candidate.glb';out=root/'fan-patch-corner-uv-candidate.glb'
expected=json.loads((root/'nonmanifold-patch-neighborhoods.json').read_text())['source_sha256']
if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise ValueError('Patch input does not match analyzed source')
access=GLBAccessors(source);rig=load_rig_glb(source);original_f=np.asarray(rig.faces);original_v=np.asarray(rig.vertices)
with np.load(root/'fan-projected-candidate-geometry.npz') as z:v=z['vertices'];faces=z['faces'];start=int(z['patch_start'])
unique,inv=np.unique(original_v,axis=0,return_inverse=True);welded=inv[original_f];kept=faces[:start];keep=[];j=0
for i,tri in enumerate(welded):
 if j<len(kept) and np.array_equal(tri,kept[j]):keep.append(i);j+=1
if j!=len(kept):raise ValueError('Prototype geometry does not match input mesh')
patch=v[faces[start:]];bvh=TriangleBVH(original_v,original_f);corners=patch.reshape(-1,3);source_face_ids,corner_bary=bvh.query(corners);bary=[]
for point,fi in zip(corners,source_face_ids):
 _,b=closest_surface(point[None,:],original_v,original_f[fi:fi+1]);bary.append(b[0])
bary=np.array(bary);source_triangles=original_f[source_face_ids]
primitive=access.document['meshes'][0]['primitives'][0];attrs=primitive['attributes'];old={key:access.read(index) for key,index in attrs.items()}
new={key:value.copy() for key,value in old.items()};patch_uv=np.einsum('ni,nij->nj',bary,old['TEXCOORD_0'][source_triangles]);dense=np.einsum('ni,nij->nj',bary,rig.weights[source_triangles]);dense,_=limit_skin_influences(dense,max_influences=16)
order=np.argsort(-dense,axis=1,kind='stable')[:,:16];ws=np.take_along_axis(dense,order,axis=1);order[ws==0]=0
normal=np.cross(patch[:,1]-patch[:,0],patch[:,2]-patch[:,0]);norm=np.linalg.norm(normal,axis=1)
if np.any(norm<1e-12):raise ValueError('Degenerate patch triangle')
normal=np.repeat(normal/norm[:,None],3,axis=0)
# Preserve exact POSITION and skin attributes at every repeated boundary position.
patch_ids=faces[start:].reshape(-1); representative=np.full(len(unique),-1,dtype=np.int64)
for index,wi in enumerate(inv):
 if representative[wi]<0:representative[wi]=index
for index,wi in enumerate(patch_ids):
 if wi<len(unique):corners[index]=old['POSITION'][representative[wi]]
new['POSITION']=np.vstack((old['POSITION'],corners));new['TEXCOORD_0']=np.vstack((old['TEXCOORD_0'],patch_uv));new['NORMAL']=np.vstack((old['NORMAL'],normal))
for group in range(4):
 new[f'JOINTS_{group}']=np.vstack((old[f'JOINTS_{group}'],order[:,4*group:4*group+4]));new[f'WEIGHTS_{group}']=np.vstack((old[f'WEIGHTS_{group}'],ws[:,4*group:4*group+4]))
# Reuse the same skin encoding on all copies of a welded patch vertex.
for wi in np.unique(patch_ids):
 slots=np.flatnonzero(patch_ids==wi)
 for group in range(4):
  for prefix in ('JOINTS','WEIGHTS'):
   key=f'{prefix}_{group}'; values=new[key]; base=len(original_v)
   if wi<len(unique):values[base+slots]=old[key][representative[wi]]
   else:values[base+slots]=values[base+slots[0]]
indices=np.vstack((original_f[keep],np.arange(len(original_v),len(original_v)+len(corners)).reshape(-1,3))).reshape(-1)
chunks=_glb_chunks(source.read_bytes());doc=json.loads(chunks[0][1]);binary=bytearray(chunks[1][1]);docprim=doc['meshes'][0]['primitives'][0]
def append(values,component,kind,target):
 while len(binary)%4:binary.append(0)
 dtype={5121:'u1',5123:'<u2',5125:'<u4',5126:'<f4'}[component];arr=np.ascontiguousarray(values,dtype=dtype);offset=len(binary);binary.extend(arr.tobytes());view=len(doc['bufferViews']);doc['bufferViews'].append(dict(buffer=0,byteOffset=offset,byteLength=arr.nbytes,target=target));aid=len(doc['accessors']);a=dict(bufferView=view,componentType=component,count=len(arr),type=kind)
 if kind in ('SCALAR','VEC3') and (kind=='SCALAR' or target==34962):a.update(min=np.min(arr,axis=0).reshape(-1).tolist(),max=np.max(arr,axis=0).reshape(-1).tolist())
 doc['accessors'].append(a);return aid
for key,values in new.items():
 oldspec=doc['accessors'][attrs[key]];docprim['attributes'][key]=append(values,oldspec['componentType'],oldspec['type'],34962)
docprim['indices']=append(indices,5125,'SCALAR',34963);doc['buffers'][0]['byteLength']=len(binary)
while len(binary)%4:binary.append(0)
j=json.dumps(doc,separators=(',',':')).encode();j+=b' '*((-len(j))%4);total=12+8+len(j)+8+len(binary)
out.write_bytes(struct.pack('<4sII',b'glTF',2,total)+struct.pack('<I4s',len(j),b'JSON')+j+struct.pack('<I4s',len(binary),b'BIN\0')+binary)
result=load_rig_glb(out);print('faces',len(result.faces),'vertices',len(result.vertices),'sha256',hashlib.sha256(out.read_bytes()).hexdigest())
