"""Add normalized bind skeletons to family-held-out pose data."""
import hashlib,json
from pathlib import Path
import numpy as np
from t3d.import_rig import load_rig_glb
src=Path('artifacts/skin-family-vertex-pilot');out=Path('artifacts/skin-family-bind-pilot');out.mkdir(exist_ok=True)
m=json.loads((src/'manifest.json').read_text());rows=[]
for row in m['assets']:
 p=src/row['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 source=Path('artifacts/rigged-inputs')/row['name']/(row['name']+'.glb');assert hashlib.sha256(source.read_bytes()).hexdigest()==row['source_sha256']
 rig,_=load_rig_glb(source).normalized()
 with np.load(p,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
 arrays['bind_joints']=np.asarray(rig.skeleton.joints,dtype=np.float32)
 dest=out/p.name;np.savez_compressed(dest,**arrays);rows.append({**row,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
m['assets']=rows;m['scope']+='; normalized bind joints added for bind-conditioned scoring';(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print('Prepared bind skeletons for',len(rows),'assets')
