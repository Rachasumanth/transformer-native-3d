"""Project source skin weights onto the v13 animated reconstruction as a control."""
import json
from pathlib import Path
import numpy as np
from t3d.glb import rewrite_skin_weights
from t3d.import_rig import load_rig_glb
from t3d.proximity import TriangleBVH

root=Path('artifacts/posed-deformation')
source=Path('artifacts/rigged-inputs/CesiumMan/CesiumMan.glb')
output=root/'cesiumman-v13-animated.glb'
source_rig,_=load_rig_glb(source).normalized()
output_rig,_=load_rig_glb(output).normalized()
report=json.loads((root/'cesiumman-v13-posed-deformation.json').read_text())
output_to_source=np.array(report['joint_mapping']['output_to_source'])
tri,bary=TriangleBVH(source_rig.vertices,source_rig.faces).query(output_rig.vertices)
weights=np.einsum('ni,nij->nj',bary,source_rig.weights[source_rig.faces[tri]])[:,output_to_source]
data,audit=rewrite_skin_weights(output.read_bytes(),weights)
(root/'cesiumman-v13-oracle-animated.glb').write_bytes(data)
(root/'cesiumman-v13-oracle-weights.json').write_text(json.dumps({'scope':'source-projected oracle control; requires source weights','influence_audit':audit},indent=2)+'\n')
