"""Target PBR accuracy on newly added patch triangles, not the entire mesh."""
import json,hashlib
from pathlib import Path
import numpy as np
from t3d.import_rig import load_rig_glb
from t3d.materials import closest_surface
from tools.audit_material_fidelity import _sample_materials,_metrics,CHANNELS
root=Path('artifacts/kaggle-textured-rig-export-v14');source_path=Path('artifacts/rigged-inputs/CesiumMan/CesiumMan.glb');candidate=root/'fan-patch-corner-uv-candidate.glb'
source,_=load_rig_glb(source_path,include_pbr=True).normalized();out=load_rig_glb(candidate,include_pbr=True)
tri=np.asarray(out.vertices)[out.faces[-28:]];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1);rng=np.random.default_rng(0);ids=rng.choice(28,4096,p=area/area.sum());uv=rng.random((4096,2));q=np.sqrt(uv[:,0]);bary=np.column_stack((1-q,q*(1-uv[:,1]),q*uv[:,1]));points=np.einsum('ni,nij->nj',bary,tri[ids]);ids+=len(out.faces)-28
source_ids,source_bary=closest_surface(points,source.vertices,source.faces,method='bvh');reference=_sample_materials(points,source_ids,source_bary,source.pbr_materials,source.face_materials,source.corner_uv);pred=_sample_materials(points,ids,bary,out.pbr_materials,out.face_materials,out.corner_uv)
baseline_path=root/'duplicate-removal-candidate.glb';baseline=load_rig_glb(baseline_path,include_pbr=True)
base_ids,base_bary=closest_surface(points,baseline.vertices,baseline.faces,method='bvh');base_values=_sample_materials(points,base_ids,base_bary,baseline.pbr_materials,baseline.face_materials,baseline.corner_uv)
r={'baseline_sha256':hashlib.sha256(baseline_path.read_bytes()).hexdigest(),'baseline_linear_pbr':{name:_metrics(reference[:,i],base_values[:,i]) for i,name in enumerate(CHANNELS)},'scope':'4096 area-weighted samples restricted to 28 patch triangles; linear attribute comparison against nearest source surface; not rendered appearance','source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'samples':len(points),'seed':0,'linear_pbr':{name:_metrics(reference[:,i],pred[:,i]) for i,name in enumerate(CHANNELS)}};(root/'fan-patch-corner-only-material.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['linear_pbr'],indent=2))
