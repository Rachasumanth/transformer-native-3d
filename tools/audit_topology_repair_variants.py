"""Audit guarded geometry repair against retained real export variants."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from t3d.import_rig import load_rig_glb
from t3d.topology_repair import repair_local_fans
from t3d.metrics import surface_comparison
source,_=load_rig_glb('artifacts/rigged-inputs/CesiumMan/CesiumMan.glb').normalized();rows=[]
for version in (10,11,12,13,14):
 for suffix in ('','-strict'):
  p=Path(f'artifacts/kaggle-textured-rig-export-v{version}/export-real-rigged{suffix}.glb')
  if not p.is_file():continue
  item={'version':version,'variant':'strict' if suffix else 'standard','input_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
  try:
   rig=load_rig_glb(p);start=time.monotonic();v,f,a=repair_local_fans(rig.vertices,rig.faces);item.update(status='accepted',repair_seconds=time.monotonic()-start,summary={k:a[k] for k in ('input_faces','output_faces','added_faces','loop_lengths','near_degenerate_faces','exact_weld_boundary_edges','exact_weld_nonmanifold_edges')},surface=surface_comparison(source.vertices,source.faces,v,f,samples=512,tolerance=.01))
  except ValueError as e:item.update(status='rejected',reason=str(e))
  rows.append(item);print(version,item['variant'],item['status'],item.get('reason',''),flush=True)
out=Path('artifacts/topology-repair-variants.json');out.write_text(json.dumps({'scope':'Single CesiumMan source, multiple resolution and remesh variants; geometry only; no UV skin, rendering, self-intersection or generic generalization claim','rows':rows},indent=2)+'\n')
