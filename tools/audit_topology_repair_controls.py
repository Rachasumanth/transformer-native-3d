"""Run experimental geometry repair on distinct existing input assets."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from t3d.import_glb import load_glb
from t3d.import_rig import load_rig_glb
from t3d.topology_repair import repair_local_fans
from t3d.metrics import surface_comparison
paths=[('static',Path('artifacts/native-inputs')/(n+'.glb')) for n in ('Lantern','WaterBottle','Avocado')]+[('rig',Path('artifacts/rigged-candidates')/n/(n+'.glb')) for n in ('RiggedFigure','Corgi','Bat','Shark')]
rows=[]
for kind,path in paths:
 item=dict(asset=path.stem,source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 try:
  mesh=load_glb(path) if kind=='static' else load_rig_glb(path);start=time.monotonic();v,f,a=repair_local_fans(mesh.vertices,mesh.faces);item.update(status='accepted',seconds=time.monotonic()-start,input_faces=a['input_faces'],output_faces=a['output_faces'],duplicate_faces_removed=len(a['removed_duplicate_faces']),region_faces_removed=len(a['removed_region_faces']),added_faces=a['added_faces'],min_double_area=a['min_double_area'],near_degenerate_faces=a['near_degenerate_faces'])
  # Only compare when the repair changed faces; unchanged welded geometry is exact.
  if a['removed_duplicate_faces'] or a['removed_region_faces']:
   item['surface_comparison']=surface_comparison(mesh.vertices,mesh.faces,v,f,samples=512,tolerance=.01)
 except ValueError as e:item.update(status='rejected',reason=str(e))
 rows.append(item);print(item['asset'],item['status'],item.get('reason',''),flush=True)
Path('artifacts/topology-repair-controls.json').write_text(json.dumps({'scope':'Seven source assets; geometry-only, not reconstructed outputs, no rendering or self-intersection verification','assets':rows},indent=2)+'\n')
