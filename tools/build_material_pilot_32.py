"""Prepare two 32-cubed textured integration assets, not a material corpus."""
import hashlib,json
from pathlib import Path
from t3d.import_glb import prepare_glb
from t3d.container import load


def main():
 out=Path('artifacts/material-pilot-32');out.mkdir(parents=True,exist_ok=True);records=[];reports=[]
 for name,original,split in [('avocado','Avocado','train'),('waterbottle','WaterBottle','validation')]:
  src=Path('artifacts/import')/name/(original+'.glb');destination=out/(name+'.t3d');report=prepare_glb(src,destination,grid_size=32,allow_multiple_crossings=True);asset=load(destination).asset
  if asset.pbr is None:raise ValueError('PBR attributes missing')
  records.append(dict(asset_id='khronos-'+name,path=destination.name,sha256=report['prepared_sha256'],split=split,source='https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/'+original,license='CC0-1.0; Microsoft, 2017',source_sha256=report['source_sha256'],source_group='khronos-'+name));reports.append(dict(asset=name,**report));print(name,report['active_voxels'],'multiple-crossing-edges',report['crossing_diagnostics']['multiple_crossing_edges'],flush=True)
 (out/'manifest.json').write_text(json.dumps(dict(version=1,assets=records),indent=2)+'\n');(out/'conversion-report.json').write_text(json.dumps(dict(scope='Two previously examined 32-cubed textured assets, multiple crossings explicitly allowed; not a material corpus',assets=reports),indent=2)+'\n')
if __name__=='__main__':main()
