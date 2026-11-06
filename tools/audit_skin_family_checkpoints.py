"""Compare family-validation checkpoints on full source vertices and clip times."""
import argparse,hashlib,json
from pathlib import Path
import torch
from tools.audit_skin_pose_checkpoints import audit

def main():
 parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--frames',type=int,default=25);args=parser.parse_args()
 if args.frames<2:parser.error('At least two frames required')
 torch.set_num_threads(2);rows=[]
 for setting in ('0.0','1.0'):
  directory=args.directory/('loss-'+setting);checkpoint=directory/'skin-pose-pilot.pt';report=json.loads((directory/'skin-pose-training-report.json').read_text())
  if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=report['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
  if set(report['train_assets'])!={'RiggedSimple','CesiumMan','Fox'} or set(report['validation_assets'])!={'Corgi','Bat','Shark'}:raise ValueError('Unexpected asset split')
  for asset in ('Corgi','Bat','Shark'):
   row=audit(checkpoint,asset,source=Path('artifacts/rigged-candidates')/asset/(asset+'.glb'),frames=args.frames)
   row['loss_setting']=setting;rows.append(row);print(asset,setting,row['mean_vertex_error'],flush=True)
 (args.directory/'family-deformation-audit.json').write_text(json.dumps(rows,indent=2)+'\n')
 pairs=[]
 for asset in ('Corgi','Bat','Shark'):
  a=next(r for r in rows if r['asset']==asset and r['loss_setting']=='0.0');b=next(r for r in rows if r['asset']==asset and r['loss_setting']=='1.0')
  pairs.append(dict(asset=asset,baseline=a['mean_vertex_error'],deformation=b['mean_vertex_error'],relative_improvement=1-b['mean_vertex_error']/a['mean_vertex_error']))
 (args.directory/'family-comparison.json').write_text(json.dumps(pairs,indent=2)+'\n')

if __name__=='__main__':main()
