"""Audit raw and containment-masked occupancy for saved paired checkpoints on CUDA."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from t3d.coupled_occupancy import CoupledOccupancyFlow

def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 root=Path('artifacts/kaggle-coupled-real-position-v14');data=Path('artifacts/coupled-real-pilot');manifest=json.loads((data/'manifest.json').read_text());raw=(data/'cesiumman-16.npz').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=manifest['data_sha256']:raise ValueError('Data hash mismatch')
 with np.load(data/'cesiumman-16.npz',allow_pickle=False) as z:shape=torch.tensor(z['shape'].reshape(-1,1),device='cuda').bool();skeleton=torch.tensor(z['skeleton'].reshape(-1,1),device='cuda').bool()
 n=manifest['grid'];coords=torch.cartesian_prod(*(torch.arange(n,device='cuda') for _ in range(3)));rows=[]
 checkpoints=[('position-baseline',root/'baseline'),('position-absolute',root/'absolute')]
 foreground=Path('artifacts/kaggle-coupled-real-foreground-v17')
 if foreground.exists():
  checkpoints.extend((f'foreground-{label}',foreground/label) for label in ('baseline','weighted'))
 for label, checkpoint_dir in checkpoints:
  p=checkpoint_dir/'coupled-real-pilot.pt';report=json.loads((checkpoint_dir/'coupled-real-report.json').read_text())
  if hashlib.sha256(p.read_bytes()).hexdigest()!=report['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
  ck=torch.load(p,map_location='cuda',weights_only=True);model=CoupledOccupancyFlow(8,width=48,heads=2,depth=2,patch_size=64,absolute_coords=ck['absolute_coords']).cuda();model.load_state_dict(ck['model']);model.eval()
  for seed in (4301,4302,4303,4304,4305):
   g=torch.Generator(device='cuda').manual_seed(seed);ns=torch.randn(shape.shape,device='cuda',generator=g);nk=torch.randn(skeleton.shape,device='cuda',generator=g)
   s,k,raw_s,raw_k=model.sample(ns,nk,coords,steps=12,guidance=1.,containment='enclosed',return_values=True);raw_mask=raw_k[:,0]>.5;target=skeleton[:,0]
   def score(pred):
    tp=int((pred&target).sum());return dict(predicted=int(pred.sum()),intersection=tp,iou=tp/max(1,int((pred|target).sum())))
   rows.append(dict(model=label,seed=seed,shape_voxels=int(s.sum()),raw_skeleton=score(raw_mask),masked_skeleton=score(k),masked_removed=int((raw_mask&~k).sum()),target_outside_mask=int((target&~(__import__('t3d.coupled_occupancy',fromlist=['closed_shell_solid_mask']).closed_shell_solid_mask(coords,s))).sum())))
 result=dict(scope='Single training asset, 5 fixed noises, 200-step checkpoints; raw vs filled-predicted-shell masked skeleton; not generalization or quality benchmark',device=torch.cuda.get_device_name(0),data_sha256=manifest['data_sha256'],rows=rows)
 Path('/kaggle/working/coupled-masking-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
