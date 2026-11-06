"""Single-asset CUDA overfit probe for paired occupancy; no generalization claim."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from t3d.coupled_occupancy import CoupledOccupancyFlow

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--absolute-coords',action='store_true');parser.add_argument('--output-dir',type=Path,default=Path('/kaggle/working'));args=parser.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 args.output_dir.mkdir(parents=True,exist_ok=True)
 torch.manual_seed(43);torch.cuda.manual_seed_all(43);device='cuda';root=Path('artifacts/coupled-real-pilot');manifest=json.loads((root/'manifest.json').read_text());p=root/'cesiumman-16.npz'
 if hashlib.sha256(p.read_bytes()).hexdigest()!=manifest['data_sha256']:raise ValueError('Pilot data hash mismatch')
 with np.load(p,allow_pickle=False) as z:shape=torch.tensor(z['shape'].reshape(-1,1),device=device);skeleton=torch.tensor(z['skeleton'].reshape(-1,1),device=device)
 n=manifest['grid'];coords=torch.cartesian_prod(*(torch.arange(n,device=device) for _ in range(3)));model=CoupledOccupancyFlow(8,width=48,heads=2,depth=2,patch_size=64,absolute_coords=args.absolute_coords).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001)
 noise_s=torch.randn_like(shape);noise_k=torch.randn_like(skeleton)
 def evaluate():
  model.eval();s,k=model.sample(noise_s,noise_k,coords,steps=12,guidance=1.,containment='enclosed')
  def score(pred,target):
   target=target[:,0].bool();tp=int((pred&target).sum());return dict(predicted=int(pred.sum()),target=int(target.sum()),intersection=tp,iou=tp/max(1,int((pred|target).sum())))
  model.train();return dict(shape=score(s,shape),skeleton=score(k,skeleton))
 before=evaluate();history=[]
 for step in range(1,201):
  opt.zero_grad(set_to_none=True);loss=model.loss(shape,skeleton,coords,drop_probability=0,containment='enclosed');value=sum(loss.values())
  if not torch.isfinite(value):raise ValueError('Nonfinite loss')
  value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  if step%25==0:history.append(dict(step=step,shape_loss=float(loss['shape'].detach()),skeleton_loss=float(loss['skeleton'].detach())));print(json.dumps(history[-1]),flush=True)
 after=evaluate();checkpoint=args.output_dir/'coupled-real-pilot.pt';torch.save({'version':1,'model':model.state_dict(),'grid':n,'seed':43,'absolute_coords':args.absolute_coords,'data_sha256':manifest['data_sha256']},checkpoint)
 result=dict(absolute_coords=args.absolute_coords,scope=manifest['scope'],device=torch.cuda.get_device_name(0),steps=200,seed=43,before=before,after=after,history=history,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
 (args.output_dir/'coupled-real-report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
