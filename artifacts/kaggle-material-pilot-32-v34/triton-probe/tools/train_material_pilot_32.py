"""Small material-only VAE training with per-channel teacher-support errors."""
import hashlib,json,time
from pathlib import Path
import torch
from t3d.dataset import AssetDataset
from t3d.training import VAETrainer


def material_errors(trainer,asset):
 trainer.model.eval();coords,features=trainer.batch([asset])
 with torch.inference_mode():
  posterior=trainer.model.encode(coords,features);out,raw,_=trainer.model.decode(posterior.coords,posterior.mean,teacher_occupancy=posterior.occupancy)
  sorted_coords,inverse=torch.unique(coords,dim=0,sorted=True,return_inverse=True)
  if not torch.equal(out,sorted_coords):raise ValueError('Teacher support mismatch')
  target=features[torch.argsort(inverse)];error=(raw.sigmoid()-target).abs().mean(dim=0)
 return dict(zip(('r','g','b','metallic','roughness','alpha'),[float(x) for x in error]))


def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 torch.manual_seed(481);torch.cuda.manual_seed_all(481);sets={s:AssetDataset('artifacts/material-pilot-32/manifest.json',split=s,require_pbr=True) for s in ('train','validation')};assets={s:d[0] for s,d in sets.items()}
 trainer=VAETrainer(mode='material',channels=(8,16,32,64,128),latent_channels=16,device='cuda',manifest_sha256=sets['train'].fingerprint,convolution_backend='triton');rows=[];start=time.perf_counter()
 for target_step in (0,100,300,600):
  while trainer.steps<target_step:
   loss=trainer.update([assets['train']])
   if trainer.steps%50==0:print(json.dumps(dict(step=trainer.steps,train_loss=loss['total'])),flush=True)
  cases={s:dict(teacher_loss=trainer.evaluate([asset]),mean_absolute_error=material_errors(trainer,asset)) for s,asset in assets.items()};row=dict(step=trainer.steps,cases=cases)
  if target_step in (300,600):
   p=Path('/kaggle/working')/f'material-pilot-{target_step}.pt';trainer.save(p);row['checkpoint_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
  rows.append(row);print(json.dumps(row),flush=True)
 result=dict(scope='One previously inspected textured train asset and one validation asset at 32 cubed; teacher support supplied; no generated mesh or render quality claim',device=torch.cuda.get_device_name(0),manifest_sha256=sets['train'].fingerprint,elapsed_seconds=time.perf_counter()-start,rows=rows)
 Path('/kaggle/working/material-pilot-32.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
