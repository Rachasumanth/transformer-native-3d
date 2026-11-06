"""Small paired 32-cubed real-asset SC-VAE training comparison on CUDA."""
import gc,hashlib,json,time
from pathlib import Path
import torch
from t3d.dataset import AssetDataset
from t3d.training import VAETrainer


def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 torch.backends.cuda.matmul.allow_tf32=False;p='artifacts/pilot-dataset-32/manifest.json';train=AssetDataset(p,split='train');valid=AssetDataset(p,split='validation');asset=train[0];holdout=valid[0];results={}
 for backend in ('reference','triton'):
  torch.manual_seed(172);torch.cuda.manual_seed_all(172)
  model=VAETrainer(mode='shape',channels=(8,16,32,64,128),latent_channels=16,device='cuda',manifest_sha256=train.fingerprint,convolution_backend=backend)
  before=model.evaluate([holdout]);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
  torch.manual_seed(173);torch.cuda.manual_seed_all(173);start=time.perf_counter();history=[]
  for step in range(1,31):
   metrics=model.update([asset])
   if step in (1,10,20,30):history.append(dict(step=step,metrics=metrics));print(json.dumps(dict(backend=backend,step=step,loss=metrics['total'])),flush=True)
  torch.cuda.synchronize();elapsed=time.perf_counter()-start;peak=max(0,torch.cuda.max_memory_allocated()-base);after=model.evaluate([holdout]);checkpoint=Path('/kaggle/working')/(backend+'-32.pt');model.save(checkpoint)
  results[backend]=dict(validation_before=before,validation_after=after,steps=30,elapsed_seconds=elapsed,peak_extra_bytes=peak,history=history,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
  del model;gc.collect();torch.cuda.empty_cache()
 result=dict(scope='One 32-cubed training asset and one validation asset; 30 steps, one training seed; teacher-forced losses; not a generation-quality benchmark',device=torch.cuda.get_device_name(0),manifest_sha256=train.fingerprint,train_voxels=len(asset.coords),validation_voxels=len(holdout.coords),results=results)
 Path('/kaggle/working/triton-paired-32.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
