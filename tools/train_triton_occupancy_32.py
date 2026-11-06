"""Continue a tiny 32-cubed VAE and measure free occupancy over training."""
import hashlib,json,time
from pathlib import Path
import torch
from t3d.container import dumps
from t3d.dataset import AssetDataset
from t3d.reconstruction import reconstruct_asset
from t3d.training import VAETrainer
from tools.audit_triton_paired_reconstruction import metrics


def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 root=Path('/kaggle/input/t3d-paired-32-checkpoints');report=json.loads((root/'triton-paired-32.json').read_text());checkpoint=root/'triton-32.pt'
 if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=report['results']['triton']['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
 manifest='artifacts/pilot-dataset-32/manifest.json';sets={x:AssetDataset(manifest,split=x) for x in ('train','validation')};trainer=VAETrainer.resume(checkpoint,device='cuda',manifest_sha256=sets['train'].fingerprint)
 if trainer.steps!=30 or trainer.config.get('convolution_backend')!='triton':raise ValueError('Unexpected checkpoint')
 rows=[];started=time.perf_counter()
 for target_step in (30,100,300,600):
  while trainer.steps<target_step:
   result=trainer.update([sets['train'][0]])
   if trainer.steps%50==0:print(json.dumps(dict(step=trainer.steps,train_loss=result['total'])),flush=True)
  cases={}
  for split,ds in sets.items():
   asset=ds[0];loss=trainer.evaluate([asset]);decoded=reconstruct_asset(asset,trainer.model,support='predicted');cases[split]=dict(teacher_forced_loss=loss,free_occupancy=metrics(asset,decoded))
   if target_step==600:
    data=dumps(decoded);(Path('/kaggle/working')/(split+'-predicted.t3d')).write_bytes(data);cases[split]['reconstruction_sha256']=hashlib.sha256(data).hexdigest()
  rows.append(dict(step=trainer.steps,cases=cases));print(json.dumps(rows[-1]),flush=True)
 out=Path('/kaggle/working/triton-occupancy-600.pt');trainer.save(out)
 result=dict(scope='Continuation from 30 to 600 steps on one 32-cubed training asset; validation asset previously inspected; posterior-mean free occupancy is not generation',device=torch.cuda.get_device_name(0),manifest_sha256=sets['train'].fingerprint,initial_checkpoint_sha256=report['results']['triton']['checkpoint_sha256'],final_checkpoint_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),elapsed_seconds=time.perf_counter()-started,rows=rows)
 Path('/kaggle/working/triton-occupancy-32.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
