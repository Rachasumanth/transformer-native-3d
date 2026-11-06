"""Warm-start a small shape VAE on a diverse but previously inspected pilot."""
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
 ckroot=Path('/kaggle/input/t3d-paired-32-checkpoints');prior=json.loads((ckroot/'triton-paired-32.json').read_text());checkpoint=ckroot/'triton-32.pt';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 if digest!=prior['results']['triton']['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
 old=AssetDataset('artifacts/pilot-dataset-32/manifest.json',split='train');sets={s:AssetDataset('artifacts/mixed-shape-pilot-32/manifest.json',split=s) for s in ('train','validation')};trainer=VAETrainer.resume(checkpoint,device='cuda',manifest_sha256=old.fingerprint)
 if trainer.steps!=30 or trainer.config.get('convolution_backend')!='triton' or trainer.occupancy_level_weights is not None:raise ValueError('Unexpected warm start')
 trainer.manifest_sha256=sets['train'].fingerprint;trainer.batch_cursor=0;trainer.epoch=0
 assets={s:[ds[i] for i in range(len(ds))] for s,ds in sets.items()};rows=[];start=time.perf_counter()
 for target in (30,100,300,600):
  while trainer.steps<target:
   index=(trainer.steps-30)%len(assets['train']);loss=trainer.update([assets['train'][index]])
   if trainer.steps%50==0:print(json.dumps(dict(step=trainer.steps,asset=sets['train'].records[index].asset_id,train_loss=loss['total'])),flush=True)
  cases=[]
  for split,ds in sets.items():
   for i,asset in enumerate(assets[split]):
    teacher=trainer.evaluate([asset]);decoded=reconstruct_asset(asset,trainer.model,support='predicted');row=dict(split=split,asset=ds.records[i].asset_id,teacher_forced_loss=teacher,free_occupancy=metrics(asset,decoded))
    if target==600:
     data=dumps(decoded);(Path('/kaggle/working')/(ds.records[i].asset_id+'-predicted.t3d')).write_bytes(data);row['reconstruction_sha256']=hashlib.sha256(data).hexdigest()
    cases.append(row)
  row=dict(step=trainer.steps,cases=cases)
  if target in (300,600):
   p=Path('/kaggle/working')/f'triton-mixed-{target}.pt';trainer.save(p);row['checkpoint_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
  rows.append(row);print(json.dumps(row),flush=True)
 result=dict(scope='Warm-start from 30 steps on Avocado, then round-robin four prior-inspected training assets; two prior-inspected validation assets; no untouched benchmark',device=torch.cuda.get_device_name(0),old_manifest_sha256=old.fingerprint,manifest_sha256=sets['train'].fingerprint,initial_checkpoint_sha256=digest,elapsed_seconds=time.perf_counter()-start,rows=rows)
 Path('/kaggle/working/triton-mixed-32.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
