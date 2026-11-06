"""Warm-start a shape VAE on a small, pack-split CC0 asset pilot."""
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
 ckroot=Path('/kaggle/input/t3d-mixed-pilot-shape-checkpoint');report=json.loads((ckroot/'triton-mixed-32.json').read_text());checkpoint=ckroot/'triton-mixed-600.pt';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 expected=next(x['checkpoint_sha256'] for x in report['rows'] if x['step']==600)
 if digest!=expected:raise ValueError('Checkpoint hash mismatch')
 old=AssetDataset('artifacts/mixed-shape-pilot-32/manifest.json',split='train');root=Path('/kaggle/input/t3d-kenney-shape-pilot-32');sets={s:AssetDataset(root/'manifest.json',split=s) for s in ('train','validation')}
 trainer=VAETrainer.resume(checkpoint,device='cuda',manifest_sha256=old.fingerprint)
 if trainer.steps!=600 or trainer.config.get('convolution_backend')!='triton' or trainer.occupancy_level_weights is not None:raise ValueError('Unexpected warm start')
 trainer.manifest_sha256=sets['train'].fingerprint;trainer.batch_cursor=0;trainer.epoch=0;trainer.shuffle_seed=2026;trainer.batch_size=1
 assets={s:[ds[i] for i in range(len(ds))] for s,ds in sets.items()};rows=[];start=time.perf_counter()
 for target in (600,800,1000,1200):
  while trainer.steps<target:
   index=trainer.next_indices(len(assets['train']),1)[0];loss=trainer.update([assets['train'][index]])
   if trainer.steps%50==0:print(json.dumps(dict(step=trainer.steps,epoch=trainer.epoch,asset=sets['train'].records[index].asset_id,loss=loss['total'])),flush=True)
  cases=[]
  for split,ds in sets.items():
   for i,asset in enumerate(assets[split]):
    teacher=trainer.evaluate([asset]);decoded=reconstruct_asset(asset,trainer.model,support='predicted');case=dict(split=split,asset=ds.records[i].asset_id,group=ds.records[i].source_group,teacher_forced_loss=teacher,free_occupancy=metrics(asset,decoded))
    if target==1200 and split=='validation':
     data=dumps(decoded);(Path('/kaggle/working')/(ds.records[i].asset_id+'-predicted.t3d')).write_bytes(data);case['reconstruction_sha256']=hashlib.sha256(data).hexdigest()
    cases.append(case)
  row=dict(step=trainer.steps,cases=cases)
  if target in (800,1000,1200):
   p=Path('/kaggle/working')/f'triton-kenney-{target}.pt';trainer.save(p);row['checkpoint_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
  rows.append(row);summary={s:sum(c['free_occupancy']['iou'] for c in cases if c['split']==s)/len(assets[s]) for s in sets};print(json.dumps(dict(step=trainer.steps,mean_iou=summary)),flush=True)
 result=dict(scope='Warm-start from previously inspected mixed-asset pilot, then train on 16 uncurated Kenney models; eight validation models from other packs; no semantic deduplication or quality benchmark',device=torch.cuda.get_device_name(0),old_manifest_sha256=old.fingerprint,manifest_sha256=sets['train'].fingerprint,initial_checkpoint_sha256=digest,elapsed_seconds=time.perf_counter()-start,rows=rows)
 Path('/kaggle/working/triton-kenney-32.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
