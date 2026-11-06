"""Inspect predicted versus teacher-forced support on paired 32-cubed VAE checkpoints."""
import hashlib,json
from pathlib import Path
import torch
from t3d.container import dumps
from t3d.dataset import AssetDataset
from t3d.extraction import voxels_to_mesh
from t3d.reconstruction import reconstruct_asset
from t3d.training import VAETrainer


def metrics(reference,predicted):
 truth={tuple(v) for v in reference.coords.tolist()};observed={tuple(v) for v in predicted.coords.tolist()};tp=len(truth&observed);union=len(truth|observed)
 return dict(target=len(truth),predicted=len(observed),intersection=tp,precision=tp/len(observed) if observed else None,recall=tp/len(truth) if truth else None,iou=tp/union if union else None)


def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 dataset_path='artifacts/pilot-dataset-32/manifest.json';datasets={s:AssetDataset(dataset_path,split=s) for s in ('train','validation')};root=Path('/kaggle/input/t3d-paired-32-checkpoints');report=json.loads((root/'triton-paired-32.json').read_text());out=Path('/kaggle/working');rows=[]
 for backend in ('reference','triton'):
  checkpoint=root/(backend+'-32.pt');digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
  if digest!=report['results'][backend]['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
  trainer=VAETrainer.resume(checkpoint,device='cuda',manifest_sha256=datasets['train'].fingerprint)
  for split,ds in datasets.items():
   asset=ds[0]
   for support in ('teacher','predicted'):
    predicted=reconstruct_asset(asset,trainer.model,support=support);vertices,faces=voxels_to_mesh(predicted)
    path=out/f'{backend}-{split}-{support}.t3d';data=dumps(predicted);path.write_bytes(data)
    rows.append(dict(backend=backend,split=split,support=support,occupancy=metrics(asset,predicted),mesh_vertices=len(vertices),mesh_triangles=len(faces),reconstruction_sha256=hashlib.sha256(data).hexdigest()))
 result=dict(scope='30-step paired shape-only VAE; teacher vs free occupancy, two previously inspected assets; no generated-asset or surface-distance benchmark',device=torch.cuda.get_device_name(0),manifest_sha256=datasets['train'].fingerprint,rows=rows)
 (out/'paired-reconstruction-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
