"""Explore fixed skeleton thresholds on a small text-occupancy pilot checkpoint."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer,CLIPTextModel
from t3d.coupled_occupancy import CoupledOccupancyFlow,shape_bounding_box_mask


def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 device='cuda';root=Path('artifacts/prompt-occupancy-pilot');raw=(root/'manifest.json').read_bytes();manifest=json.loads(raw)
 ckroot=Path('/kaggle/input/t3d-text-occupancy-pilot-checkpoint');p=ckroot/'prompt-occupancy.pt';report=json.loads((ckroot/'prompt-occupancy-report.json').read_text())
 if hashlib.sha256(p.read_bytes()).hexdigest()!=report['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
 ck=torch.load(p,map_location=device,weights_only=True)
 if hashlib.sha256(raw).hexdigest()!=ck['manifest_sha256'] or ck['foreground_weight']!=8.:raise ValueError('Checkpoint provenance mismatch')
 revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268';model_id='openai/clip-vit-base-patch32'
 tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision);encoder=CLIPTextModel.from_pretrained(model_id,revision=revision).to(device).eval().requires_grad_(False)
 projection=nn.Linear(encoder.config.hidden_size,32).to(device);projection.load_state_dict(ck['text_projection']);projection.eval()
 model=CoupledOccupancyFlow(32,width=48,heads=2,depth=2,patch_size=64,absolute_coords=True).to(device);model.load_state_dict(ck['model']);model.eval()
 n=manifest['grid'];coords=torch.cartesian_prod(*(torch.arange(n,device=device) for _ in range(3)));rows=[];thresholds=[.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]
 with torch.inference_mode():
  for i,asset in enumerate(manifest['assets']):
   path=root/asset['file']
   if hashlib.sha256(path.read_bytes()).hexdigest()!=asset['data_sha256']:raise ValueError('Data hash mismatch')
   with np.load(path,allow_pickle=False) as z:
    shape=torch.tensor(z['shape'].reshape(-1,1),device=device);target=torch.tensor(z['skeleton'].reshape(-1),device=device).bool()
   encoded=tokenizer(asset['prompt'],return_tensors='pt',padding=False,truncation=True);encoded={k:v.to(device) for k,v in encoded.items()};tokens=projection(encoder(**encoded).last_hidden_state[0])
   for repeat in range(3):
    seed=8301+i+100*repeat;g=torch.Generator(device=device).manual_seed(seed);ns=torch.randn(shape.shape,device=device,generator=g);nk=torch.randn(shape.shape,device=device,generator=g)
    surface,_,_,raw_k=model.sample(ns,nk,coords,tokens,steps=12,guidance=1.,containment='aabb',return_values=True);box=shape_bounding_box_mask(coords,surface)
    scores={}
    for threshold in thresholds:
     pred=(raw_k[:,0]>threshold)&box;tp=int((pred&target).sum());count=int(pred.sum());truth=int(target.sum());scores[str(threshold)]=dict(iou=tp/max(1,count+truth-tp),precision=tp/max(1,count),recall=tp/max(1,truth),predicted=count,target=truth)
    rows.append(dict(asset=asset['name'],split=asset['split'],seed=seed,scores=scores))
 training=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
 mean=lambda group,t,field:float(np.mean([r['scores'][str(t)][field] for r in group]))
 selected=max(thresholds,key=lambda t:mean(training,t,'iou'))
 result=dict(scope='Threshold selection on three training assets, reported on one previously inspected validation asset; no untouched benchmark',checkpoint_sha256=report['checkpoint_sha256'],manifest_sha256=ck['manifest_sha256'],thresholds=thresholds,selected_train_iou_threshold=selected,summary={split:{str(t):{field:mean(group,t,field) for field in ('iou','precision','recall','predicted','target')} for t in thresholds} for split,group in [('train',training),('validation',validation)]},rows=rows)
 Path('/kaggle/working/prompt-skeleton-threshold.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
