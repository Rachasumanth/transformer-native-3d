"""Small text-conditioned occupancy experiment; no generalization claim."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from t3d.conditioning import PromptConditioner
from t3d.coupled_occupancy import CoupledOccupancyFlow

def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 torch.manual_seed(73);torch.cuda.manual_seed_all(73);device='cuda';root=Path('artifacts/prompt-occupancy-pilot');manifest_raw=(root/'manifest.json').read_bytes();manifest=json.loads(manifest_raw);rows=manifest['assets'];assets={}
 if len({r['source_sha256'] for r in rows})!=len(rows):raise ValueError('Duplicate asset source')
 for r in rows:
  p=root/r['file']
  if hashlib.sha256(p.read_bytes()).hexdigest()!=r['data_sha256']:raise ValueError('Data hash mismatch')
  with np.load(p,allow_pickle=False) as z:assets[r['name']]=(torch.tensor(z['shape'].reshape(-1,1),device=device),torch.tensor(z['skeleton'].reshape(-1,1),device=device))
 conditioner=PromptConditioner.from_pretrained(image_model='facebook/dinov2-small',image_revision='ed25f3a31f01632728cabb09d1542f84ab7b0056',text_model='openai/clip-vit-base-patch32',text_revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',output_channels=32).to(device).train()
 raw={}
 with torch.no_grad():
  for row in rows:
   encoded=conditioner.tokenizer(row['prompt'],return_tensors='pt',padding=False,truncation=True);encoded={k:v.to(device) for k,v in encoded.items()};raw[row['name']]=conditioner.text_encoder(**encoded).last_hidden_state[0].detach()
 n=manifest['grid'];coords=torch.cartesian_prod(*(torch.arange(n,device=device) for _ in range(3)));model=CoupledOccupancyFlow(32,width=48,heads=2,depth=2,patch_size=64,absolute_coords=True).to(device);optimizer=torch.optim.AdamW(list(model.parameters())+list(conditioner.text_projection.parameters()),lr=.001)
 train=[x for x in rows if x['split']=='train'];rng=np.random.default_rng(73);history=[]
 for step in range(1,301):
  row=train[int(rng.integers(len(train)))];shape,skeleton=assets[row['name']];tokens=conditioner.text_projection(raw[row['name']]);optimizer.zero_grad(set_to_none=True)
  losses=model.loss(shape,skeleton,coords,tokens,drop_probability=.1,containment='aabb');loss=sum(losses.values())
  if not torch.isfinite(loss):raise ValueError('Nonfinite flow loss')
  loss.backward();torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(conditioner.text_projection.parameters()),1.);optimizer.step()
  if step%25==0:history.append(dict(step=step,asset=row['name'],shape_loss=float(losses['shape'].detach()),skeleton_loss=float(losses['skeleton'].detach())));print(json.dumps(history[-1]),flush=True)
 model.eval();conditioner.eval();evaluations=[]
 for i,row in enumerate(rows):
  shape,skeleton=assets[row['name']];generator=torch.Generator(device=device).manual_seed(7301+i);noise_s=torch.randn(shape.shape,device=device,generator=generator);noise_k=torch.randn(skeleton.shape,device=device,generator=generator)
  wrong=next(x for x in train if x['name']!=row['name']);results={};masks={}
  for label,prompt_row in [('correct',row),('different',wrong)]:
   with torch.no_grad():tokens=conditioner.text_projection(raw[prompt_row['name']]);pred_s,pred_k=model.sample(noise_s,noise_k,coords,tokens,steps=12,guidance=1.,containment='aabb')
   def score(pred,target):
    target=target[:,0].bool();inter=int((pred&target).sum());return dict(predicted=int(pred.sum()),target=int(target.sum()),intersection=inter,iou=inter/max(1,int((pred|target).sum())))
   results[label]=dict(prompt=prompt_row['prompt'],shape=score(pred_s,shape),skeleton=score(pred_k,skeleton));masks[label]=(pred_s,pred_k)
  evaluations.append(dict(asset=row['name'],split=row['split'],evaluation_seed=7301+i,results=results,shape_prompt_difference=int((masks['correct'][0]^masks['different'][0]).sum()),skeleton_prompt_difference=int((masks['correct'][1]^masks['different'][1]).sum())))
 checkpoint=Path('/kaggle/working/prompt-occupancy.pt');torch.save(dict(version=1,model=model.state_dict(),text_projection=conditioner.text_projection.state_dict(),manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),models=conditioner.model_provenance),checkpoint)
 report=dict(scope=manifest['scope']+'; 300-step T4 integration pilot, train/validation prompts; prompt swaps share evaluation noise; no image conditioning',device=torch.cuda.get_device_name(0),models=conditioner.model_provenance,manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),history=history,evaluations=evaluations)
 Path('/kaggle/working/prompt-occupancy-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
