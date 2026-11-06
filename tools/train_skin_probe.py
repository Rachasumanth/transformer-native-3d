"""Small real-source SkinAE optimization run with an asset-held-out evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from t3d.skin_ae import SkinAE,skin_reconstruction_loss


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-sha256',required=True)
    args=parser.parse_args();assert torch.cuda.is_available()
    torch.manual_seed(23);torch.cuda.manual_seed_all(23)
    root=Path('artifacts/skin-pilot');raw=(root/'manifest.json').read_bytes();manifest=json.loads(raw)
    data=[]
    for row in manifest['assets']:
        path=root/row['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Pilot identity mismatch')
        with np.load(path,allow_pickle=False) as a:
            sample={k:torch.from_numpy(a[k]).cuda() for k in ('joints','parents','weights')}
        data.append((row,sample))
    config=dict(width=64,heads=4,layers=2,frequencies=4)
    model=SkinAE(**config).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    def evaluate():
        model.eval();result={}
        with torch.no_grad():
            for row,sample in data:
                target=sample['weights'];_,_,logits=model(sample['joints'],sample['parents'],target)
                predicted=logits.softmax(-1)
                result[row['name']]=dict(split=row['split'],kl=float(skin_reconstruction_loss(logits,target)),
                    uniform_kl=float(skin_reconstruction_loss(torch.zeros_like(logits),target)),
                    mean_absolute_weight_error=float((predicted-target).abs().mean()),
                    dominant_joint_accuracy=float((predicted.argmax(-1)==target.argmax(-1)).float().mean()))
        model.train();return result
    report=dict(source_sha256=args.source_sha256,manifest_sha256=hashlib.sha256(raw).hexdigest(),
                device=torch.cuda.get_device_name(0),config=config,seed=23,updates=400,
                scope=manifest['scope'],initial=evaluate(),history=[])
    train=[sample for row,sample in data if row['split']=='train']
    for step in range(400):
        sample=train[step%len(train)];target=sample['weights']
        ids=torch.randperm(len(target),device='cuda')[:256]
        _,_,logits=model(sample['joints'],sample['parents'],target[ids])
        loss=skin_reconstruction_loss(logits,target[ids])
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if (step+1)%100==0:
            report['history'].append(dict(update=step+1,metrics=evaluate()))
            print(json.dumps(report['history'][-1]),flush=True)
    report['final']=evaluate()
    checkpoint=Path('/kaggle/working/skin-pilot.pt')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config,
                    step=400,manifest_sha256=report['manifest_sha256'],source_sha256=args.source_sha256,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),checkpoint)
    report['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model.freeze();report['frozen']=not model.training and all(not p.requires_grad for p in model.parameters())
    Path('/kaggle/working/skin-training-report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
