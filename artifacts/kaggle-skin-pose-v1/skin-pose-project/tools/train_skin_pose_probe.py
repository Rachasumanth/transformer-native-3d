"""Train/evaluate SkinAE over animation frames with a source-held-out rig."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from t3d.skin_ae import SkinAE,skin_reconstruction_loss


def load_pose_data(root):
    root=Path(root);manifest_path=root/'manifest.json';raw=manifest_path.read_bytes()
    manifest=json.loads(raw)
    if manifest.get('version')!=2:raise ValueError('Expected version-2 pose manifest')
    data=[];source_splits={}
    for row in manifest['assets']:
        if row['split'] not in ('train','validation','test'):
            raise ValueError('Invalid source split')
        digest=row['source_sha256']
        if digest in source_splits and source_splits[digest]!=row['split']:
            raise ValueError('A source asset leaks across dataset splits')
        source_splits[digest]=row['split']
        path=root/row['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Pose data hash mismatch')
        with np.load(path,allow_pickle=False) as archive:
            points=np.asarray(archive['points'],dtype=np.float32)
            joints=np.asarray(archive['joints'],dtype=np.float32)
            parents=np.asarray(archive['parents'],dtype=np.int64)
            weights=np.asarray(archive['weights'],dtype=np.float32)
            times=np.asarray(archive['animation_time'],dtype=np.float32)
            clips=np.asarray(archive['animation_index'],dtype=np.int32)
        if (points.ndim!=3 or points.shape[2]!=3 or joints.shape!=(len(points),len(parents),3)
                or weights.ndim!=2 or weights.shape[1]!=len(parents) or times.shape!=(len(points),)
                or clips.shape!=(len(points),) or not np.isfinite(points).all()
                or not np.isfinite(joints).all() or not np.isfinite(weights).all()
                or np.any(weights<0) or not np.allclose(weights.sum(1),1,atol=1e-5,rtol=0)):
            raise ValueError('Malformed animation pose data')
        for i in range(len(points)):
            data.append(dict(asset=row['name'],split=row['split'],pose=i,
                points=torch.from_numpy(points[i]),joints=torch.from_numpy(joints[i]),
                parents=torch.from_numpy(parents),weights=torch.from_numpy(weights),
                animation_index=int(clips[i]),time=float(times[i])))
    train_assets={row['name'] for row in manifest['assets'] if row['split']=='train'}
    validation_assets={row['name'] for row in manifest['assets'] if row['split']=='validation'}
    if len(train_assets)<2 or not validation_assets or train_assets&validation_assets:
        raise ValueError('Need at least two training rigs and disjoint validation rigs')
    return manifest,hashlib.sha256(raw).hexdigest(),data


@torch.no_grad()
def evaluate(model,data,device):
    model.eval();groups={}
    for sample in data:
        joints=sample['joints'].to(device);parents=sample['parents'].to(device)
        target=sample['weights'].to(device)
        _,_,logits=model(joints,parents,target)
        row=groups.setdefault(sample['asset'],dict(split=sample['split'],poses=0,
            kl=0.,uniform_kl=0.,mean_absolute_weight_error=0.,dominant_joint_accuracy=0.))
        prediction=logits.softmax(-1)
        row['poses']+=1
        row['kl']+=float(skin_reconstruction_loss(logits,target))
        row['uniform_kl']+=float(skin_reconstruction_loss(torch.zeros_like(logits),target))
        row['mean_absolute_weight_error']+=float((prediction-target).abs().mean())
        row['dominant_joint_accuracy']+=float((prediction.argmax(-1)==target.argmax(-1)).float().mean())
    for row in groups.values():
        count=row['poses']
        for key in ('kl','uniform_kl','mean_absolute_weight_error','dominant_joint_accuracy'):
            row[key]/=count
    model.train();return groups


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-sha256',required=True)
    parser.add_argument('--data',default='artifacts/skin-pose-pilot')
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--batch-size',type=int,default=256)
    args=parser.parse_args()
    if args.steps<=0 or args.batch_size<=0:raise ValueError('Steps and batch size must be positive')
    if not torch.cuda.is_available():raise RuntimeError('Kaggle CUDA is required for this training probe')
    torch.manual_seed(31);torch.cuda.manual_seed_all(31);rng=np.random.default_rng(31)
    manifest,manifest_hash,data=load_pose_data(args.data)
    device=torch.device('cuda');config=dict(width=64,heads=4,layers=2,frequencies=4)
    model=SkinAE(**config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    train=[sample for sample in data if sample['split']=='train']
    validation=[sample for sample in data if sample['split']=='validation']
    train_by_asset={name:[s for s in train if s['asset']==name]
                    for name in sorted({s['asset'] for s in train})}
    initial=evaluate(model,data,device);best=evaluate(model,validation,device)
    best_score=float(np.mean([row['kl'] for row in best.values()]));best_step=0
    best_state=copy.deepcopy({key:value.detach().cpu() for key,value in model.state_dict().items()})
    history=[]
    for step in range(1,args.steps+1):
        asset=rng.choice(list(train_by_asset))
        sample=train_by_asset[asset][int(rng.integers(len(train_by_asset[asset])))]
        weights=sample['weights'].to(device)
        ids=torch.randperm(len(weights),device=device)[:min(args.batch_size,len(weights))]
        _,_,logits=model(sample['joints'].to(device),sample['parents'].to(device),weights[ids])
        loss=skin_reconstruction_loss(logits,weights[ids])
        if not torch.isfinite(loss):raise ValueError('Nonfinite pose training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise ValueError('Nonfinite SkinAE gradient norm')
        optimizer.step()
        if step%100==0 or step==args.steps:
            metrics=evaluate(model,validation,device)
            score=float(np.mean([row['kl'] for row in metrics.values()]))
            history.append(dict(step=step,validation=metrics,training_loss=float(loss.detach())))
            if score<best_score:
                best_score=score;best_step=step;best=metrics
                best_state=copy.deepcopy({key:value.detach().cpu() for key,value in model.state_dict().items()})
            print(json.dumps(history[-1]),flush=True)
    model.load_state_dict(best_state);model.freeze()
    checkpoint=Path('/kaggle/working/skin-pose-pilot.pt')
    torch.save(dict(version=1,model=best_state,config=config,best_step=best_step,
        source_sha256=args.source_sha256,manifest_sha256=manifest_hash,
        train_assets=sorted(train_by_asset),validation_assets=sorted({s['asset'] for s in validation}),
        rng_state=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all()),checkpoint)
    report=dict(source_sha256=args.source_sha256,manifest_sha256=manifest_hash,
        device=torch.cuda.get_device_name(0),config=config,seed=31,steps=args.steps,
        train_pose_count=len(train),validation_pose_count=len(validation),
        train_assets=sorted(train_by_asset),validation_assets=sorted({s['asset'] for s in validation}),
        scope=manifest['scope'],
        best_step=best_step,best_validation=best,history=history,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),frozen=True)
    # Record initial metrics from the exact pre-training weights (the value is
    # kept separately so reporting does not construct an unrelated model).
    report['initial']=initial
    Path('/kaggle/working/skin-pose-training-report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
