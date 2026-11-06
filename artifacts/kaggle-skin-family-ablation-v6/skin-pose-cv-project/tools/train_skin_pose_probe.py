"""Train/evaluate SkinAE over animation frames with a source-held-out rig."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from t3d.skin_ae import SkinAE,skin_reconstruction_loss


def load_pose_data(root, *, holdout_asset=None):
    root=Path(root);manifest_path=root/'manifest.json';raw=manifest_path.read_bytes()
    manifest=json.loads(raw)
    if manifest.get('version')!=2:raise ValueError('Expected version-2 pose manifest')
    data=[];source_splits={};assets=set()
    for row in manifest['assets']:
        if row['split'] not in ('train','validation','test'):
            raise ValueError('Invalid source split')
        assets.add(row['name'])
        split=('validation' if row['name']==holdout_asset else 'train') if holdout_asset else row['split']
        digest=row['source_sha256']
        if digest in source_splits and source_splits[digest]!=split:
            raise ValueError('A source asset leaks across dataset splits')
        source_splits[digest]=split
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
            bind_points=np.asarray(archive['bind_points'],dtype=np.float32) if 'bind_points' in archive else None
            deformations=np.asarray(archive['deformations'],dtype=np.float32) if 'deformations' in archive else None
        if (points.ndim!=3 or points.shape[2]!=3 or joints.shape!=(len(points),len(parents),3)
                or weights.ndim!=2 or weights.shape[1]!=len(parents) or times.shape!=(len(points),)
                or clips.shape!=(len(points),) or not np.isfinite(points).all()
                or not np.isfinite(joints).all() or not np.isfinite(weights).all()
                or np.any(weights<0) or not np.allclose(weights.sum(1),1,atol=1e-5,rtol=0)):
            raise ValueError('Malformed animation pose data')
        if (bind_points is None) != (deformations is None):raise ValueError('Incomplete deformation data')
        if bind_points is not None and (bind_points.shape!=(len(weights),3) or deformations.shape!=(len(points),len(parents),4,4) or not np.isfinite(bind_points).all() or not np.isfinite(deformations).all()):raise ValueError('Malformed deformation data')
        for i in range(len(points)):
            data.append(dict(asset=row['name'],split=split,pose=i,
                points=torch.from_numpy(points[i]),joints=torch.from_numpy(joints[i]),
                parents=torch.from_numpy(parents),weights=torch.from_numpy(weights),
                animation_index=int(clips[i]),time=float(times[i]),
                bind_points=None if bind_points is None else torch.from_numpy(bind_points),
                deformations=None if deformations is None else torch.from_numpy(deformations[i])))
    train_assets={row['name'] for row in manifest['assets'] if row['split']=='train'}
    validation_assets={row['name'] for row in manifest['assets'] if row['split']=='validation'}
    if holdout_asset is not None and holdout_asset not in assets:
        raise ValueError(f'Unknown held-out asset: {holdout_asset}')
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
            kl=0.,uniform_kl=0.,mean_absolute_weight_error=0.,dominant_joint_accuracy=0.,joint_position_error=0.))
        prediction=logits.softmax(-1)
        row['joint_position_error']+=float(torch.linalg.vector_norm((prediction-target) @ joints,dim=-1).mean())
        row['poses']+=1
        row['kl']+=float(skin_reconstruction_loss(logits,target))
        row['uniform_kl']+=float(skin_reconstruction_loss(torch.zeros_like(logits),target))
        row['mean_absolute_weight_error']+=float((prediction-target).abs().mean())
        row['dominant_joint_accuracy']+=float((prediction.argmax(-1)==target.argmax(-1)).float().mean())
    for row in groups.values():
        count=row['poses']
        for key in ('kl','uniform_kl','mean_absolute_weight_error','dominant_joint_accuracy','joint_position_error'):
            row[key]/=count
    model.train();return groups


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-sha256',required=True)
    parser.add_argument('--data',default='artifacts/skin-pose-pilot')
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--holdout-asset')
    parser.add_argument('--joint-position-loss',type=float,default=0.)
    parser.add_argument('--deformation-loss',type=float,default=0.)
    parser.add_argument('--output-dir',default='/kaggle/working')
    args=parser.parse_args()
    if not np.isfinite(args.joint_position_loss) or args.joint_position_loss<0:raise ValueError('Joint position loss must be finite and nonnegative')
    if not np.isfinite(args.deformation_loss) or args.deformation_loss<0:raise ValueError('Deformation loss must be finite and nonnegative')
    if args.steps<=0 or args.batch_size<=0:raise ValueError('Steps and batch size must be positive')
    if not torch.cuda.is_available():raise RuntimeError('Kaggle CUDA is required for this training probe')
    torch.manual_seed(31);torch.cuda.manual_seed_all(31);rng=np.random.default_rng(31)
    manifest,manifest_hash,data=load_pose_data(args.data,holdout_asset=args.holdout_asset)
    if args.deformation_loss and any(x['deformations'] is None for x in data):raise ValueError('Deformation loss requires deformation data')
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
        if args.joint_position_loss:
            delta=(logits.softmax(-1)-weights[ids]) @ sample['joints'].to(device)
            loss=loss+args.joint_position_loss*delta.square().sum(-1).mean()
        if args.deformation_loss:
            rest=sample['bind_points'].to(device)[ids]
            matrices=sample['deformations'].to(device)
            transformed=torch.einsum('jab,nb->nja',matrices[:,:3,:3],rest)+matrices[None,:,:3,3]
            delta=torch.einsum('nj,njc->nc',logits.softmax(-1)-weights[ids],transformed)
            loss=loss+args.deformation_loss*delta.square().sum(-1).mean()
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
    output_dir=Path(args.output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    suffix=f'-holdout-{args.holdout_asset}' if args.holdout_asset else ''
    checkpoint=output_dir/f'skin-pose-pilot{suffix}.pt'
    torch.save(dict(version=1,model=best_state,config=config,best_step=best_step,
        source_sha256=args.source_sha256,manifest_sha256=manifest_hash,
        train_assets=sorted(train_by_asset),validation_assets=sorted({s['asset'] for s in validation}),
        rng_state=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all()),checkpoint)
    report=dict(source_sha256=args.source_sha256,manifest_sha256=manifest_hash,
        device=torch.cuda.get_device_name(0),config=config,seed=31,steps=args.steps,
        joint_position_loss=args.joint_position_loss,deformation_loss=args.deformation_loss,
        train_pose_count=len(train),validation_pose_count=len(validation),
        train_assets=sorted(train_by_asset),validation_assets=sorted({s['asset'] for s in validation}),
        scope=manifest['scope'],
        best_step=best_step,best_validation=best,history=history,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),frozen=True)
    # Record initial metrics from the exact pre-training weights (the value is
    # kept separately so reporting does not construct an unrelated model).
    report['initial']=initial
    Path(output_dir/f'skin-pose-training-report{suffix}.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
