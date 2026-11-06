"""Evaluate held-out SkinAE checkpoints using actual animated vertex error.

CPU evaluation only; training remains on Kaggle. Usage:
python -m tools.audit_skin_pose_checkpoints DOWNLOADED_OUTPUT_DIRECTORY
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from t3d.skin_ae import SkinAE
from t3d.import_rig import load_rig_glb, rig_joint_world
from t3d.articulation import joint_deformation_matrices, linear_blend_skinning
from t3d.gltf_accessors import GLBAccessors


def audit(checkpoint, asset, *, source=None, external=False, frames=5):
    source=Path(source) if source else Path('artifacts/rigged-inputs')/asset/f'{asset}.glb'
    rig, transform=load_rig_glb(source).normalized()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    if external:
        manifest=json.loads(Path('artifacts/skin-pose-pilot/manifest.json').read_text())
        known={row['name']:row['source_sha256'] for row in manifest['assets']}
        if any(name not in known for name in saved['train_assets']):
            raise ValueError('Cannot verify all training asset hashes')
        if any(known[name]==source_hash for name in saved['train_assets']):
            raise ValueError('External source matches a training asset by hash')
    if asset in saved['train_assets'] or (not external and asset not in saved['validation_assets']):
        raise ValueError('Checkpoint must hold out the evaluated source asset')
    model=SkinAE(**saved['config']);model.load_state_dict(saved['model']);model.freeze()
    with torch.no_grad():
        _,_,logits=model(torch.tensor(rig.skeleton.joints,dtype=torch.float32),
            torch.tensor(rig.skeleton.parents,dtype=torch.int64),
            torch.tensor(rig.weights,dtype=torch.float32))
        weights=logits.softmax(-1).numpy()
    canonical=np.eye(4);canonical[:3,:3]*=transform.scale
    canonical[:3,3]=-transform.scale*np.asarray(transform.center)
    _,bind_world=rig_joint_world(source)
    glb=GLBAccessors(source);poses=[]
    for clip,animation in enumerate(glb.document.get('animations',[])):
        keys=[glb.read(s['input']).reshape(-1) for s in animation['samplers']]
        for time in np.linspace(min(k.min() for k in keys),max(k.max() for k in keys),frames):
            _,world=rig_joint_world(source,animation_index=clip,time=float(time))
            matrices=canonical@joint_deformation_matrices(bind_world,world)@np.linalg.inv(canonical)
            target=linear_blend_skinning(rig.vertices,rig.weights,matrices)
            predicted=linear_blend_skinning(rig.vertices,weights,matrices)
            error=np.linalg.norm(predicted-target,axis=1)
            poses.append(dict(clip=clip,time=float(time),mean=float(error.mean()),
                p95=float(np.quantile(error,.95)),maximum=float(error.max())))
    return dict(asset=asset,source_sha256=source_hash,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        scope='Held-out source bind weights encoded and decoded by SkinAE; actual LBS vertex error, not generated geometry or source-free rig prediction',
        weight_mae=float(np.abs(weights-rig.weights).mean()),poses=poses,
        mean_vertex_error=float(np.mean([p['mean'] for p in poses])))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--frames',type=int,default=5);parser.add_argument('--external-asset');parser.add_argument('--external-source',type=Path);args=parser.parse_args()
    if args.frames < 2:parser.error('--frames must be at least two')
    if bool(args.external_asset)!=bool(args.external_source):parser.error('Provide both --external-asset and --external-source')
    torch.set_num_threads(2)
    rows=[]
    for checkpoint in sorted(args.directory.rglob('skin-pose-pilot-holdout-*.pt')):
        asset=args.external_asset or checkpoint.stem.removeprefix('skin-pose-pilot-holdout-')
        row=audit(checkpoint,asset,source=args.external_source,external=bool(args.external_asset),frames=args.frames);rows.append(row);print(json.dumps(row),flush=True)
    if not rows:raise ValueError('No held-out checkpoints found')
    (args.directory/(f'external-{args.external_asset}-deformation-audit-{args.frames}frames.json' if args.external_asset and args.frames != 5 else f'external-{args.external_asset}-deformation-audit.json' if args.external_asset else 'deformation-audit.json')).write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
