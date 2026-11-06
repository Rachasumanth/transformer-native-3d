"""Two-real-asset training/held-out execution check on Kaggle, not quality evidence."""
import argparse
import json
import hashlib
import numpy as np
from pathlib import Path
import torch
from t3d.dataset import AssetDataset
from t3d.training import VAETrainer
from t3d.reconstruction import reconstruct_asset,encode_latents
from t3d.latent_container import save_latents,load_latents
from t3d.extraction import voxels_to_mesh
from t3d.metrics import surface_comparison
from t3d.container import save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--balance-edges',action='store_true')
    parser.add_argument('--render-weight',type=float,default=0.)
    args=parser.parse_args()
    if args.steps<1:raise ValueError('steps must be positive')
    assert torch.cuda.is_available()
    manifests=[p for p in Path('/kaggle/input').rglob('manifest.json')
               if any(parent.name=='transformer-native-3d-pilot' for parent in p.parents)]
    if len(manifests)!=1:
        raise ValueError(f'Expected one pilot manifest, found {len(manifests)}')
    manifest=manifests[0]
    torch.manual_seed(26)
    report=dict(scope='two licensed integration assets at 16^3; no generalization claim',
                manifest_sha256=None,device=torch.cuda.get_device_name(0),results={})
    models={}
    for mode in ('shape','material'):
        train=AssetDataset(manifest,split='train',require_pbr=mode=='material')
        validation=AssetDataset(manifest,split='validation',require_pbr=mode=='material')
        assert len(train)==1 and len(validation)==1
        edge_weight=None
        if mode=='shape' and args.balance_edges:
            positive=train[0].intersected_flag.sum(axis=0)
            edge_weight=np.clip((len(train[0].coords)-positive)/np.maximum(positive,1),1,20).tolist()
        trainer=VAETrainer(mode=mode,channels=(4,8,16,32,64),latent_channels=8,
                           learning_rate=1e-3,manifest_sha256=train.fingerprint,edge_positive_weight=edge_weight,
                           render_weight=args.render_weight if mode=='shape' else 0.)
        heldout_before=trainer.evaluate([validation[0]])
        history=[]
        checkpoints=[]
        for step in range(1,args.steps+1):
            update=trainer.update([train[0]])
            if step==50:
                trainer.save(Path('/kaggle/working')/f'{mode}-step-50.pt')
            if step in {1,10,50,100,args.steps}:
                history.append(dict(step=step,loss=update))
                checkpoints.append(dict(step=step,train=trainer.evaluate([train[0]]),
                                        validation=trainer.evaluate([validation[0]])))
        checkpoint=Path('/kaggle/working')/f'{mode}-pilot.pt'
        trainer.save(checkpoint)
        restored=VAETrainer.resume(checkpoint,manifest_sha256=train.fingerprint)
        heldout_after=restored.evaluate([validation[0]])
        models[mode]=restored.model
        report['manifest_sha256']=train.fingerprint
        report['results'][mode]=dict(train_asset=train.records[0].asset_id,validation_asset=validation.records[0].asset_id,
                                    train_voxels=len(train[0].coords),validation_voxels=len(validation[0].coords),
                                    steps=restored.steps,edge_positive_weight=edge_weight,render_weight=trainer.render_weight,train=history,evaluations=checkpoints,validation_before=heldout_before,
                                    validation_after=heldout_after)
    report['reconstruction']={}
    for split in ('train','validation'):
        dataset=AssetDataset(manifest,split=split,require_pbr=True)
        source=dataset[0]
        identities={mode:hashlib.sha256(Path(f'/kaggle/working/{mode}-pilot.pt').read_bytes()).hexdigest()
                    for mode in ('shape','material')}
        encoded=encode_latents(source,models['shape'],models['material'],
                    shape_model_sha256=identities['shape'],material_model_sha256=identities['material'])
        latent_path=Path(f'/kaggle/working/{split}-latents.t3d')
        save_latents(latent_path,encoded,curve='h')
        reloaded=load_latents(latent_path,expected_models=identities)
        report.setdefault('latent_storage',{})[split]=dict(tokens=len(reloaded.latents.coords),
                    shape_channels=reloaded.latents.shape.shape[1],material_channels=reloaded.latents.material.shape[1],
                    bytes=latent_path.stat().st_size,checkpoint_sha256=identities)
        reference_v,reference_f=voxels_to_mesh(source)
        for support in ('teacher','predicted'):
            decoded=reconstruct_asset(source,models['shape'],models['material'],support=support)
            vertices,faces=voxels_to_mesh(decoded)
            name=f'{split}-{support}'
            save(Path('/kaggle/working')/f'{name}.t3d',decoded)
            lines=['# Raw VAE reconstruction; face orientation has not been repaired']
            lines.extend('v '+' '.join(map(str,row)) for row in vertices)
            lines.extend('f '+' '.join(str(int(i)+1) for i in row) for row in faces)
            Path(f'/kaggle/working/{name}.obj').write_text('\n'.join(lines)+'\n')
            measurement=surface_comparison(reference_v,reference_f,vertices,faces,samples=256)
            report['reconstruction'][name]=dict(active_voxels=len(decoded.coords),reference_voxels=len(source.coords),
                    latent_support='posterior mean',support=support,metrics=measurement,
                    reference='decoded source .t3d mesh, not original GLB triangles')
    if args.steps>=50:
        early={mode:VAETrainer.resume(Path('/kaggle/working')/f'{mode}-step-50.pt',
                   manifest_sha256=report['manifest_sha256']).model for mode in ('shape','material')}
        report['step50_reconstruction']={}
        for split in ('train','validation'):
            source=AssetDataset(manifest,split=split,require_pbr=True)[0]
            reference_v,reference_f=voxels_to_mesh(source)
            for support in ('teacher','predicted'):
                decoded=reconstruct_asset(source,early['shape'],early['material'],support=support)
                vertices,faces=voxels_to_mesh(decoded)
                name=f'{split}-{support}'
                save(Path('/kaggle/working')/f'step50-{name}.t3d',decoded)
                lines=['# Step 50 VAE reconstruction; face orientation not repaired']
                lines.extend('v '+' '.join(map(str,row)) for row in vertices)
                lines.extend('f '+' '.join(str(int(i)+1) for i in row) for row in faces)
                Path(f'/kaggle/working/step50-{name}.obj').write_text('\n'.join(lines)+'\n')
                report['step50_reconstruction'][name]=dict(active_voxels=len(decoded.coords),
                    metrics=surface_comparison(reference_v,reference_f,vertices,faces,samples=256),
                    reference='decoded source .t3d mesh, not original GLB triangles')
    from tools.audit_edge_predictions import counts
    report['edge_threshold_sweep']={}
    for split in ('train','validation'):
        source=AssetDataset(manifest,split=split,require_pbr=True)[0]
        rv,rf=voxels_to_mesh(source)
        report['edge_threshold_sweep'][split]=[]
        for edge_threshold in (.35,.5,.65,.8):
            decoded=reconstruct_asset(source,models['shape'],support='predicted',threshold=.5,
                                      edge_threshold=edge_threshold)
            v,f=voxels_to_mesh(decoded)
            report['edge_threshold_sweep'][split].append(dict(edge_threshold=edge_threshold,
                occupancy_threshold=.5,active_voxels=len(decoded.coords),edge_counts=counts(source,decoded),
                metrics=surface_comparison(rv,rf,v,f,samples=256)))
    Path('/kaggle/working/corpus-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
