"""Audit decoder occupancy logits and actual free-running pruning on GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from t3d.dataset import AssetDataset,batch_arrays
from t3d.training import VAETrainer
from tools.audit_decoder_trace import stage_counts

CHECKPOINT_SHA256={
    20:'81c3b0145972e3da6d518c260991ad2080487d4e31397367fedc754e4aa5de8e',
    24:'50eb25502ab09929a96f9498d0eeac79c81553d40fd8ea25a2939cf1d6d94cc2',
}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-sha256',required=True)
    args=parser.parse_args()
    assert torch.cuda.is_available()
    def locate(filename,dataset):
        paths=[p for p in Path('/kaggle/input').rglob(filename) if dataset in p.parts]
        if len(paths)!=1:raise ValueError(f'Expected one {filename} in {dataset}, found {len(paths)}')
        return paths[0]
    data=locate('manifest.json','transformer-native-3d-pilot')
    report=dict(source_sha256=args.source_sha256,device=torch.cuda.get_device_name(0),
                scope='two-asset pilot; posterior-mean reconstruction; no quality claim',models={})
    for version in (20,24):
        path=locate(f'v{version}-shape.pt','transformer-native-3d-trace-checkpoints')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=CHECKPOINT_SHA256[version]:raise ValueError('checkpoint identity mismatch')
        fingerprint=AssetDataset(data,split='train').fingerprint
        model=VAETrainer.resume(path,manifest_sha256=fingerprint).model.eval()
        row=dict(checkpoint_sha256=digest,assets={})
        report['models'][str(version)]=row
        for split in ('train','validation'):
            asset=AssetDataset(data,split=split)[0]
            coords,features=batch_arrays([asset],'shape')
            with torch.no_grad():
                posterior=model.encode(torch.from_numpy(coords).cuda(),torch.from_numpy(features).cuda())
                free=[];teacher=[]
                model.decode(posterior.coords,posterior.mean,trace=free)
                model.decode(posterior.coords,posterior.mean,teacher_occupancy=posterior.occupancy,trace=teacher)
            stages=[]
            for i,(pred,teach) in enumerate(zip(free,teacher)):
                downsample=2**(3-i)
                counts=stage_counts(pred['parents'][:,1:].cpu().numpy(),pred['children'][:,1:].cpu().numpy(),asset.coords,downsample)
                expected=teach['selected'];selected=teach['logits'].sigmoid()>.5
                counts['teacher_context']=dict(true_positive=int((expected&selected).sum()),
                    false_positive=int((~expected&selected).sum()),false_negative=int((expected&~selected).sum()),
                    parents=len(teach['parents']))
                stages.append(counts)
            row['assets'][split]=stages
    Path('/kaggle/working/decoder-trace-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
