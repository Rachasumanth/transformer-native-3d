"""Compare final occupied voxels and their ancestors across 2x scales."""
import argparse
import json
from pathlib import Path
import numpy as np
from t3d.container import load


def compare(reference,decoded,levels=4):
    if reference.grid_size!=decoded.grid_size or not np.allclose(reference.aabb,decoded.aabb):
        raise ValueError('grid metadata mismatch')
    rows=[]
    for level in range(levels+1):
        def keys(asset):return {tuple(map(int,row)) for row in asset.coords//(2**level)}
        truth,predicted=keys(reference),keys(decoded)
        tp=len(truth&predicted);fp=len(predicted-truth);fn=len(truth-predicted)
        rows.append(dict(downsample=2**level,target=len(truth),predicted=len(predicted),
            true_positive=tp,false_positive=fp,false_negative=fn,
            precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+',type=Path)
    args=parser.parse_args()
    for run in args.runs:
        report={}
        for split,name in (('train','avocado'),('validation','waterbottle')):
            truth=load(Path('artifacts/pilot-dataset')/f'{name}.t3d').asset
            for prefix in ('step50-',''):
                for support in ('teacher','predicted'):
                    key=f'{prefix}{split}-{support}'
                    report[key]=compare(truth,load(run/f'{key}.t3d').asset)
        (run/'support-pyramid-audit.json').write_text(json.dumps(report,indent=2)+'\n')
        print(run,report['validation-predicted'],flush=True)


if __name__=='__main__':main()
