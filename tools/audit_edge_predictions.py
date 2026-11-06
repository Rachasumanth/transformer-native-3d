"""Compare decoded per-axis edge flags against prepared voxel targets."""
import argparse
import json
from pathlib import Path
import numpy as np
from t3d.container import load


def counts(reference,decoded):
    if reference.grid_size!=decoded.grid_size or not np.allclose(reference.aabb,decoded.aabb):
        raise ValueError('grid metadata mismatch')
    def edges(asset,axis):
        return {tuple(map(int,row)) for row in asset.coords[asset.intersected_flag[:,axis]]}
    reference_support={tuple(map(int,row)) for row in reference.coords}
    predicted_support={tuple(map(int,row)) for row in decoded.coords}
    result=[]
    for axis in range(3):
        target,predicted=edges(reference,axis),edges(decoded,axis)
        tp=len(target&predicted);fp=len(predicted-target);fn=len(target-predicted)
        false_positive=predicted-target
        false_negative=target-predicted
        result.append(dict(axis=axis,true_positive=tp,false_positive=fp,false_negative=fn,
            false_positive_on_reference_support=len(false_positive&reference_support),
            false_positive_outside_reference_support=len(false_positive-reference_support),
            false_negative_on_predicted_support=len(false_negative&predicted_support),
            false_negative_missing_support=len(false_negative-predicted_support),
            precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--dataset',type=Path,default=Path('artifacts/pilot-dataset'))
    args=parser.parse_args()
    result={}
    for split,name in (('train','avocado'),('validation','waterbottle')):
        reference=load(args.dataset/f'{name}.t3d').asset
        for prefix in ('step50-',''):
            for support in ('teacher','predicted'):
                key=f'{prefix}{split}-{support}'
                result[key]=counts(reference,load(args.run/f'{key}.t3d').asset)
    (args.run/'edge-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
