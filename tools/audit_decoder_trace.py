"""Measure occupancy decisions at each decoded scale, including pruned parents."""
import numpy as np


def stage_counts(parents,children,source_coords,downsample):
    if downsample not in (1,2,4,8):raise ValueError('invalid downsampling')
    def keys(a):
        a=np.asarray(a)
        if a.ndim!=2 or a.shape[1]!=3:raise ValueError('coordinates must be (N,3)')
        return {tuple(map(int,row)) for row in a}
    parent_set=keys(parents);selected=keys(children)
    source=np.asarray(source_coords)
    truth=keys(source//downsample)
    reachable={c for c in truth if tuple(v//2 for v in c) in parent_set}
    tp=len(selected&truth);fp=len(selected-truth)
    return dict(downsample=downsample,parents=len(parent_set),predicted=len(selected),target=len(truth),
                true_positive=tp,false_positive=fp,false_negative=len(truth-selected),
                missed_unreachable=len(truth-reachable),missed_reachable=len(reachable-selected),
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/len(truth) if truth else None)
