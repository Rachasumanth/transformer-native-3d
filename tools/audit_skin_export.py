"""Compare exported skin weights with source transfer and a nearest-joint baseline."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from t3d.articulation import transfer_skin_weights
from t3d.import_rig import load_rig_glb


def _metrics(predicted, target):
    predicted=np.asarray(predicted,dtype=float)
    target=np.asarray(target,dtype=float)
    if predicted.shape!=target.shape or predicted.ndim!=2:
        raise ValueError('Predicted and target weights must have the same (N,J) shape')
    eps=1e-8
    kl=float(np.mean(np.sum(np.where(target>0,target*np.log((target+eps)/(predicted+eps)),0),axis=1)))
    return {
        'weight_mae':float(np.mean(np.abs(predicted-target))),
        'weight_kl':kl,
        'dominant_joint_accuracy':float(np.mean(np.argmax(predicted,axis=1)==np.argmax(target,axis=1))),
    }


def audit(source_path, output_path):
    source,transform=load_rig_glb(source_path).normalized()
    output=load_rig_glb(output_path)
    if len(output.skeleton.joints)!=len(source.skeleton.joints):
        raise ValueError('Source and output skeleton joint counts differ')
    distances=np.linalg.norm(output.skeleton.joints[:,None]-source.skeleton.joints[None],axis=-1)
    output_to_source=np.argmin(distances,axis=1)
    if len(set(output_to_source.tolist()))!=len(output_to_source):
        raise ValueError('Output joints cannot be matched uniquely to source joints')
    aligned=np.zeros_like(output.weights)
    for output_id,source_id in enumerate(output_to_source):
        aligned[:,source_id]=output.weights[:,output_id]
    target=transfer_skin_weights(output.vertices,source.vertices,source.faces,source.weights)
    nearest=np.argmin(np.linalg.norm(output.vertices[:,None]-source.skeleton.joints[None],axis=-1),axis=1)
    nearest_weights=np.eye(len(source.skeleton.joints),dtype=float)[nearest]
    return {
        'source_sha256':hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        'output_sha256':hashlib.sha256(Path(output_path).read_bytes()).hexdigest(),
        'source_vertices':len(source.vertices),'source_faces':len(source.faces),
        'output_vertices':len(output.vertices),'output_faces':len(output.faces),
        'joint_count':len(source.skeleton.joints),
        'normalization_scale':float(transform.scale),
        'sample_count':len(output.vertices),
        'output_to_source_joint_ids':output_to_source.tolist(),
        'skinae':_metrics(aligned,target),
        'nearest_joint_baseline':_metrics(nearest_weights,target),
        'target':'Source skin weights barycentrically projected from the closest source triangle to each output vertex.',
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_glb',type=Path)
    parser.add_argument('output_glb',type=Path)
    parser.add_argument('--json',type=Path)
    args=parser.parse_args()
    report=audit(args.source_glb,args.output_glb)
    rendered=json.dumps(report,indent=2)+'\n'
    if args.json:args.json.write_text(rendered)
    print(rendered,end='')


if __name__=='__main__':main()
