"""Verify that a checksummed raw container preserves a recoverable rig field."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from t3d.articulation import infer_skeleton_from_field
from t3d.container import load
from t3d.import_rig import load_rig_glb


def audit(container_path, source_glb):
    raw=load(container_path)
    asset=raw.asset
    if asset.skeleton_field is None or asset.skin_embedding is None:
        raise ValueError('Container does not contain both rig feature streams')
    centers=asset.aabb[0]+(asset.coords.astype(float)+.5)*asset.voxel_size
    tolerance=max(float(np.min(asset.voxel_size))*.25,1e-6)
    inferred=infer_skeleton_from_field(centers,asset.skeleton_field,tolerance=tolerance)
    source,_=load_rig_glb(source_glb).normalized()
    if len(inferred.joints)!=len(source.skeleton.joints):
        raise ValueError(f'Inferred {len(inferred.joints)} joints, expected {len(source.skeleton.joints)}')
    distances=np.linalg.norm(inferred.joints[:,None]-source.skeleton.joints[None],axis=-1)
    inferred_to_source=np.argmin(distances,axis=1)
    if len(set(inferred_to_source.tolist()))!=len(inferred_to_source):
        raise ValueError('Recovered joints do not map uniquely to the source rig')
    source_parents=np.full(len(source.skeleton.joints),-2,dtype=int)
    for inferred_id,source_id in enumerate(inferred_to_source):
        parent=int(inferred.parents[inferred_id])
        source_parents[source_id]=-1 if parent<0 else int(inferred_to_source[parent])
    return {
        'container_sha256':hashlib.sha256(Path(container_path).read_bytes()).hexdigest(),
        'source_glb_sha256':hashlib.sha256(Path(source_glb).read_bytes()).hexdigest(),
        'curve':raw.curve,'grid_size':list(asset.grid_size),'active_voxels':len(asset.coords),
        'skeleton_field_shape':list(asset.skeleton_field.shape),
        'skin_embedding_shape':list(asset.skin_embedding.shape),
        'cluster_tolerance':tolerance,'inferred_joint_count':len(inferred.joints),
        'inferred_parent_ids_in_source_order':source_parents.tolist(),
        'source_parent_ids':source.skeleton.parents.tolist(),
        'parent_hierarchy_matches':bool(np.array_equal(source_parents,source.skeleton.parents)),
        'max_joint_position_error':float(np.max(np.min(distances,axis=1))),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('container',type=Path)
    parser.add_argument('source_glb',type=Path)
    parser.add_argument('--json',type=Path)
    args=parser.parse_args()
    report=audit(args.container,args.source_glb)
    rendered=json.dumps(report,indent=2)+'\n'
    if args.json:args.json.write_text(rendered)
    print(rendered,end='')


if __name__=='__main__':main()
