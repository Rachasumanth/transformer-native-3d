"""Sample core glTF rig animations and audit mesh/joint motion invariants."""
import hashlib
import json
from pathlib import Path
import numpy as np

from t3d.gltf_accessors import GLBAccessors
from t3d.import_rig import load_rig_glb


def edge_topology(faces):
    edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    edges.sort(axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True)
    return dict(boundary_edges=int(np.count_nonzero(counts==1)),
                nonmanifold_edges=int(np.count_nonzero(counts>2)))


def audit_asset(path, *, fractions=(0.,.25,.5,.75,1.)):
    path=Path(path);glb=GLBAccessors(path);doc=glb.document
    animations=doc.get('animations',[])
    source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    base,transform=load_rig_glb(path).normalized()
    result=dict(source=str(path),source_sha256=source_hash,
                geometry=dict(vertices=len(base.vertices),faces=len(base.faces),
                              joints=len(base.skeleton.joints),topology=edge_topology(base.faces)),
                animations=[])
    for animation_index,animation in enumerate(animations):
        keyed=[]
        for sampler in animation.get('samplers',[]):
            values=glb.read(sampler['input'])
            if values.ndim!=2 or values.shape[1]!=1 or not len(values):
                raise ValueError('Animation input must contain scalar times')
            keyed.extend((float(values[:,0].min()),float(values[:,0].max())))
        if not keyed:raise ValueError('Animation has no keyed time range')
        start,end=min(keyed),max(keyed)
        times=sorted(set(start+(end-start)*float(f) for f in fractions))
        poses=[load_rig_glb(path,animation_index=animation_index,time=t) for t in times]
        if any(p.vertices.shape!=poses[0].vertices.shape or p.faces.shape!=poses[0].faces.shape
               or p.skeleton.parents.shape!=poses[0].skeleton.parents.shape for p in poses):
            raise ValueError('Animated pose changed the rig topology or counts')
        ref_vertices=np.asarray(transform.apply(poses[0].vertices),dtype=float)
        ref_joints=np.asarray(transform.apply(poses[0].skeleton.joints),dtype=float)
        parents=poses[0].skeleton.parents
        root_lengths=np.linalg.norm(ref_joints-ref_joints[np.maximum(parents,0)],axis=1)
        root_lengths[parents<0]=0
        pose_rows=[]
        for time,pose in zip(times,poses):
            vertices=np.asarray(transform.apply(pose.vertices),dtype=float)
            joints=np.asarray(transform.apply(pose.skeleton.joints),dtype=float)
            if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
                raise ValueError('Animation produced nonfinite geometry')
            displacement=np.linalg.norm(vertices-ref_vertices,axis=1)
            joint_displacement=np.linalg.norm(joints-ref_joints,axis=1)
            lengths=np.linalg.norm(joints-joints[np.maximum(parents,0)],axis=1)
            valid=parents>=0
            ratios=lengths[valid]/np.maximum(root_lengths[valid],1e-12)
            pose_rows.append(dict(time=time,
                vertex_displacement_mean_from_first=float(displacement.mean()),
                vertex_displacement_max_from_first=float(displacement.max()),
                joint_displacement_max_from_first=float(joint_displacement.max()),
                bone_length_ratio_min=float(ratios.min()) if len(ratios) else 1.,
                bone_length_ratio_max=float(ratios.max()) if len(ratios) else 1.,
                bounds_min=vertices.min(axis=0).tolist(),bounds_max=vertices.max(axis=0).tolist(),
                topology=edge_topology(pose.faces)))
        channel_paths=sorted({channel.get('target',{}).get('path','unknown')
                              for channel in animation.get('channels',[])})
        interpolation=sorted({s.get('interpolation','LINEAR') for s in animation.get('samplers',[])})
        result['animations'].append(dict(index=animation_index,name=animation.get('name'),
            time_range=[start,end],channel_paths=channel_paths,interpolation=interpolation,
            pose_count=len(pose_rows),poses=pose_rows))
    return result


def main():
    root=Path(__file__).resolve().parents[1]
    cases=[]
    for name in ('RiggedSimple','CesiumMan','Fox'):
        path=root/'artifacts/rigged-inputs'/name/f'{name}.glb'
        cases.append(dict(asset=name,**audit_asset(path)))
    report=dict(scope='Core glTF animation sampling and linear-blend skinning execution audit; not an external deformation-quality benchmark',
        sampling='five evenly spaced times per animation over the union of sampler key ranges; shared node-default normalization per source asset',
        cases=cases)
    output=root/'artifacts/rigged-inputs/animation-audit.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Wrote {output}')
    for case in cases:
        print(case['asset'],[(a['name'],a['pose_count']) for a in case['animations']],flush=True)


if __name__=='__main__':main()
