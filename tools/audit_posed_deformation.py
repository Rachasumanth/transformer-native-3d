"""Retarget a source glTF clip onto an exported rig and audit posed deformation.

The exported GLB has reconstructed geometry, SkinAE weights and a permuted but
matching joint hierarchy. This audit maps joints by normalized bind position,
transfers source bind-relative joint motion through the exported weights, and
compares the deformed export surface against the source's own posed surface.
It is a correspondence-based integration check, not a viewer or render test.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np

from t3d.animation import retarget_animation
from t3d.articulation import joint_deformation_matrices, linear_blend_skinning
from t3d.import_rig import load_rig_glb, rig_joint_world
from t3d.metrics import surface_comparison


def _canonical_matrix(transform):
    scale=float(transform.scale)
    matrix=np.eye(4)
    matrix[:3,:3]*=scale
    matrix[:3,3]=-scale*np.asarray(transform.center,dtype=float)
    return matrix


def _nearest_bijection(source_joints, output_joints):
    distances=np.linalg.norm(np.asarray(source_joints)[:,None]-np.asarray(output_joints)[None],axis=2)
    mapping={}
    used=set()
    for source in range(len(distances)):
        for output in np.argsort(distances[source],kind='stable'):
            if int(output) not in used:
                mapping[source]=int(output)
                used.add(int(output))
                break
    if len(mapping)!=len(source_joints):
        raise ValueError('Joint mapping must be one-to-one')
    matched=np.array([distances[source,output] for source,output in mapping.items()])
    return mapping,matched


def _apply(transform, points):
    return np.asarray(transform.apply(points),dtype=float)


def audit_posed_deformation(source, output, *, animation_index=0,
                            fractions=(0.,.25,.5,.75,1.), samples=512,
                            tolerance=.01, animated_path=None):
    source=Path(source);output=Path(output)
    source_bind,source_transform=load_rig_glb(source).normalized()
    output_bind,output_transform=load_rig_glb(output).normalized()
    source_matrix=_canonical_matrix(source_transform)
    source_matrix_inverse=np.linalg.inv(source_matrix)
    source_rest_vertices=np.asarray(source_bind.vertices)
    mapping,matched=_nearest_bijection(source_bind.skeleton.joints,
                                      output_bind.skeleton.joints)
    source_to_output=np.array([mapping[i] for i in range(len(mapping))],dtype=np.int64)
    output_to_source=np.empty_like(source_to_output)
    output_to_source[source_to_output]=np.arange(len(source_to_output))
    parent_expected=np.where(source_bind.skeleton.parents<0,-1,
                             source_to_output[np.maximum(source_bind.skeleton.parents,0)])
    parent_mismatches=int(np.count_nonzero(
        output_bind.skeleton.parents[source_to_output]!=parent_expected))

    _,source_bind_world=rig_joint_world(source)
    output_faces=np.asarray(output_bind.faces)
    output_vertices=np.asarray(output_bind.vertices)
    output_weights=np.asarray(output_bind.weights)

    # Source weights projected onto the export surface are an oracle control that
    # separates retarget-math error from SkinAE weight error.
    from t3d.proximity import TriangleBVH
    projected,bary=TriangleBVH(source_rest_vertices,source_bind.faces).query(output_vertices)
    source_weights_at_output=np.einsum(
        'ni,nij->nj',bary,source_bind.weights[source_bind.faces[projected]])
    source_weights_at_output=source_weights_at_output[:,output_to_source]
    weight_delta=np.abs(source_weights_at_output-output_weights)

    # Sampler ranges are shared by all channels; sample the union of key ranges.
    from t3d.gltf_accessors import GLBAccessors
    glb=GLBAccessors(source)
    doc=glb.document
    animations=doc.get('animations',[])
    if not 0<=animation_index<len(animations):raise ValueError('Invalid animation index')
    animation=animations[animation_index]
    keyed=[]
    for sampler in animation.get('samplers',[]):
        values=glb.read(sampler['input'])
        keyed.append((float(np.min(values)),float(np.max(values))))
    start,end=min(k[0] for k in keyed),max(k[1] for k in keyed)
    times=sorted(set(start+(end-start)*float(f) for f in fractions))

    rest_metrics=surface_comparison(
        source_rest_vertices,source_bind.faces,
        output_vertices,output_faces,samples=samples,tolerance=tolerance)
    poses=[]
    reference_by_time={}
    deformed_by_time={}
    for time in times:
        posed=load_rig_glb(source,animation_index=animation_index,time=time)
        _,pose_world=rig_joint_world(source,animation_index=animation_index,time=time)
        reference_vertices=_apply(source_transform,np.asarray(posed.vertices))
        reference_faces=np.asarray(posed.faces)
        if reference_vertices.shape!=source_rest_vertices.shape:
            raise ValueError('Animation changed source vertex count')
        deformation=joint_deformation_matrices(source_bind_world,pose_world)
        deformation=source_matrix@deformation@source_matrix_inverse
        # Output joint j receives the motion of its matched source joint.
        output_deformation=deformation[output_to_source]
        deformed=linear_blend_skinning(output_vertices,output_weights,output_deformation)
        oracle=linear_blend_skinning(output_vertices,source_weights_at_output,output_deformation)
        reference_by_time[time]=reference_vertices
        deformed_by_time[time]=deformed
        if not np.isfinite(deformed).all() or not np.isfinite(oracle).all():
            raise ValueError('Retargeted animation produced nonfinite vertices')
        metrics=surface_comparison(reference_vertices,reference_faces,deformed,output_faces,
                                   samples=samples,tolerance=tolerance)
        oracle_metrics=surface_comparison(reference_vertices,reference_faces,oracle,output_faces,
                                          samples=samples,tolerance=tolerance)
        rest_at_time=surface_comparison(reference_vertices,reference_faces,
                                        output_vertices,output_faces,
                                        samples=samples,tolerance=tolerance)
        source_motion=np.linalg.norm(
            reference_vertices-source_rest_vertices,axis=1)
        output_motion=np.linalg.norm(deformed-output_vertices,axis=1)
        poses.append(dict(time=time,
            source_vertex_motion_mean=float(source_motion.mean()),
            source_vertex_motion_max=float(source_motion.max()),
            retargeted_vertex_motion_mean=float(output_motion.mean()),
            retargeted_vertex_motion_max=float(output_motion.max()),
            retargeted_surface=metrics,
            oracle_weight_surface=oracle_metrics,
            rest_pose_surface=rest_at_time,
            mean_distance_improvement=float((rest_at_time['mean_distance']
                                             -metrics['mean_distance'])
                                            if rest_at_time.get('valid') and metrics.get('valid') else 0.)))
        poses[-1].pop('rest_pose_surface')
        poses[-1]['rest_pose_mean_distance']=rest_at_time.get('mean_distance')
        poses[-1]['rest_pose_fscore']=rest_at_time.get('fscore')

    animated,animation_audit=retarget_animation(output,source,animation_index=animation_index)
    roundtrip=[]
    with tempfile.TemporaryDirectory() as directory:
        if animated_path is None:
            reload_path=Path(directory)/'animated.glb'
            reload_path.write_bytes(animated)
        else:
            reload_path=Path(animated_path)
            reload_path.parent.mkdir(parents=True,exist_ok=True)
            reload_path.write_bytes(animated)
        animated_sha256=hashlib.sha256(reload_path.read_bytes()).hexdigest()
        for time in times:
            rig=load_rig_glb(reload_path,animation_index=animation_audit['output_animation_index'],
                             time=time)
            vertices=np.asarray(output_transform.apply(rig.vertices),dtype=float)
            metrics=surface_comparison(reference_by_time[time],source_bind.faces,vertices,
                                       rig.faces,samples=samples,tolerance=tolerance)
            roundtrip.append(dict(time=time,surface=metrics,
                max_vertex_difference_from_in_memory=float(
                    np.max(np.abs(vertices-deformed_by_time[time])))))
    valid=all(pose['retargeted_surface'].get('valid') for pose in poses)
    result=dict(
        scope=('Source clip retargeted through exported SkinAE weights and a '
               'position-matched joint mapping; correspondence-based surface '
               'comparison, not viewer or rendered deformation quality'),
        source=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
        output=dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    vertices=int(len(output_vertices)),faces=int(len(output_faces)),
                    joints=int(len(output_bind.skeleton.joints))),
        animation=dict(index=animation_index,name=animation.get('name'),
                       time_range=[start,end],fractions=list(fractions),samples=samples,
                       tolerance=tolerance),
        weight_comparison=dict(
            skin_ae_vs_source_transfer_mean_absolute=float(weight_delta.mean()),
            skin_ae_vs_source_transfer_max_absolute=float(weight_delta.max()),
            dominant_joint_agreement=float(np.mean(
                np.argmax(output_weights,axis=1)==np.argmax(source_weights_at_output,axis=1)))),
        joint_mapping=dict(source_to_output=source_to_output.tolist(),
                           output_to_source=output_to_source.tolist(),
                           matched_position_distance_max=float(matched.max()),
                           matched_position_distance_mean=float(matched.mean()),
                           parent_mismatches=parent_mismatches),
        rest_pose=dict(surface=rest_metrics),
        exported_animation=dict(path=None if animated_path is None else str(animated_path),
                                sha256=animated_sha256,audit=animation_audit,
                                roundtrip=roundtrip),
        poses=poses,
        valid=valid)
    if valid:
        result['summary']=dict(
            retargeted_mean_distance=float(np.mean(
                [p['retargeted_surface']['mean_distance'] for p in poses])),
            retargeted_fscore=float(np.mean(
                [p['retargeted_surface']['fscore'] for p in poses])),
            oracle_weight_mean_distance=float(np.mean(
                [p['oracle_weight_surface']['mean_distance'] for p in poses])),
            oracle_weight_fscore=float(np.mean(
                [p['oracle_weight_surface']['fscore'] for p in poses])),
            rest_mean_distance=float(np.mean(
                [p['rest_pose_mean_distance'] for p in poses])),
            mean_distance_improvement=float(np.mean(
                [p['mean_distance_improvement'] for p in poses])),
            max_export_roundtrip_vertex_difference=float(np.max(
                [r['max_vertex_difference_from_in_memory'] for r in roundtrip])),
            roundtrip_invalid_times=[r['time'] for r in roundtrip
                                     if not r['surface'].get('valid')])
        if not result['summary']['roundtrip_invalid_times']:
            result['summary']['roundtrip_mean_distance']=float(np.mean(
                [r['surface']['mean_distance'] for r in roundtrip]))
            result['summary']['roundtrip_fscore']=float(np.mean(
                [r['surface']['fscore'] for r in roundtrip]))
        else:
            result['valid']=False
    return result


def main():
    root=Path(__file__).resolve().parents[1]
    destination=root/'artifacts/posed-deformation'
    destination.mkdir(parents=True,exist_ok=True)
    rigged=root/'artifacts/rigged-inputs'
    roundtrip=rigged/'glb-export-roundtrip'
    cases=[
        dict(label='cesiumman-v13',
             source=rigged/'CesiumMan/CesiumMan.glb',
             output=root/'artifacts/kaggle-textured-rig-export-v13/export-real-rigged.glb',
             animation_index=0),
        dict(label='cesiumman-v13-top1',
             source=rigged/'CesiumMan/CesiumMan.glb',
             output=destination/'cesiumman-v13-top1.glb',animation_index=0),
        dict(label='cesiumman-v13-top2',
             source=rigged/'CesiumMan/CesiumMan.glb',
             output=destination/'cesiumman-v13-top2.glb',animation_index=0),
        dict(label='cesiumman-v13-top3',
             source=rigged/'CesiumMan/CesiumMan.glb',
             output=destination/'cesiumman-v13-top3.glb',animation_index=0),
        dict(label='cesiumman-v13-top4',
             source=rigged/'CesiumMan/CesiumMan.glb',
             output=destination/'cesiumman-v13-top4.glb',animation_index=0),
        dict(label='riggedsimple-roundtrip',
             source=rigged/'RiggedSimple/RiggedSimple.glb',
             output=roundtrip/'RiggedSimple.glb',animation_index=0),
        dict(label='cesiumman-roundtrip',
             source=rigged/'CesiumMan/CesiumMan.glb',
             output=roundtrip/'CesiumMan.glb',animation_index=0),
        dict(label='fox-survey',
             source=rigged/'Fox/Fox.glb',output=roundtrip/'Fox.glb',
             animation_index=0),
        dict(label='fox-walk',
             source=rigged/'Fox/Fox.glb',output=roundtrip/'Fox.glb',
             animation_index=1),
        dict(label='fox-run',
             source=rigged/'Fox/Fox.glb',output=roundtrip/'Fox.glb',
             animation_index=2),
    ]
    for case in cases:
        label=case['label']
        result=audit_posed_deformation(
            case['source'],case['output'],animation_index=case['animation_index'],
            animated_path=destination/f'{label}-animated.glb')
        result['case']=label
        path=destination/f'{label}-posed-deformation.json'
        path.write_text(json.dumps(result,indent=2)+'\n')
        print(f'Wrote {path}')
        print(label,'summary',result.get('summary'),flush=True)
        print('export',result['exported_animation']['sha256'],
              'roundtrip',result['summary'].get('roundtrip_mean_distance'),
              'max_vertex_delta',result['summary']['max_export_roundtrip_vertex_difference'],
              'invalid_roundtrip_times',result['summary']['roundtrip_invalid_times'],
              flush=True)


if __name__=='__main__':main()
