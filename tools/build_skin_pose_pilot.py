"""Prepare hash-tracked, asset-disjoint SkinAE targets over rig animations."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

from t3d.import_rig import load_rig_glb
from t3d.gltf_accessors import GLBAccessors


def sample_asset(name, split, source, output, *, samples=1024, fractions=(0.,.25,.5,.75,1.)):
    glb=GLBAccessors(source);doc=glb.document
    base,transform=load_rig_glb(source).normalized()
    triangles=base.vertices[base.faces]
    area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
    if not np.isfinite(area).all() or area.sum()<=0:raise ValueError('Invalid source triangle area')
    rng=np.random.default_rng(1700+sum(name.encode()))
    face_ids=rng.choice(len(area),size=samples,p=area/area.sum())
    u,v=rng.random((2,samples));root=np.sqrt(u)
    bary=np.column_stack((1-root,root*(1-v),root*v))
    weights=np.einsum('ni,nij->nj',bary,base.weights[base.faces[face_ids]])
    if not np.allclose(weights.sum(axis=1),1,atol=1e-6,rtol=0):
        raise ValueError('Barycentric skin weights are not normalized')
    clips=[]
    for animation_index,animation in enumerate(doc.get('animations',[])):
        times=[]
        for sampler in animation.get('samplers',[]):
            values=glb.read(sampler['input'])
            if values.ndim!=2 or values.shape[1]!=1 or not len(values):
                raise ValueError('Animation input must contain scalar key times')
            times.extend((float(values[:,0].min()),float(values[:,0].max())))
        if not times:raise ValueError('Animation clip has no key times')
        start,end=min(times),max(times)
        clips.append((animation_index,animation.get('name'),
                      sorted(set(start+(end-start)*float(f) for f in fractions))))
    if not clips:
        clips=[(None,'node-default',[None])]
    pose_points=[];pose_joints=[];pose_clip=[];pose_time=[];clip_rows=[]
    for animation_index,clip_name,times in clips:
        for time in times:
            rig=(load_rig_glb(source) if animation_index is None else
                 load_rig_glb(source,animation_index=animation_index,time=time))
            if not np.array_equal(rig.faces,base.faces) or not np.array_equal(rig.skeleton.parents,base.skeleton.parents):
                raise ValueError('Animation changed topology or joint hierarchy')
            pose_vertices=np.asarray(transform.apply(rig.vertices),dtype=float)
            joint_positions=np.asarray(transform.apply(rig.skeleton.joints),dtype=float)
            sample_triangles=pose_vertices[rig.faces[face_ids]]
            points=np.einsum('ni,nij->nj',bary,sample_triangles)
            if not np.isfinite(points).all() or not np.isfinite(joint_positions).all():
                raise ValueError('Animation produced nonfinite training values')
            pose_points.append(points.astype(np.float32));pose_joints.append(joint_positions.astype(np.float32))
            pose_clip.append(animation_index if animation_index is not None else -1)
            pose_time.append(float(time) if time is not None else 0.)
        clip_rows.append(dict(animation_index=animation_index,name=clip_name,
                              times=[None if t is None else float(t) for t in times]))
    arrays=dict(points=np.stack(pose_points),joints=np.stack(pose_joints),
                parents=base.skeleton.parents,weights=weights.astype(np.float32),
                source_faces=face_ids.astype(np.int64),barycentric=bary.astype(np.float32),
                animation_index=np.asarray(pose_clip,dtype=np.int32),
                animation_time=np.asarray(pose_time,dtype=np.float32))
    path=output/f'{name}-poses.npz';np.savez_compressed(path,**arrays)
    row=dict(name=name,split=split,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),samples=samples,
        poses=len(pose_points),joints=len(base.skeleton.joints),faces=len(base.faces),animations=clip_rows,
        normalization=dict(center=transform.center,scale=transform.scale))
    return row


def main():
    root=Path(__file__).resolve().parents[1];out=root/'artifacts/skin-pose-pilot'
    out.mkdir(exist_ok=True);rows=[]
    for name,split in [('RiggedSimple','train'),('CesiumMan','train'),('Fox','validation')]:
        source=root/'artifacts/rigged-inputs'/name/f'{name}.glb'
        rows.append(sample_asset(name,split,source,out))
        for filename in ('README.md','SOURCE.json'):
            shutil.copyfile(source.parent/filename,out/f'{name}-{filename}')
    shutil.copyfile(root/'artifacts/rigged-inputs/Cesium-LegalMark.txt',out/'Cesium-LegalMark.txt')
    manifest=dict(version=2,scope='Three-asset animated SkinAE integration pilot; one source asset per split; not a representative rig corpus',
        sampling='1024 area-uniform surface samples from node-default pose, barycentric skin weights retained across posed frames; five evenly spaced times per animation; one shared normalization per asset',
        split_policy='RiggedSimple and CesiumMan train; Fox asset-held-out validation; no animation frame crosses source split',assets=rows)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
