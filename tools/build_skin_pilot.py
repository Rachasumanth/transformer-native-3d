"""Prepare area-sampled rig skinning targets and retained attribution records."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from t3d.import_rig import load_rig_glb


def main():
    root=Path(__file__).resolve().parents[1];out=root/'artifacts/skin-pilot';out.mkdir(exist_ok=True)
    rows=[]
    for name,split in [('RiggedSimple','train'),('CesiumMan','train'),('Fox','validation')]:
        source=root/'artifacts/rigged-inputs'/name
        rig,transform=load_rig_glb(source/f'{name}.glb').normalized()
        triangles=rig.vertices[rig.faces]
        areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
        if not np.isfinite(areas).all() or areas.sum()<=0:raise ValueError('Invalid source triangle area')
        rng=np.random.default_rng(17)
        ids=rng.choice(len(areas),size=1024,p=areas/areas.sum())
        u,v=rng.random((2,1024));r=np.sqrt(u)
        bary=np.column_stack((1-r,r*(1-v),r*v))
        points=np.einsum('ni,nij->nj',bary,triangles[ids])
        weights=np.einsum('ni,nij->nj',bary,rig.weights[rig.faces[ids]])
        path=out/f'{name}.npz'
        np.savez_compressed(path,points=points.astype('f4'),weights=weights.astype('f4'),
                            joints=rig.skeleton.joints.astype('f4'),parents=rig.skeleton.parents,
                            source_faces=ids,barycentric=bary.astype('f4'))
        rows.append(dict(name=name,split=split,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         source_sha256=rig.source_sha256,samples=len(points),joints=len(rig.skeleton.joints),
                         maximum_bind_palette_identity_delta=rig.maximum_bind_palette_identity_delta,
                         normalization=dict(center=transform.center,scale=transform.scale)))
        for filename in ('README.md','SOURCE.json'):
            shutil.copyfile(source/filename,out/f'{name}-{filename}')
    shutil.copyfile(root/'artifacts/rigged-inputs/Cesium-LegalMark.txt',out/'Cesium-LegalMark.txt')
    manifest=dict(scope='three-asset SkinAE integration pilot; node-default poses only; no research quality claim',
                  sampling='1024 area-uniform triangle samples per asset; barycentric skin weights; seed 17',assets=rows)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
