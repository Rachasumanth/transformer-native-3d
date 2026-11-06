"""Compare the custom reference converter with pinned native outputs."""
import hashlib
import json
from pathlib import Path
import numpy as np

from t3d.container import load, save
from t3d.extraction import voxels_to_mesh
from t3d.import_glb import load_glb
from t3d.metrics import surface_comparison
from t3d.normalization import normalize_asset
from t3d.voxelize import mesh_to_voxels


def main():
    root=Path(__file__).resolve().parents[1]
    native=root/'artifacts/kaggle-native-v4'
    inputs=root/'artifacts/native-inputs'
    out=root/'artifacts/native-comparison'
    out.mkdir(parents=True,exist_ok=True)
    report=dict(scope='paired custom CPU and upstream native conversions; sampled metrics, not research benchmark',cases=[])
    for name in ('Avocado','WaterBottle','Lantern'):
        source=load_glb(inputs/f'{name}.glb')
        vertices,_,_=normalize_asset(source.vertices)
        vertices=np.clip(vertices,-.5,.5)
        for size in (16,32):
            diagnostic={}
            custom=mesh_to_voxels(vertices,source.faces,grid_size=(size,)*3,
                                  aabb=[[-.5]*3,[.5]*3],diagnostics=diagnostic)
            custom_path=out/f'custom-{name}-{size}.t3d'
            save(custom_path,custom)
            cv,cf=voxels_to_mesh(custom)
            nv,nf=voxels_to_mesh(load(native/f'native-{name}-{size}.t3d').asset)
            custom_metrics=[surface_comparison(vertices,source.faces,cv,cf,samples=256,seed=s) for s in (0,1,2)]
            native_metrics=[surface_comparison(vertices,source.faces,nv,nf,samples=256,seed=s) for s in (0,1,2)]
            report['cases'].append(dict(asset=name,grid_size=size,source_sha256=source.source_sha256,
                custom_output_sha256=hashlib.sha256(custom_path.read_bytes()).hexdigest(),
                custom_crossing_audit=diagnostic,
                custom=dict(mean_distance=float(np.mean([m['mean_distance'] for m in custom_metrics])),
                            fscores=[m['fscore'] for m in custom_metrics],
                            triangles=len(cf),boundary_edges=custom_metrics[0]['boundary_edges'],
                            nonmanifold_edges=custom_metrics[0]['nonmanifold_edges']),
                native=dict(mean_distance=float(np.mean([m['mean_distance'] for m in native_metrics])),
                            fscores=[m['fscore'] for m in native_metrics],
                            triangles=len(nf),boundary_edges=native_metrics[0]['boundary_edges'],
                            nonmanifold_edges=native_metrics[0]['nonmanifold_edges'])))
            print(name,size,report['cases'][-1]['custom'],report['cases'][-1]['native'],flush=True)
    path=out/'paired-comparison.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Wrote {path}')


if __name__=='__main__':main()
