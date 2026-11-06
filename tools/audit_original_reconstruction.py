"""Compare pilot outputs directly with the normalized source GLB surfaces."""
import hashlib
import json
from pathlib import Path
import numpy as np
from t3d.import_glb import load_glb
from t3d.normalization import normalize_asset
from t3d.container import load
from t3d.extraction import voxels_to_mesh
from t3d.metrics import surface_comparison


def main():
    root=Path(__file__).resolve().parents[1]
    report=dict(scope='two sample assets; unsquared point-to-triangle metric; not research benchmark',
                run_source_sha256={str(v):json.loads((root/f'artifacts/kaggle-v{v}/gpu-report.json').read_text())['source_sha256'] for v in (20,24)},assets={})
    for split,name,filename in [('train','avocado','Avocado.glb'),('validation','waterbottle','WaterBottle.glb')]:
        source=load_glb(root/'artifacts/import'/name/filename)
        ref,_,_=normalize_asset(source.vertices)
        paths={'prepared':root/'artifacts/pilot-dataset'/f'{name}.t3d'}
        for version in (20,24):
            for support in ('teacher','predicted'):
                paths[f'v{version}-{support}']=root/f'artifacts/kaggle-v{version}/{split}-{support}.t3d'
        results={}
        for label,path in paths.items():
            asset=load(path).asset
            if not np.allclose(asset.aabb,[[-.5]*3,[.5]*3],rtol=0,atol=1e-6):
                raise ValueError(f'{path} does not use the expected canonical bounds')
            vertices,faces=voxels_to_mesh(asset)
            measurements=[surface_comparison(ref,source.faces,vertices,faces,samples=256,seed=seed) for seed in (0,1,2)]
            results[label]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),measurements=measurements)
            print(split,label,[(m.get('fscore'),m.get('mean_distance')) for m in measurements],flush=True)
        report['assets'][split]=dict(source_sha256=source.source_sha256,source_faces=len(source.faces),results=results)
    out=root/'artifacts/import/original-reconstruction-audit.json'
    out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
