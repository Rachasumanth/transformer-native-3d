"""Build pinned upstream O-Voxel and audit its CPU converter on source assets."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from t3d.native_voxelize import UPSTREAM_COMMIT,mesh_to_voxels_native
from t3d.import_glb import load_glb
from t3d.normalization import normalize_asset
from t3d.extraction import voxels_to_mesh
from t3d.metrics import surface_comparison
from t3d.container import save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-sha256',required=True)
    args=parser.parse_args()
    root=Path('/kaggle/temp/t3d-native-upstream')
    env=dict(os.environ,MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.5')
    with open('/kaggle/working/native-build.log','w') as log:
        def run(cmd,timeout=1800):subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
        from tools.build_cuda_dependency import build_and_install
        for name,url,commit in (
            ('CuMesh','https://github.com/JeffreyXiang/CuMesh.git','12289e1062f0603f2f0d0771b02e1395d247f26f'),
            ('FlexGEMM','https://github.com/JeffreyXiang/FlexGEMM.git','6dd94a859c26ee8246888502eada3dd8ad85532e'),
            ('nvdiffrast','https://github.com/NVlabs/nvdiffrast.git','253ac4fcea7de5f396371124af597e6cc957bfae')):
            build_and_install(name,url,commit,log,cache_dir='/kaggle/input/transformer-native-3d-cuda-wheels-cu130')
        # Native extension wheels include PyTorch/CUDA ABI details and cannot be
        # reused across Kaggle image updates. Rebuild pinned O-Voxel source for
        # the runtime just observed, while using hash-verified cu130 dependencies.
        source=root/'o-voxel'
        run(['git','clone','https://github.com/microsoft/TRELLIS.2.git',str(root)])
        run(['git','-C',str(root),'checkout',UPSTREAM_COMMIT])
        run(['git','-C',str(root),'submodule','update','--init','--recursive','o-voxel/third_party/eigen'])
        run([sys.executable,'-m','pip','wheel',str(source),'--no-build-isolation','--no-deps',
             '--wheel-dir','/kaggle/working/native-wheels'])
        native_wheels=list(Path('/kaggle/working/native-wheels').glob('o_voxel-*.whl'))
        if len(native_wheels)!=1:raise RuntimeError('expected one O-Voxel wheel')
        expected=hashlib.sha256(native_wheels[0].read_bytes()).hexdigest()
        run([sys.executable,'-m','pip','install',str(native_wheels[0]),'--no-deps'])
        import platform,torch
        receipt=dict(upstream_commit=UPSTREAM_COMMIT,wheel=native_wheels[0].name,sha256=expected,
                     python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
                     cxx11_abi=torch.compiled_with_cxx11_abi(),architectures='7.5')
        Path('/kaggle/working/native-wheel-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('Built pinned O-Voxel wheel for current runtime',json.dumps(receipt),flush=True)
    import torch
    assert torch.cuda.is_available()
    report=dict(source_sha256=args.source_sha256,upstream_commit=UPSTREAM_COMMIT,
                torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
                native_wheel_sha256=expected,scope='CPU native geometry converter; no material, rig or production-quality claim',assets={})
    for name in ('Avocado','WaterBottle','Lantern'):
        paths=[p for p in Path('/kaggle/input').rglob(f'{name}.glb') if 'transformer-native-3d-source-assets' in p.parts]
        if len(paths)!=1:raise ValueError(f'Expected one {name} source, found {len(paths)}')
        path=paths[0];mesh=load_glb(path)
        vertices,_,_=normalize_asset(mesh.vertices)
        vertices=np.clip(vertices,-.5,.5)
        _,inverse=np.unique(vertices,axis=0,return_inverse=True)
        source_edges=Counter(tuple(sorted((int(a),int(b)))) for tri in inverse[mesh.faces] for a,b in zip(tri,np.roll(tri,-1)))
        item=dict(source_sha256=mesh.source_sha256,source_faces=len(mesh.faces),
                  source_boundary_edges=sum(n==1 for n in source_edges.values()),
                  source_nonmanifold_edges=sum(n>2 for n in source_edges.values()),resolutions={})
        report['assets'][name]=item
        for size in (16,32):
            diagnostics={};start=time.perf_counter()
            asset=mesh_to_voxels_native(vertices,mesh.faces,grid_size=(size,)*3,aabb=[[-.5]*3,[.5]*3],diagnostics=diagnostics)
            seconds=time.perf_counter()-start
            v,f=voxels_to_mesh(asset)
            metric=surface_comparison(vertices,mesh.faces,v,f,samples=256,seed=0)
            out=Path(f'/kaggle/working/native-{name}-{size}.t3d');save(out,asset)
            item['resolutions'][str(size)]=dict(conversion_seconds=seconds,active_voxels=len(asset.coords),
                diagnostics=diagnostics,metrics=metric,output_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
            Path('/kaggle/working/native-converter-report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(name,size,json.dumps(item['resolutions'][str(size)]),flush=True)


if __name__=='__main__':main()
