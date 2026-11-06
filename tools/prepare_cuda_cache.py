"""Prepare a private Kaggle wheel-cache dataset from inspected notebook outputs."""
import argparse
import json
from pathlib import Path
import shutil
from tools.build_cuda_dependency import compatible_wheel
from tools.probe_cumesh import COMMIT
from tools.probe_export import PINS

EXPECTED={'CuMesh':('https://github.com/JeffreyXiang/CuMesh.git',COMMIT),**PINS}
RUNTIME_KEYS=('python','torch','cuda','cxx11_abi','architectures')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args()
    if args.destination.exists() and any(args.destination.iterdir()):
        raise ValueError('destination must be empty')
    verified=[]
    runtime=None
    for name,(url,commit) in EXPECTED.items():
        path=args.source/name/'receipt.json'
        if not path.exists():
            if args.allow_partial:continue
            raise ValueError(f'missing dependency receipt: {name}')
        receipt=json.loads(path.read_text())
        current={key:receipt[key] for key in RUNTIME_KEYS}
        if runtime is not None and runtime!=current:
            raise ValueError('dependency runtime receipts disagree')
        runtime=current
        result=compatible_wheel(args.source,name,url,commit,current)
        if result is None:raise ValueError(f'source identity mismatch for {name}')
        verified.append((name,*result))
    if not verified:raise ValueError('no verified wheels available')
    args.destination.mkdir(parents=True,exist_ok=True)
    manifests=[]
    for name,wheel,receipt in verified:
        shutil.copyfile(wheel,args.destination/wheel.name)
        (args.destination/f'{name}.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        manifests.append(dict(dependency=name,commit=receipt['commit'],wheel=wheel.name,sha256=receipt['sha256']))
    metadata=dict(title='Transformer Native 3D CUDA Wheels',id='sumu143/transformer-native-3d-cuda-wheels',
                  licenses=[{'name':'other'}])
    (args.destination/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (args.destination/'cache-manifest.json').write_text(json.dumps(dict(runtime=runtime,wheels=manifests),indent=2)+'\n')
    (args.destination/'README.md').write_text(
        'Private build cache for the transformer-native 3D project.\n\n'
        'Compiled artifacts retain their upstream licensing; package license files are embedded in wheels. '
        'Sources and exact revisions are recorded in each receipt. This cache is for the recorded '
        'Python/PyTorch/CUDA ABI and sm_75 runtime; the notebook checks identity and hashes before installation.\n')
    print(json.dumps(dict(destination=str(args.destination),wheels=manifests),indent=2))


if __name__=='__main__':main()
