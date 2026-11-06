"""Build and retain pinned CUDA wheels with runtime and hash receipts on Kaggle."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def compatible_wheel(cache, name, url, commit, runtime):
    """Return a verified wheel/receipt pair, or None for absent/incompatible cache."""
    folder=Path(cache)/name
    receipt_path=folder/'receipt.json'
    if not receipt_path.exists():
        # Kaggle dataset uploads use flat files; raw notebook outputs retain
        # per-dependency directories. Support both explicit receipt layouts.
        folder=Path(cache)
        receipt_path=folder/f'{name}.receipt.json'
    if not receipt_path.exists():return None
    receipt=json.loads(receipt_path.read_text())
    expected=dict(name=name,url=url,commit=commit,**runtime)
    for key,value in expected.items():
        stored=receipt.get(key)
        if key=='python':
            stored='.'.join(str(stored).split('.')[:2])
            value='.'.join(value.split('.')[:2])
        if stored!=value:return None
    filename=receipt.get('wheel','')
    if not isinstance(filename,str) or Path(filename).name!=filename or not filename.endswith('.whl'):
        raise ValueError('invalid cached wheel filename')
    wheel=folder/filename
    if hashlib.sha256(wheel.read_bytes()).hexdigest()!=receipt.get('sha256'):
        raise ValueError('cached wheel checksum mismatch')
    return wheel,receipt


def build_and_install(name, url, commit, log, *, cache_dir='/kaggle/input/transformer-native-3d-cuda-wheels'):
    import torch
    root=Path('/kaggle/working')
    source=root/name
    wheels=root/'dependency-wheels'/name
    wheels.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.5')
    runtime=dict(python=platform.python_version(),torch=str(torch.__version__),
                 cuda=torch.version.cuda,cxx11_abi=torch.compiled_with_cxx11_abi(),
                 architectures=env['TORCH_CUDA_ARCH_LIST'])
    def run(cmd,timeout=1200):
        subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    cached=compatible_wheel(cache_dir,name,url,commit,runtime)
    if cached is not None:
        previous,receipt=cached
        wheel=wheels/previous.name
        shutil.copyfile(previous,wheel)
        run([sys.executable,'-m','pip','install',str(wheel),'--no-deps'])
        receipt=dict(receipt,reused_from=str(previous))
        (wheels/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(f'Reused verified {name} wheel',file=log,flush=True)
        return receipt
    run(['git','clone',url,str(source)],180)
    run(['git','-C',str(source),'checkout',commit],60)
    run(['git','-C',str(source),'submodule','update','--init','--recursive'])
    actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if actual!=commit:raise RuntimeError('dependency checkout identity mismatch')
    run([sys.executable,'-m','pip','wheel',str(source),'--no-build-isolation','--no-deps','--wheel-dir',str(wheels)])
    built=list(wheels.glob('*.whl'))
    if len(built)!=1:raise RuntimeError('expected exactly one dependency wheel')
    wheel=built[0]
    run([sys.executable,'-m','pip','install',str(wheel),'--no-deps'])
    receipt=dict(name=name,url=url,commit=actual,wheel=wheel.name,
        sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),python=platform.python_version(),
        torch=torch.__version__,cuda=torch.version.cuda,cxx11_abi=torch.compiled_with_cxx11_abi(),
        architectures=env['TORCH_CUDA_ARCH_LIST'],max_jobs=env['MAX_JOBS'])
    (wheels/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    # Installed packages and retained wheels suffice; avoid uploading full source
    # checkouts/build trees as thousands of unrelated notebook output files.
    shutil.rmtree(source)
    return receipt
