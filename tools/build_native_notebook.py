"""Create a private Kaggle notebook for the pinned upstream converter audit."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

from tools.notebook_bundle import source_bundle

ROOT=Path(__file__).resolve().parents[1]

def main():
    files=sorted([*ROOT.glob('t3d/*.py'),ROOT/'tools/probe_native_converter.py',ROOT/'tools/build_cuda_dependency.py',ROOT/'pyproject.toml'])
    payload = source_bundle(ROOT, files)
    digest = hashlib.sha256(payload).hexdigest()
    setup=('import base64,hashlib,io,zipfile,pathlib,subprocess,sys\n'
        f'payload=base64.b64decode({base64.b64encode(payload).decode()!r})\n'
        f'assert hashlib.sha256(payload).hexdigest()=={digest!r}\n'
        'root=pathlib.Path("/kaggle/working/t3d-project");root.mkdir(exist_ok=True)\n'
        'with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)\n'
        'subprocess.run([sys.executable,"-m","pip","install","hilbertcurve>=2,<3","trimesh>=4,<5","Pillow>=10","ninja","opencv-python-headless","plyfile","zstandard"],check=True)\n'
        'import torch\nassert torch.cuda.is_available()\n')
    run=('subprocess.run([sys.executable,"-m","tools.probe_native_converter",'
         '"--source-sha256",hashlib.sha256(payload).hexdigest()],cwd=root,check=True)\n')
    notebook=dict(cells=[dict(cell_type='markdown',metadata={},source='# Native converter audit\nCPU conversion using a CUDA-built upstream extension; results are not production certification.'),
        *[dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s) for s in (setup,run)]],
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    output=ROOT/'kaggle-native';output.mkdir(exist_ok=True)
    (output/'native-audit.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
    metadata=dict(id='sumu143/transformer-native-3d-converter-audit',title='Transformer Native 3D Converter Audit',
        code_file='native-audit.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,
        enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=['sumu143/transformer-native-3d-source-assets','sumu143/transformer-native-3d-cuda-wheels-cu130'],
        competition_sources=[],kernel_sources=[],model_sources=[])
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(files=len(files),source_sha256=digest)))

if __name__=='__main__':main()
