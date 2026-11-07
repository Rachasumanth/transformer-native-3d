"""Bundle the GPU runtime and resource profiling notebook for Kaggle GPU dual Tesla T4."""
import base64
import hashlib
import json
from pathlib import Path
from tools.notebook_bundle import source_bundle

root = Path.cwd()
files = [
    *sorted((root / 't3d').glob('*.py')),
    root / 'tools/profile_gpu_runtime.py',
]
payload = source_bundle(root, files)
digest = hashlib.sha256(payload).hexdigest()

source = (
    'import base64,hashlib,io,pathlib,subprocess,sys,zipfile,json\n'
    f'payload=base64.b64decode({base64.b64encode(payload).decode()!r})\n'
    f'assert hashlib.sha256(payload).hexdigest()=={digest!r}\n'
    'root=pathlib.Path("/kaggle/working/t3d-profiling");root.mkdir(exist_ok=True)\n'
    'with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)\n'
    'subprocess.run([sys.executable,"-m","pip","install","hilbertcurve>=2,<3","transformers>=4.40,<5","trimesh>=4.0"],check=True)\n'
    'subprocess.run([sys.executable,"-m","tools.profile_gpu_runtime","--checkpoints-dir","/kaggle/input/t3d-trained-checkpoints","--output-report","/kaggle/working/gpu-runtime-profile-report.json"],cwd=root,check=True)\n'
    'p=pathlib.Path("/kaggle/working/gpu-runtime-profile-report.json")\n'
    'assert p.exists(),"Expected output gpu-runtime-profile-report.json"\n'
    'r=json.loads(p.read_text())\n'
    f'r["source_sha256"]={digest!r}\n'
    'p.write_text(json.dumps(r,indent=2)+"\\n")\n'
)

out = root / 'kaggle-gpu-runtime-profile'
out.mkdir(exist_ok=True)
(out / 'gpu_profiling.ipynb').write_text(json.dumps({
    'cells': [{
        'cell_type': 'code',
        'id': 'profile_gpu',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': source
    }],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
    'nbformat': 4,
    'nbformat_minor': 5
}, indent=2) + '\n')

(out / 'kernel-metadata.json').write_text(json.dumps({
    'id': 'dileepkumarchakali/transformer-native-3d-gpu-runtime-profile',
    'title': 'Transformer Native 3D GPU Runtime Profile',
    'code_file': 'gpu_profiling.ipynb',
    'language': 'python',
    'kernel_type': 'notebook',
    'is_private': True,
    'enable_gpu': True,
    'enable_internet': True,
    'machine_shape': 'NvidiaTeslaT4',
    'dataset_sources': [
        'dileepkumarchakali/t3d-trained-checkpoints'
    ],
    'competition_sources': [],
    'kernel_sources': [],
    'model_sources': []
}, indent=2) + '\n')

print(f"Built kaggle-gpu-runtime-profile with source SHA256 {digest}")
