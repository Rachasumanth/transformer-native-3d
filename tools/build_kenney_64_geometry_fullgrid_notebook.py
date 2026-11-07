"""Bundle the full 4³-grid Stage 2 geometry DiT training notebook for Kaggle GPU."""
import base64
import hashlib
import json
from pathlib import Path

from tools.notebook_bundle import source_bundle

root = Path.cwd()
files = [
    *sorted((root / 't3d').glob('*.py')),
    root / 'tools/train_kenney_64_geometry_dit_fullgrid.py',
    root / 'tools/generate_corpus_prompts.py',
]
payload = source_bundle(root, files)
digest = hashlib.sha256(payload).hexdigest()

source = (
    'import base64,hashlib,io,pathlib,subprocess,sys,zipfile,json\n'
    f'payload=base64.b64decode({base64.b64encode(payload).decode()!r})\n'
    f'assert hashlib.sha256(payload).hexdigest()=={digest!r}\n'
    'root=pathlib.Path("/kaggle/working/geometry-64-fullgrid");root.mkdir(exist_ok=True)\n'
    'with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)\n'
    'subprocess.run([sys.executable,"-m","pip","install","hilbertcurve>=2,<3","transformers>=4.30"],check=True)\n'
    'subprocess.run([sys.executable,"-m","tools.train_kenney_64_geometry_dit_fullgrid"],cwd=root,check=True)\n'
    'p=pathlib.Path("/kaggle/working/geometry-dit-64-fullgrid-report.json")\n'
    'assert p.exists(),"Expected output json report"\n'
    'r=json.loads(p.read_text())\n'
    f'r["source_sha256"]={digest!r}\n'
    'p.write_text(json.dumps(r,indent=2)+"\\n")\n'
)

out = root / 'kaggle-kenney-64-geometry-fullgrid'
out.mkdir(exist_ok=True)
(out / 'geometry_64_fullgrid.ipynb').write_text(json.dumps({
    'cells': [{
        'cell_type': 'code',
        'id': 'train',
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
    'id': 'sumu143/transformer-native-3d-kenney-64-geometry-fullgrid',
    'title': 'Transformer Native 3D Kenney 64 Geometry Fullgrid',
    'code_file': 'geometry_64_fullgrid.ipynb',
    'language': 'python',
    'kernel_type': 'notebook',
    'is_private': True,
    'enable_gpu': True,
    'enable_internet': True,
    'machine_shape': 'NvidiaTeslaT4',
    'dataset_sources': [
        'sumu143/t3d-kenney-64-pipeline'
    ],
    'competition_sources': [],
    'kernel_sources': [],
    'model_sources': []
}, indent=2) + '\n')

print(f"Generated 64³ full-grid geometry DiT notebook in {out}")
print(f"Bundle SHA256: {digest}")
