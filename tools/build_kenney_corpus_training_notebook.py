"""Bundle the full 868-asset Kenney corpus training notebook for Kaggle GPU."""
import base64, hashlib, json
from pathlib import Path
from tools.notebook_bundle import source_bundle

root = Path.cwd()
files = [
    *sorted((root / 't3d').glob('*.py')),
    root / 'tools/train_kenney_corpus_32.py',
    root / 'tools/audit_triton_paired_reconstruction.py',
    root / 'artifacts/mixed-shape-pilot-32/manifest.json'
]
payload = source_bundle(root, files)
digest = hashlib.sha256(payload).hexdigest()

source = (
    'import base64,hashlib,io,pathlib,subprocess,sys,zipfile,json\n'
    f'payload=base64.b64decode({base64.b64encode(payload).decode()!r})\n'
    f'assert hashlib.sha256(payload).hexdigest()=={digest!r}\n'
    'root=pathlib.Path("/kaggle/working/corpus-train");root.mkdir(exist_ok=True)\n'
    'with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)\n'
    'subprocess.run([sys.executable,"-m","pip","install","hilbertcurve>=2,<3"],check=True)\n'
    'subprocess.run([sys.executable,"-m","tools.train_kenney_corpus_32"],cwd=root,check=True)\n'
    'p=pathlib.Path("/kaggle/working/triton-kenney-corpus-32.json")\n'
    'assert p.exists(),"Expected output json report"\n'
    'r=json.loads(p.read_text())\n'
    f'r["source_sha256"]={digest!r}\n'
    'p.write_text(json.dumps(r,indent=2)+"\\n")\n'
)

out = root / 'kaggle-kenney-corpus-training'
out.mkdir(exist_ok=True)
(out / 'corpus_train.ipynb').write_text(json.dumps({
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
    'id': 'sumu143/transformer-native-3d-skinae-cross-validation',
    'title': 'Transformer Native 3D Kenney Corpus Training',
    'code_file': 'corpus_train.ipynb',
    'language': 'python',
    'kernel_type': 'notebook',
    'is_private': True,
    'enable_gpu': True,
    'enable_internet': True,
    'machine_shape': 'NvidiaTeslaT4',
    'dataset_sources': [
        'sumu143/t3d-mixed-pilot-shape-checkpoint',
        'sumu143/t3d-kenney-corpus-32'
    ],
    'competition_sources': [],
    'kernel_sources': [],
    'model_sources': []
}, indent=2) + '\n')

print(f"Generated notebook in {out}")
print(f"Bundle SHA256: {digest}")
