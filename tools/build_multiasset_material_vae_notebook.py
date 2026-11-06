"""Bundle the Multi-Asset Material SC-VAE training notebook for Kaggle GPU."""
import base64
import hashlib
import json
from pathlib import Path
from tools.notebook_bundle import source_bundle

root = Path.cwd()

# Gather Python source files (code only, dataset comes from mounted Kaggle dataset)
files = [
    *sorted((root / 't3d').glob('*.py')),
    root / 'tools/train_multiasset_material_vae.py'
]

print(f"Bundling {len(files)} code files into payload...")
payload = source_bundle(root, files)
digest = hashlib.sha256(payload).hexdigest()
payload_b64 = base64.b64encode(payload).decode()
print(f"Payload size: {len(payload) / 1024:.1f} KB (base64: {len(payload_b64) / 1024:.1f} KB), SHA-256: {digest}")

source = (
    'import base64,hashlib,io,pathlib,subprocess,sys,zipfile,json\n'
    f'payload=base64.b64decode({payload_b64!r})\n'
    f'assert hashlib.sha256(payload).hexdigest()=={digest!r}, "Payload corruption"\n'
    'root=pathlib.Path("/kaggle/working/material-train");root.mkdir(parents=True,exist_ok=True)\n'
    'with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)\n'
    'subprocess.run([sys.executable,"-m","pip","install","hilbertcurve>=2,<3"],check=True)\n'
    'subprocess.run([sys.executable,"-m","tools.train_multiasset_material_vae"],cwd=root,check=True)\n'
    'p=pathlib.Path("/kaggle/working/material-multiasset-report.json")\n'
    'assert p.exists(),"Expected material-multiasset-report.json"\n'
    'r=json.loads(p.read_text())\n'
    f'r["bundle_sha256"]={digest!r}\n'
    'p.write_text(json.dumps(r,indent=2)+"\\n")\n'
    'print("EXECUTION COMPLETE. Report contents:")\n'
    'print(json.dumps(r,indent=2))\n'
)

out_dir = root / 'kaggle-material-multiasset-32'
out_dir.mkdir(parents=True, exist_ok=True)
notebook_path = out_dir / 'material_multiasset_vae.ipynb'
notebook_path.write_text(json.dumps({
    'cells': [{
        'cell_type': 'code',
        'id': 'train_material_multiasset',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': source
    }],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
    'nbformat': 4,
    'nbformat_minor': 5
}, indent=2) + '\n')

# Determine active Kaggle username
try:
    import kaggle
    owner = kaggle.api.get_config_value('username') or 'dileepkumarchakali'
except Exception:
    owner = 'dileepkumarchakali'

metadata_path = out_dir / 'kernel-metadata.json'
metadata_path.write_text(json.dumps({
    'id': f'{owner}/transformer-native-3d-material-multiasset-vae',
    'title': 'Transformer Native 3D Material Multiasset VAE',
    'code_file': 'material_multiasset_vae.ipynb',
    'language': 'python',
    'kernel_type': 'notebook',
    'is_private': True,
    'enable_gpu': True,
    'enable_internet': True,
    'machine_shape': 'NvidiaTeslaT4',
    'dataset_sources': [
        f'{owner}/t3d-material-multiasset-32'
    ],
    'competition_sources': [],
    'kernel_sources': [],
    'model_sources': []
}, indent=2) + '\n')

print(f"Generated notebook in {notebook_path}")
print(f"Generated kernel metadata in {metadata_path} for owner: {owner}")
