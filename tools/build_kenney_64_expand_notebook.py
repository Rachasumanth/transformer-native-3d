"""Bundle the 19-kit 64³ corpus expansion notebook for Kaggle CPU."""
import base64
import hashlib
import json
from pathlib import Path

from tools.notebook_bundle import source_bundle

root = Path.cwd()
files = [
    *sorted((root / 't3d').glob('*.py')),
    root / 'tools/fetch_kenney_index.py',
    root / 'tools/prepare_kenney_catalog.py',
    root / 'tools/make_id_split.py',
    root / 'tools/generate_corpus_prompts.py',
    root / 'artifacts/kenney-raw-expanded/candidate-index.json',
]
payload = source_bundle(root, files)
digest = hashlib.sha256(payload).hexdigest()

source = (
    'import base64,hashlib,io,pathlib,subprocess,sys,zipfile\n'
    f'payload=base64.b64decode({base64.b64encode(payload).decode()!r})\n'
    f'assert hashlib.sha256(payload).hexdigest()=={digest!r}\n'
    'root=pathlib.Path("/kaggle/working/t3d-expand");root.mkdir(exist_ok=True)\n'
    'with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)\n'
    'subprocess.run([sys.executable,"-m","pip","install","-q","hilbertcurve>=2,<3","transformers>=4.30","requests","trimesh>=4"],check=True)\n'
    'index=root/"artifacts/kenney-raw-expanded/candidate-index.json"\n'
    'raw=root/"artifacts/kenney-catalog-download"\n'
    'corpus=root/"artifacts/kenney-corpus-64-expanded"\n'
    'subprocess.run([sys.executable,"-u","-m","tools.fetch_kenney_index","--index",str(index),"--raw-root",str(raw)],cwd=root,check=True)\n'
    'import json\n'
    'n=len(json.loads(index.read_text())["candidates"])\n'
    'subprocess.run([sys.executable,"-u","-m","tools.prepare_kenney_catalog","--index",str(index),"--raw-root",str(raw),"--output",str(corpus),"--grid-size","64","--start","0","--stop",str(n),"--stride","1","--workers","4"],cwd=root,check=True)\n'
    'id_root=root/"artifacts/kenney-corpus-64-id-expanded"\n'
    'subprocess.run([sys.executable,"-u","-m","tools.make_id_split","--manifest",str(corpus/"manifest.json"),"--output",str(id_root),"--force"],cwd=root,check=True)\n'
    'archive=pathlib.Path("/kaggle/working/kenney-corpus-64-id-expanded.zip")\n'
    'with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:\n'
    '    for p in sorted(id_root.iterdir()):\n'
    '        if p.is_file():z.write(p,arcname=p.name)\n'
    'summary={"source_sha256":'+repr(digest)+',"candidates":n,"corpus":str(corpus),"id_corpus":str(id_root),"archive":str(archive),"archive_bytes":archive.stat().st_size}\n'
    'pathlib.Path("/kaggle/working/expand-summary.json").write_text(json.dumps(summary,indent=2)+"\\n")\n'
    'print(json.dumps(summary))\n'
)

out = root / 'kaggle-kenney-64-expand'
out.mkdir(exist_ok=True)
(out / 'kenney_64_expand.ipynb').write_text(json.dumps({
    'cells': [{
        'cell_type': 'code',
        'id': 'expand',
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
    'id': 'santustarahe/transformer-native-3d-kenney-64-expand',
    'title': 'Transformer Native 3D Kenney 64 Expand',
    'code_file': 'kenney_64_expand.ipynb',
    'language': 'python',
    'kernel_type': 'notebook',
    'is_private': True,
    'enable_gpu': False,
    'enable_internet': True,
    'dataset_sources': [],
    'competition_sources': [],
    'kernel_sources': [],
    'model_sources': []
}, indent=2) + '\n')

print(f"Generated 19-kit expansion notebook in {out}")
print(f"Bundle SHA256: {digest}")
