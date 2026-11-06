"""Create a separate Kaggle evaluation notebook using existing kernel outputs."""
import base64
import hashlib
import json
from pathlib import Path
from tools.notebook_bundle import source_bundle


def main():
    root = Path(__file__).resolve().parents[1]
    files = [root / p for p in ('t3d/__init__.py', 't3d/bpt.py', 't3d/sampling.py',
        't3d/bpt_ar.py', 't3d/bpt_grammar.py', 'tests/test_bpt_ar.py',
        'tests/test_bpt_grammar.py', 'tools/evaluate_bpt_constraints.py',
        'artifacts/bpt-training-pilot/manifest.json')]
    payload = source_bundle(root, files)
    digest = hashlib.sha256(payload).hexdigest()
    sources = ['import base64,hashlib,io,pathlib,subprocess,sys,zipfile\n'
        f'payload=base64.b64decode({base64.b64encode(payload).decode()!r})\n'
        f'assert hashlib.sha256(payload).hexdigest()=={digest!r}\n'
        'root=pathlib.Path("/kaggle/working/t3d-project")\nroot.mkdir(exist_ok=True)\n'
        'with zipfile.ZipFile(io.BytesIO(payload)) as z:\n    z.extractall(root)\n',
        'import torch\nassert torch.cuda.is_available()\n'
        'r=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],cwd=root,capture_output=True,text=True)\n'
        '(root.parent/"tests.log").write_text(r.stdout+r.stderr)\nprint(r.stdout+r.stderr)\nassert r.returncode==0\n',
        f'subprocess.run([sys.executable,"-m","tools.evaluate_bpt_constraints","--source-sha256",{digest!r}],cwd=root,check=True)\n']
    out = root / 'kaggle-bpt-constraints'
    out.mkdir(exist_ok=True)
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(
        name='python3', display_name='Python 3', language='python')),
        cells=[dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s) for s in sources])
    (out / 'constraints.ipynb').write_text(json.dumps(notebook, indent=2) + '\n')
    meta = dict(id='sumu143/transformer-native-3d-bpt-constraints', title='Transformer Native 3D BPT Constraints',
        code_file='constraints.ipynb', language='python', kernel_type='notebook', is_private=True,
        enable_gpu=True, enable_internet=True, dataset_sources=['sumu143/transformer-native-3d-bpt-pilot'],
        kernel_sources=['sumu143/transformer-native-3d-bpt-multiasset'], competition_sources=[], model_sources=[])
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(digest)


if __name__ == '__main__':
    main()
