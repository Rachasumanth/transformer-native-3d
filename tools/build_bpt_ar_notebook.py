"""Bundle the Track B causal model and licensed cleaned fixture for Kaggle."""
import base64
import hashlib
import json
from pathlib import Path

from tools.notebook_bundle import source_bundle


def main():
    root = Path(__file__).resolve().parents[1]
    files = [root / name for name in (
        't3d/__init__.py', 't3d/bpt.py', 't3d/bpt_grammar.py', 't3d/sampling.py', 't3d/bpt_ar.py',
        'tests/test_bpt_ar.py', 'tools/train_bpt_ar_probe.py',
        'artifacts/bpt/cleaned_Avocado_all_24_rotations.npz',
        'artifacts/bpt/cleaned_Box_all_24_rotations.npz',
        'artifacts/bpt/cleaned_PolyHaven_food_lime_01_all_24_rotations.npz')]
    # Include the preserved upstream notice with the embedded cleaned asset.
    notices = [root / name for name in ('artifacts/import/avocado/SOURCE.md',
        'artifacts/bpt-candidates/Box/LICENSE.md',
        'artifacts/bpt-candidates/PolyHaven_food_lime_01/LICENSE.md')]
    if not all(path.is_file() for path in notices):
        raise RuntimeError('Preserved Avocado license required')
    files.extend(notices)
    payload = source_bundle(root, files)
    digest = hashlib.sha256(payload).hexdigest()
    sources = [
        'import base64, hashlib, io, json, pathlib, subprocess, sys, zipfile\n'
        f'payload = base64.b64decode({base64.b64encode(payload).decode()!r})\n'
        f'assert hashlib.sha256(payload).hexdigest() == {digest!r}\n'
        'root = pathlib.Path("/kaggle/working/t3d-project")\nroot.mkdir(exist_ok=True)\n'
        'with zipfile.ZipFile(io.BytesIO(payload)) as archive:\n    archive.extractall(root)\n',
        'import torch\nassert torch.cuda.is_available(), "Kaggle GPU required"\n'
        'subprocess.run([sys.executable, "-m", "pip", "install", "numpy"], check=True)\n'
        'result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_bpt_ar.py", "-v"], cwd=root, capture_output=True, text=True)\n'
        '(root.parent / "bpt-ar-tests.log").write_text(result.stdout + result.stderr)\n'
        'print(result.stdout + result.stderr)\nassert result.returncode == 0\n',
        'subprocess.run([sys.executable, "-m", "tools.train_bpt_ar_probe", "--source-sha256", '
        f'{digest!r}], cwd=root, check=True)\n',
    ]
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')),
        cells=[dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s)
               for s in sources])
    destination = root / 'kaggle-bpt-ar'
    destination.mkdir(exist_ok=True)
    (destination / 'bpt-ar.ipynb').write_text(json.dumps(notebook, indent=2) + '\n')
    metadata = dict(id='sumu143/transformer-native-3d-bpt-ar-probe',
        title='Transformer Native 3D BPT AR Probe', code_file='bpt-ar.ipynb',
        language='python', kernel_type='notebook', is_private=True,
        enable_gpu=True, enable_internet=True, machine_shape='NvidiaTeslaT4',
        dataset_sources=[], competition_sources=[], kernel_sources=[], model_sources=[])
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(dict(source_sha256=digest, files=len(files), notebook=str(destination))))


if __name__ == '__main__':
    main()
