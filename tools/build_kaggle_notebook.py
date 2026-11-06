"""Bundle only project code into a private Kaggle GPU bootstrap notebook."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

from tools.notebook_bundle import source_bundle

ROOT = Path(__file__).resolve().parents[1]

def main():
    files = sorted([*ROOT.glob('t3d/*.py'), *ROOT.glob('tests/test_*.py'), ROOT/'pyproject.toml',
                    ROOT/'tools/roundtrip_demo.py', ROOT/'tools/train_compression_probe.py', ROOT/'tools/train_corpus_probe.py',
                    ROOT/'tools/benchmark_extraction.py', ROOT/'tools/probe_cumesh.py',
                    ROOT/'tools/probe_export.py', ROOT/'tools/build_cuda_dependency.py',
                    ROOT/'tools/audit_edge_predictions.py'])
    payload = source_bundle(ROOT, files)
    digest = hashlib.sha256(payload).hexdigest()
    cells = []
    def code(source):
        cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=source))
    cells.append(dict(cell_type='markdown', metadata={}, source=
        '# Transformer-native 3D GPU bootstrap\nPrivate project verification on Kaggle. '
        'Runs unit tests and a synthetic CUDA optimization probe; this is not SC-VAE training.'))
    code('import base64, hashlib, io, json, pathlib, subprocess, sys, zipfile\n'
         f'payload = base64.b64decode({base64.b64encode(payload).decode()!r})\n'
         f'assert hashlib.sha256(payload).hexdigest() == {digest!r}\n'
         'root = pathlib.Path("/kaggle/working/t3d-project")\nroot.mkdir(exist_ok=True)\n'
         'with zipfile.ZipFile(io.BytesIO(payload)) as archive:\n    archive.extractall(root)\n'
         'print("Source SHA256:", hashlib.sha256(payload).hexdigest())\n')
    code('subprocess.run([sys.executable, "-m", "pip", "install", "hilbertcurve>=2,<3", "trimesh>=4,<5", "Pillow>=10", "opencv-python-headless"], check=True)\n'
         'import torch\nassert torch.cuda.is_available(), "GPU allocation required; refusing CPU fallback"\n'
         'print("PyTorch", torch.__version__)\n'
         'print("CUDA devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\n')
    code('result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],\n'
         '                        cwd=root, text=True, capture_output=True)\n'
         '(root.parent/"unit-tests.log").write_text(result.stdout+result.stderr)\n'
         'print(result.stdout+result.stderr)\nassert result.returncode == 0, "Project tests failed"\n'
         'subprocess.run([sys.executable, "-m", "tools.roundtrip_demo"], cwd=root, check=True)\n')
    code('torch.manual_seed(42)\ndevice = torch.device("cuda:0")\n'
         'x = torch.randn(256, 32, device=device)\ntarget = torch.randn(256, 8, device=device)\n'
         'model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8)).to(device)\n'
         'optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)\nlosses = []\n'
         'for step in range(30):\n    optimizer.zero_grad(set_to_none=True)\n'
         '    loss = torch.nn.functional.mse_loss(model(x), target)\n'
         '    loss.backward()\n    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())\n'
         '    optimizer.step()\n    losses.append(loss.item())\n'
         'torch.cuda.synchronize()\nassert losses[-1] < losses[0]\n'
         'report = dict(source_sha256=hashlib.sha256(payload).hexdigest(), torch=torch.__version__,\n'
         '              cuda=torch.version.cuda, devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],\n'
         '              unit_tests_passed=True, probe="synthetic MLP CUDA optimization, not SC-VAE training",\n'
         '              initial_loss=losses[0], final_loss=losses[-1], steps=len(losses))\n'
         '(root.parent/"gpu-report.json").write_text(json.dumps(report, indent=2))\n'
         'torch.save(model.state_dict(), root.parent/"synthetic-probe.pt")\nprint(json.dumps(report, indent=2))\n')
    code('subprocess.run([sys.executable, "-m", "tools.train_compression_probe"], cwd=root, check=True)\n')
    code('subprocess.run([sys.executable, "-m", "tools.benchmark_extraction", "--source-sha256",\n'
         '                hashlib.sha256(payload).hexdigest()], cwd=root, check=True)\n')
    code('subprocess.run([sys.executable, "-m", "tools.probe_cumesh", "--source-sha256",\n'
         '                hashlib.sha256(payload).hexdigest()], cwd=root, check=True)\n')
    code('subprocess.run([sys.executable, "-m", "tools.probe_export", "--source-sha256",\n'
         '                hashlib.sha256(payload).hexdigest()], cwd=root, check=True)\n')
    code('subprocess.run([sys.executable, "-m", "tools.train_corpus_probe", "--render-weight", "0.1"], cwd=root, check=True)\n')
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')),
                    nbformat=4, nbformat_minor=5)
    destination = ROOT/'kaggle'
    destination.mkdir(exist_ok=True)
    (destination/'gpu-bootstrap.ipynb').write_text(json.dumps(notebook, indent=2)+'\n')
    metadata = dict(id='sumu143/transformer-native-3d-gpu-bootstrap', title='Transformer Native 3D GPU Bootstrap',
                    code_file='gpu-bootstrap.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=True, enable_internet=True, machine_shape='NvidiaTeslaT4',
                    dataset_sources=['sumu143/transformer-native-3d-pilot',
                                     'sumu143/transformer-native-3d-cuda-wheels'],
                    competition_sources=[], kernel_sources=[], model_sources=[])
    (destination/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(dict(files=len(files), source_sha256=digest, notebook=str(destination/'gpu-bootstrap.ipynb'))))

if __name__ == '__main__':
    main()
