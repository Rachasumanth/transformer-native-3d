"""Build private Kaggle notebook for saved decoder checkpoint analysis."""
import base64,hashlib,io,json,zipfile
from pathlib import Path
from tools.notebook_bundle import source_bundle

ROOT=Path(__file__).resolve().parents[1]

def main():
    files=sorted([*ROOT.glob('t3d/*.py'),ROOT/'tools/audit_decoder_trace.py',ROOT/'tools/probe_decoder_trace.py',
                  ROOT/'tests/test_sc_vae.py',ROOT/'tests/test_decoder_trace_audit.py',ROOT/'pyproject.toml'])
    payload = source_bundle(ROOT, files)
    digest = hashlib.sha256(payload).hexdigest()
    setup=('import base64,hashlib,io,json,pathlib,subprocess,sys,zipfile\n'
           f'payload=base64.b64decode({base64.b64encode(payload).decode()!r})\n'
           f'assert hashlib.sha256(payload).hexdigest()=={digest!r}\n'
           'root=pathlib.Path("/kaggle/working/t3d-project");root.mkdir(exist_ok=True)\n'
           'with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)\n'
           'subprocess.run([sys.executable,"-m","pip","install","hilbertcurve>=2,<3","trimesh>=4,<5","Pillow>=10"],check=True)\n'
           'import torch\nassert torch.cuda.is_available()\n')
    run=('result=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],cwd=root,capture_output=True,text=True)\n'
         '(root.parent/"trace-tests.log").write_text(result.stdout+result.stderr)\n'
         'print(result.stdout+result.stderr)\nassert result.returncode==0\n'
         'subprocess.run([sys.executable,"-m","tools.probe_decoder_trace","--source-sha256",hashlib.sha256(payload).hexdigest()],cwd=root,check=True)\n')
    cells=[dict(cell_type='markdown',metadata={},source='# Sparse decoder trace audit'),
           *[dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s) for s in (setup,run)]]
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    output=ROOT/'kaggle-trace';output.mkdir(exist_ok=True)
    (output/'decoder-trace.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
    metadata=dict(id='sumu143/transformer-native-3d-decoder-trace',title='Transformer Native 3D Decoder Trace',
        code_file='decoder-trace.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,
        enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=['sumu143/transformer-native-3d-pilot','sumu143/transformer-native-3d-trace-checkpoints'],
        competition_sources=[],kernel_sources=[],model_sources=[])
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(files=len(files),source_sha256=digest)))

if __name__=='__main__':main()
