"""Bundle field-to-skin inference checks for a private Kaggle Tesla T4 run."""
import base64
import hashlib
import json
from pathlib import Path

from tools.notebook_bundle import source_bundle


def main():
    root=Path(__file__).resolve().parents[1]
    checkpoint=root/'artifacts/kaggle-skin-pose-v1/skin-pose-pilot.pt'
    files=[root/'t3d/__init__.py',root/'t3d/articulation.py',root/'t3d/export.py',
        root/'t3d/schema.py',root/'t3d/serialization.py',root/'t3d/skin_ae.py',
        root/'tests/test_export.py',root/'tests/test_skin_ae.py',
        root/'tools/probe_skin_export.py',checkpoint]
    payload=source_bundle(root,files);digest=hashlib.sha256(payload).hexdigest()
    source=('import base64,hashlib,io,pathlib,subprocess,sys,zipfile,torch\n'
        'assert torch.cuda.is_available()\n'
        f'payload=base64.b64decode({base64.b64encode(payload).decode()!r})\n'
        f'assert hashlib.sha256(payload).hexdigest()=={digest!r}\n'
        'root=pathlib.Path("/kaggle/working/skin-export-project");root.mkdir(exist_ok=True)\n'
        'with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)\n'
        'logs=[]\n'
        'for pattern in ("test_export.py","test_skin_ae.py"):\n'
        '  r=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-p",pattern,"-v"],cwd=root,capture_output=True,text=True)\n'
        '  logs.append(r.stdout+r.stderr);print(logs[-1]);assert r.returncode==0\n'
        '(root.parent/"skin-export-tests.log").write_text("\\n".join(logs))\n'
        'subprocess.run([sys.executable,"-m","tools.probe_skin_export","--checkpoint",'
        'str(root/"artifacts/kaggle-skin-pose-v1/skin-pose-pilot.pt"),"--output",'
        '"/kaggle/working/skin-export-probe.json"],cwd=root,check=True)\n')
    out=root/'kaggle-skin-export';out.mkdir(exist_ok=True)
    notebook=dict(cells=[dict(id='skin-export-inference',cell_type='code',metadata={},execution_count=None,outputs=[],source=source)],
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        nbformat=4,nbformat_minor=5)
    (out/'skin-export.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
    metadata=dict(id='sumu143/transformer-native-3d-skinae-export-inference',
        title='Transformer Native 3D SkinAE Export Inference',code_file='skin-export.ipynb',
        language='python',kernel_type='notebook',is_private=True,enable_gpu=True,
        enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=[],
        competition_sources=[],kernel_sources=[],model_sources=[])
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(digest)


if __name__=='__main__':main()
