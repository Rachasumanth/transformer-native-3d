"""Bundle a three-fold asset-held-out SkinAE animation experiment for Kaggle."""
import base64
import hashlib
import json
from pathlib import Path

from tools.notebook_bundle import source_bundle


def main():
    root=Path(__file__).resolve().parents[1]
    files=[*sorted((root/'t3d').glob('*.py')),
           root/'tests/test_import_rig.py',root/'tests/test_skin_ae.py',
           root/'tools/train_skin_pose_probe.py',
           *sorted((root/'artifacts/skin-family-vertex-pilot').glob('*'))]
    payload=source_bundle(root,files);digest=hashlib.sha256(payload).hexdigest()
    source=('import base64,hashlib,io,json,pathlib,subprocess,sys,zipfile,torch\n'
        'assert torch.cuda.is_available()\n'
        f'payload=base64.b64decode({base64.b64encode(payload).decode()!r})\n'
        f'assert hashlib.sha256(payload).hexdigest()=={digest!r}\n'
        'root=pathlib.Path("/kaggle/working/skin-pose-cv-project");root.mkdir(exist_ok=True)\n'
        'with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)\n'
        'r=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],cwd=root,capture_output=True,text=True)\n'
        '(root.parent/"skin-pose-cv-tests.log").write_text(r.stdout+r.stderr)\n'
        'print(r.stdout+r.stderr)\nassert r.returncode==0\n'
        'for strength in (0.0, 1.0):\n'
        '  subprocess.run([sys.executable,"-m","tools.train_skin_pose_probe","--output-dir",f"/kaggle/working/loss-{strength}","--data","artifacts/skin-family-vertex-pilot","--deformation-loss",str(strength),"--select-metric","deformation_error","--source-sha256",'
        f'{digest!r}],cwd=root,check=True)\n')
    output=root/'kaggle-skin-family-deformation-select';output.mkdir(exist_ok=True)
    notebook=dict(cells=[dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source)],
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        nbformat=4,nbformat_minor=5)
    (output/'skin-pose-cv.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
    metadata=dict(id='sumu143/transformer-native-3d-skinae-cross-validation',
        title='Transformer Native 3D SkinAE Cross Validation',code_file='skin-pose-cv.ipynb',
        language='python',kernel_type='notebook',is_private=True,enable_gpu=True,
        enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=[],
        competition_sources=[],kernel_sources=[],model_sources=[])
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(digest)


if __name__=='__main__':main()
