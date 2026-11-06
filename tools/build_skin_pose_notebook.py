"""Bundle the animated SkinAE pilot and submit-ready Kaggle notebook."""
import base64
import hashlib
import json
from pathlib import Path

from tools.notebook_bundle import source_bundle


def main():
    root=Path(__file__).resolve().parents[1]
    files=[root/'t3d/__init__.py',root/'t3d/articulation.py',root/'t3d/gltf_accessors.py',
           root/'t3d/import_rig.py',root/'t3d/normalization.py',root/'t3d/skin_ae.py',
           root/'tests/test_import_rig.py',root/'tests/test_skin_ae.py',
           root/'tools/train_skin_pose_probe.py',
           *sorted((root/'artifacts/skin-pose-pilot').glob('*'))]
    payload=source_bundle(root,files);digest=hashlib.sha256(payload).hexdigest()
    source=('import base64,hashlib,io,json,pathlib,subprocess,sys,zipfile,torch\n'
        'assert torch.cuda.is_available()\n'
        f'payload=base64.b64decode({base64.b64encode(payload).decode()!r})\n'
        f'assert hashlib.sha256(payload).hexdigest()=={digest!r}\n'
        'root=pathlib.Path("/kaggle/working/skin-pose-project");root.mkdir(exist_ok=True)\n'
        'with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)\n'
        'r=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],cwd=root,capture_output=True,text=True)\n'
        '(root.parent/"skin-pose-tests.log").write_text(r.stdout+r.stderr)\n'
        'print(r.stdout+r.stderr)\nassert r.returncode==0\n'
        f'subprocess.run([sys.executable,"-m","tools.train_skin_pose_probe","--source-sha256",{digest!r}],cwd=root,check=True)\n')
    output=root/'kaggle-skin-pose';output.mkdir(exist_ok=True)
    notebook=dict(cells=[dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source)],
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        nbformat=4,nbformat_minor=5)
    (output/'skin-pose.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
    metadata=dict(id='sumu143/transformer-native-3d-skinae-multi-pose-pilot',
        title='Transformer Native 3D SkinAE Multi Pose Pilot',code_file='skin-pose.ipynb',
        language='python',kernel_type='notebook',is_private=True,enable_gpu=True,
        enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=[],
        competition_sources=[],kernel_sources=[],model_sources=[])
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(digest)


if __name__=='__main__':main()
