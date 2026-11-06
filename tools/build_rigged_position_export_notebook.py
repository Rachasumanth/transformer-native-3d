"""Build the source-matched Kaggle full raw-to-rigged-GLB integration probe."""
import base64
import hashlib
import json
from pathlib import Path

from tools.notebook_bundle import source_bundle


def main():
    root=Path(__file__).resolve().parents[1]
    files=[*sorted((root/'t3d').glob('*.py')),
        root/'tests/test_articulation.py',root/'tests/test_import_rig.py',root/'tests/test_export.py',
        root/'tests/test_glb.py',root/'tests/test_materials.py',root/'tests/test_metrics.py',root/'tests/test_skin_ae.py',
        root/'tools/build_cuda_dependency.py',root/'tools/probe_export.py',
        root/'tools/audit_rig_pbr_textures.py',root/'tools/audit_material_fidelity.py',
        root/'artifacts/kaggle-skin-position-ablation-v3/loss-1.0/skin-pose-pilot-holdout-CesiumMan.pt']
    payload=source_bundle(root,files);digest=hashlib.sha256(payload).hexdigest()
    source=('import base64,hashlib,io,pathlib,subprocess,sys,zipfile,torch\n'
        'assert torch.cuda.is_available(),"CUDA allocation required"\n'
        f'payload=base64.b64decode({base64.b64encode(payload).decode()!r})\n'
        f'assert hashlib.sha256(payload).hexdigest()=={digest!r}\n'
        'root=pathlib.Path("/kaggle/working/t3d-rigged-export");root.mkdir(exist_ok=True)\n'
        'with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)\n'
        'subprocess.run([sys.executable,"-m","pip","install","hilbertcurve>=2,<3","trimesh>=4,<5","Pillow>=10","opencv-python-headless","ninja"],check=True)\n'
        'logs=[]\n'
        'for pattern in ("test_articulation.py","test_import_rig.py","test_export.py","test_glb.py","test_materials.py","test_metrics.py","test_skin_ae.py"):\n'
        '  r=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-p",pattern,"-v"],cwd=root,capture_output=True,text=True)\n'
        '  logs.append(r.stdout+r.stderr);print(logs[-1]);assert r.returncode==0\n'
        '(root.parent/"rigged-export-tests.log").write_text("\\n".join(logs))\n'
        'fixture=pathlib.Path("/kaggle/input/transformer-native-3d-textured-rig-inputs")\n'
        'subprocess.run([sys.executable,"-m","tools.probe_export","--source-sha256",'
        f'{digest!r},"--skinae-checkpoint",str(root/"artifacts/kaggle-skin-position-ablation-v3/loss-1.0/skin-pose-pilot-holdout-CesiumMan.pt"),'
        '"--real-rig-source",str(fixture/"CesiumMan.glb"),'
        '"--real-grid-size","128","--real-texture-size","2048",'
        '"--dependency-cache","/kaggle/input/transformer-native-3d-cuda-wheels-cu130"],cwd=root,check=True)\n')
    out=root/'kaggle-rigged-position-export';out.mkdir(exist_ok=True)
    notebook=dict(cells=[dict(id='rigged-export-probe',cell_type='code',metadata={},
        execution_count=None,outputs=[],source=source)],
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        nbformat=4,nbformat_minor=5)
    (out/'rigged-export.ipynb').write_text(json.dumps(notebook,indent=2)+'\n')
    metadata=dict(id='sumu143/transformer-native-3d-rigged-export-probe',
        title='Transformer Native 3D Rigged Export Probe',code_file='rigged-export.ipynb',
        language='python',kernel_type='notebook',is_private=True,enable_gpu=True,
        enable_internet=True,machine_shape='NvidiaTeslaT4',
        dataset_sources=['sumu143/transformer-native-3d-cuda-wheels-cu130',
                         'sumu143/transformer-native-3d-textured-rig-inputs'],
        competition_sources=[],kernel_sources=[],model_sources=[])
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(digest)


if __name__=='__main__':main()
