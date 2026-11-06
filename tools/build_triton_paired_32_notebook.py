"""Bundle an optional Triton sparse convolution comparison for Kaggle."""
import base64,hashlib,json
from pathlib import Path
from tools.notebook_bundle import source_bundle
root=Path.cwd();files=[*sorted((root/'t3d').glob('*.py')),root/'tools/train_triton_paired_32.py',root/'artifacts/pilot-dataset-32/manifest.json',root/'artifacts/pilot-dataset-32/avocado.t3d',root/'artifacts/pilot-dataset-32/waterbottle.t3d'];payload=source_bundle(root,files);digest=hashlib.sha256(payload).hexdigest()
source=('import base64,hashlib,io,pathlib,subprocess,sys,zipfile,json\n'
 f'payload=base64.b64decode({base64.b64encode(payload).decode()!r})\n'
 f'assert hashlib.sha256(payload).hexdigest()=={digest!r}\n'
 'root=pathlib.Path("/kaggle/working/triton-probe");root.mkdir(exist_ok=True)\n'
 'with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)\n'
 'subprocess.run([sys.executable,"-m","pip","install","hilbertcurve>=2,<3"],check=True)\n'
 'subprocess.run([sys.executable,"-m","tools.train_triton_paired_32"],cwd=root,check=True)\n'
 'p=root.parent/"triton-paired-32.json";r=json.loads(p.read_text())\n'
 f'r["source_sha256"]={digest!r};p.write_text(json.dumps(r,indent=2)+"\\n")\n')
out=root/'kaggle-triton-paired-32';out.mkdir(exist_ok=True)
(out/'triton.ipynb').write_text(json.dumps({'cells':[{'cell_type':'code','id':'triton','metadata':{},'execution_count':None,'outputs':[],'source':source}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5},indent=2)+'\n')
(out/'kernel-metadata.json').write_text(json.dumps({'id':'sumu143/transformer-native-3d-skinae-cross-validation','title':'Transformer Native 3D SkinAE Cross Validation','code_file':'triton.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]},indent=2)+'\n');print(digest)
