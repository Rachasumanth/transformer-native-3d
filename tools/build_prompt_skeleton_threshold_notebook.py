"""Bundle the pilot threshold diagnosis for a private Kaggle checkpoint."""
import base64,hashlib,json
from pathlib import Path
from tools.notebook_bundle import source_bundle
root=Path.cwd();files=[*sorted((root/'t3d').glob('*.py')),root/'tools/audit_prompt_skeleton_threshold.py',*sorted((root/'artifacts/prompt-occupancy-pilot').glob('*'))];payload=source_bundle(root,files);digest=hashlib.sha256(payload).hexdigest()
source=('import base64,hashlib,io,pathlib,subprocess,sys,zipfile,json\n'
 f'payload=base64.b64decode({base64.b64encode(payload).decode()!r})\n'
 f'assert hashlib.sha256(payload).hexdigest()=={digest!r}\n'
 'root=pathlib.Path("/kaggle/working/threshold-probe");root.mkdir(exist_ok=True)\n'
 'with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)\n'
 'subprocess.run([sys.executable,"-m","pip","install","hilbertcurve>=2,<3","transformers>=4.45,<5","safetensors"],check=True)\n'
 'subprocess.run([sys.executable,"-m","tools.audit_prompt_skeleton_threshold"],cwd=root,check=True)\n'
 'p=root.parent/"prompt-skeleton-threshold.json";r=json.loads(p.read_text())\n'
 f'r["source_sha256"]={digest!r};p.write_text(json.dumps(r,indent=2)+"\\n")\n')
out=root/'kaggle-prompt-skeleton-threshold';out.mkdir(exist_ok=True)
(out/'threshold.ipynb').write_text(json.dumps({'cells':[{'cell_type':'code','id':'threshold','metadata':{},'execution_count':None,'outputs':[],'source':source}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5},indent=2)+'\n')
(out/'kernel-metadata.json').write_text(json.dumps({'id':'sumu143/transformer-native-3d-skinae-cross-validation','title':'Transformer Native 3D SkinAE Cross Validation','code_file':'threshold.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':['sumu143/t3d-text-occupancy-pilot-checkpoint'],'competition_sources':[],'kernel_sources':[],'model_sources':[]},indent=2)+'\n');print(digest)
