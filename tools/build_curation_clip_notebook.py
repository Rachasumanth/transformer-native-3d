"""Build a small GPU probe of uncalibrated multi-view CLIP scoring."""
import base64,hashlib,json
from pathlib import Path
from tools.notebook_bundle import source_bundle
root=Path.cwd();names=('avocado','cube','sphere','plane');files=[root/'tools/score_curation_views.py',root/'artifacts/curation-clip-probe/prompts.json']
for name in names:
 p=root/'artifacts'/('curation-render-'+name);files.append(p/'manifest.json');files.extend(sorted(p.glob('*-color.png')))
payload=source_bundle(root,files);digest=hashlib.sha256(payload).hexdigest()
source=('import base64,hashlib,io,pathlib,subprocess,sys,zipfile,json\n'
 f'payload=base64.b64decode({base64.b64encode(payload).decode()!r})\n'
 f'assert hashlib.sha256(payload).hexdigest()=={digest!r}\n'
 'root=pathlib.Path("/kaggle/working/curation-probe");root.mkdir(exist_ok=True)\n'
 'with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)\n'
 'subprocess.run([sys.executable,"-m","pip","install","transformers>=4.45,<5","safetensors"],check=True)\n'
 f'for name in {names!r}:\n'
 '  subprocess.run([sys.executable,"tools/score_curation_views.py","--renders",str(root/"artifacts"/("curation-render-"+name)),"--prompts",str(root/"artifacts/curation-clip-probe/prompts.json"),"--output",str(root.parent/("curation-scores-"+name+".json"))],cwd=root,check=True)\n'
 f'  p=root.parent/("curation-scores-"+name+".json");r=json.loads(p.read_text());r["source_sha256"]={digest!r};p.write_text(json.dumps(r,indent=2)+"\\n")\n')
out=root/'kaggle-curation-clip';out.mkdir(exist_ok=True)
(out/'curation.ipynb').write_text(json.dumps({'cells':[{'cell_type':'code','id':'curation','metadata':{},'execution_count':None,'outputs':[],'source':source}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5},indent=2)+'\n')
(out/'kernel-metadata.json').write_text(json.dumps({'id':'sumu143/transformer-native-3d-skinae-cross-validation','title':'Transformer Native 3D SkinAE Cross Validation','code_file':'curation.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]},indent=2)+'\n');print(digest)
