"""Bundle coupled occupancy CUDA probe for Kaggle."""
import base64,hashlib,json
from pathlib import Path
from tools.notebook_bundle import source_bundle
root=Path.cwd();files=[*sorted((root/'t3d').glob('*.py')),root/'tools/train_coupled_real_pilot.py',*sorted((root/'artifacts/coupled-real-pilot').glob('*'))];payload=source_bundle(root,files);digest=hashlib.sha256(payload).hexdigest()
source=('import base64,hashlib,io,pathlib,subprocess,sys,zipfile\n'
 f'payload=base64.b64decode({base64.b64encode(payload).decode()!r})\n'
 f'assert hashlib.sha256(payload).hexdigest()=={digest!r}\n'
 'root=pathlib.Path("/kaggle/working/coupled-probe");root.mkdir(exist_ok=True)\n'
 'with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(root)\n'
 'subprocess.run([sys.executable,"-m","pip","install","hilbertcurve>=2,<3"],check=True)\n'
 'import json\n'
 'for label,flags in (("baseline",[]),("weighted",["--foreground-weight","8"])):\n'
 '  output=root.parent/label\n'
 '  subprocess.run([sys.executable,"-m","tools.train_coupled_real_pilot","--output-dir",str(output),*flags],cwd=root,check=True)\n'
 '  report_path=output/"coupled-real-report.json";report=json.loads(report_path.read_text())\n'
 f'  report["source_sha256"]={digest!r};report_path.write_text(json.dumps(report,indent=2)+"\\n")\n')
out=root/'kaggle-coupled-real-foreground';out.mkdir(exist_ok=True);(out/'coupled.ipynb').write_text(json.dumps({'cells':[{'cell_type':'code','id':'coupled','metadata':{},'execution_count':None,'outputs':[],'source':source}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5},indent=2)+'\n')
(out/'kernel-metadata.json').write_text(json.dumps({'id':'sumu143/transformer-native-3d-skinae-cross-validation','title':'Transformer Native 3D SkinAE Cross Validation','code_file':'coupled.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]},indent=2)+'\n');print(digest)
