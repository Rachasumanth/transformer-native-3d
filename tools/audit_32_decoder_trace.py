"""Audit free decoder pruning across levels of a prepared-asset checkpoint."""
import argparse,hashlib,json
from pathlib import Path
import torch
from t3d.dataset import AssetDataset,batch_arrays
from t3d.sc_vae import SparseCompressionVAE
from tools.audit_decoder_trace import stage_counts


def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--reference-backend',action='store_true',help='Run checkpoint weights with reference convolution');args=p.parse_args()
 if args.device=='cpu':torch.set_num_threads(2)
 state=torch.load(args.checkpoint,map_location='cpu',weights_only=False);config=dict(state['config'])
 if args.reference_backend:config['convolution_backend']='reference'
 m=SparseCompressionVAE(**config).to(args.device);m.load_state_dict(state['model']);m.eval();rows=[]
 for split in ('train','validation'):
  ds=AssetDataset(args.manifest,split=split)
  if state['manifest_sha256']!=ds.fingerprint:raise ValueError('Dataset fingerprint mismatch')
  asset=ds[0];coords,features=batch_arrays([asset],'shape');coords=torch.from_numpy(coords).to(args.device);features=torch.from_numpy(features).to(args.device)
  with torch.inference_mode():
   posterior=m.encode(coords,features);trace=[];m.decode(posterior.coords,posterior.mean,trace=trace)
  stages=[]
  for i,entry in enumerate(trace):
   stats=stage_counts(entry['parents'][:,1:].cpu().numpy(),entry['children'][:,1:].cpu().numpy(),asset.coords,8//(2**i));stats['stage']=i;stages.append(stats)
  rows.append(dict(split=split,asset=ds.records[0].asset_id,stages=stages))
 result=dict(scope='One 30-step checkpoint and two previously inspected assets; free decoder at threshold .5, posterior means; not generation',checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),manifest_sha256=ds.fingerprint,execution_backend=config.get('convolution_backend','reference'),rows=rows)
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
