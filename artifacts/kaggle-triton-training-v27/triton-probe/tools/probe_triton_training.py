"""Check opt-in Triton SC-VAE training, parity and checkpoint resume on T4."""
import json
from pathlib import Path
import torch
from t3d.dataset import AssetDataset
from t3d.training import VAETrainer


def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(812)
 ds=AssetDataset('artifacts/pilot-dataset/manifest.json',split='train');asset=ds[0]
 opts=dict(mode='shape',channels=(4,8,16,32,64),latent_channels=8,device='cuda',manifest_sha256=ds.fingerprint)
 a=VAETrainer(**opts);b=VAETrainer(**opts,convolution_backend='triton');b.model.load_state_dict(a.model.state_dict());rows=[]
 for step in range(3):
  torch.manual_seed(300+step);torch.cuda.manual_seed_all(300+step);x=a.update([asset])
  torch.manual_seed(300+step);torch.cuda.manual_seed_all(300+step);y=b.update([asset])
  if abs(x['total']-y['total'])>2e-4:raise AssertionError('Training loss parity failed')
  errors=[float((p-q).abs().max()) for p,q in zip(a.model.parameters(),b.model.parameters())]
  if max(errors)>3e-4:raise AssertionError('Training parameter parity failed')
  rows.append(dict(step=step+1,reference=x,experimental=y,max_parameter_error=max(errors)))
 checkpoint=Path('/kaggle/working/triton-trainer.pt');b.save(checkpoint);expected=b.update([asset]);restored=VAETrainer.resume(checkpoint,device='cuda',manifest_sha256=ds.fingerprint);actual=restored.update([asset])
 if restored.config.get('convolution_backend')!='triton':raise AssertionError('Backend not restored')
 for name in expected:
  if abs(expected[name]-actual[name])>1e-6:raise AssertionError('Resume metrics mismatch')
 for p,q in zip(b.model.parameters(),restored.model.parameters()):torch.testing.assert_close(p,q,atol=1e-6,rtol=1e-6)
 result=dict(scope='Three paired stochastic updates and checkpoint resume on one 16-cubed prepared asset; no trained model quality claim',device=torch.cuda.get_device_name(0),manifest_sha256=ds.fingerprint,rows=rows,resume_expected=expected,resume_actual=actual,restored_backend=restored.config['convolution_backend'])
 Path('/kaggle/working/triton-training-probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
