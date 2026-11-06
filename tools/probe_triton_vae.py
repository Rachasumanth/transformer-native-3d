"""Compare experimental Triton blocks inside a tiny real-asset SC-VAE."""
import json,time,types
from pathlib import Path
import torch
from t3d.dataset import AssetDataset,batch_arrays
from t3d.sc_vae import SparseCompressionVAE,reconstruction_loss
from t3d.sparse_blocks import SubmanifoldConv3d
from t3d.triton_sparse import triton_submanifold_conv


def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(912)
 dataset=AssetDataset('artifacts/pilot-dataset/manifest.json',split='train');coords,features=batch_arrays([dataset[0]],'shape');coords=torch.from_numpy(coords).cuda();features=torch.from_numpy(features).cuda()
 reference=SparseCompressionVAE('shape',channels=(4,8,16,32,64),latent_channels=8).cuda();experimental=SparseCompressionVAE('shape',channels=(4,8,16,32,64),latent_channels=8).cuda();experimental.load_state_dict(reference.state_dict())
 def forward(self,coords,features):return triton_submanifold_conv(coords,features,self.weight,self.bias)
 for block in experimental.modules():
  if isinstance(block,SubmanifoldConv3d):block.forward=types.MethodType(forward,block)
 def pass_once(model):
  model.zero_grad(set_to_none=True);loss,_=reconstruction_loss(model,coords,features,stochastic=False);loss.backward();return loss
 l1=pass_once(reference);l2=pass_once(experimental);torch.testing.assert_close(l1,l2,atol=2e-5,rtol=2e-5)
 errors=[]
 for (name,p),(other,q) in zip(reference.named_parameters(),experimental.named_parameters()):
  if name!=other or (p.grad is None)!=(q.grad is None):raise AssertionError('Gradient structure mismatch')
  if p.grad is not None:
   torch.testing.assert_close(p.grad,q.grad,atol=2e-4,rtol=2e-4)
   errors.append((name,float((p.grad-q.grad).abs().max())))
 def bench(model):
  pass_once(model);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();start=time.perf_counter()
  for _ in range(3):pass_once(model)
  torch.cuda.synchronize();return dict(ms=(time.perf_counter()-start)*1000/3,peak_extra_bytes=max(0,torch.cuda.max_memory_allocated()-base))
 result=dict(scope='One low-resolution prepared Avocado asset, tiny deterministic teacher-forced VAE; no optimizer step and no learned quality claim',device=torch.cuda.get_device_name(0),manifest_sha256=dataset.fingerprint,active_voxels=len(coords),loss_reference=float(l1),loss_experimental=float(l2),worst_gradient_error=max(errors,key=lambda x:x[1]),reference=bench(reference),experimental=bench(experimental))
 Path('/kaggle/working/triton-vae-probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
