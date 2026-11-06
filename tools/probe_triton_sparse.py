"""T4 parity and runtime probe for experimental sparse convolution."""
import json,time
from pathlib import Path
import torch
from t3d.sparse_blocks import SubmanifoldConv3d
from t3d.triton_sparse import triton_submanifold_conv


def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 import triton
 torch.backends.cuda.matmul.allow_tf32=False
 torch.manual_seed(713);device='cuda';cases=[]
 for n,c in ((31,4),(128,16),(512,33)):
  all_coords=torch.cartesian_prod(torch.arange(2),torch.arange(10),torch.arange(10),torch.arange(10));coords=all_coords[torch.randperm(len(all_coords))[:n]].to(device)
  f=torch.randn(n,c,device=device,requires_grad=True);conv=SubmanifoldConv3d(c).to(device);w=conv.weight;b=conv.bias;g=torch.randn_like(f)
  expected=conv(coords,f);expected_grads=torch.autograd.grad(expected,(f,w,b),g,retain_graph=True)
  actual=triton_submanifold_conv(coords,f,w,b);actual_grads=torch.autograd.grad(actual,(f,w,b),g)
  errors=[float((expected-actual).abs().max())]+[float((x-y).abs().max()) for x,y in zip(expected_grads,actual_grads)]
  torch.testing.assert_close(actual,expected,atol=5e-5,rtol=5e-5)
  for x,y in zip(expected_grads,actual_grads):torch.testing.assert_close(x,y,atol=1e-4,rtol=1e-4)
  def bench(fn):
   with torch.no_grad():
    for _ in range(2):fn()
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(5):fn()
    torch.cuda.synchronize();return (time.perf_counter()-start)*200
  cases.append(dict(n=n,channels=c,max_abs_error_forward_grad_features_grad_weight_grad_bias=errors,reference_ms=bench(lambda:conv(coords,f)),experimental_ms=bench(lambda:triton_submanifold_conv(coords,f,w,b))))
 result=dict(scope='One T4; tiny random sparse grids; forward timing includes validation and neighbor-map construction; backward uses PyTorch; not a production benchmark',device=torch.cuda.get_device_name(0),torch=torch.__version__,triton=triton.__version__,cases=cases)
 Path('/kaggle/working/triton-sparse-probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
