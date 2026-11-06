"""CUDA integration probe for coupled occupancy; synthetic, not trained quality."""
import json
from pathlib import Path
import torch
from t3d.generation import ThreeStageFlow
from t3d.coupled_occupancy import closed_shell_solid_mask

def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 torch.manual_seed(19);device='cuda';m=ThreeStageFlow(4,3,8,grid_size=(4,4,4),rig_channels=2,width=48,heads=2,depth=2,patch_size=16,separate_skeleton=True).to(device)
 coords=m.grid_coords;inside=((coords-1.5).abs()<=1.5).all(1);coords=coords[inside];skeleton=coords[(coords[:,0]==1)&(coords[:,2]==1)];shape=torch.randn(len(coords),4,device=device);material=torch.randn(len(coords),3,device=device);rig=torch.randn(len(coords),2,device=device);condition=torch.randn(5,8,device=device)
 optimizer=torch.optim.AdamW(m.parameters(),lr=.002);history=[];adapter_grad=False
 for step in range(8):
  optimizer.zero_grad(set_to_none=True);loss=m.losses(coords,shape,material,condition,rig=rig,skeleton_coords=skeleton,drop_probability=0);total=sum(loss.values())
  if not torch.isfinite(total):raise ValueError('Nonfinite loss')
  total.backward();adapter_grad |= any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.occupancy.shape_to_skeleton.parameters());optimizer.step();history.append({k:float(v.detach()) for k,v in loss.items()})
 if not adapter_grad:raise AssertionError('No adapter gradient')
 generated=m.generate(condition,steps=3,occupancy_threshold=.5);assert not len(generated.skeleton_coords) or (((generated.skeleton_coords>=generated.coords.min(0).values)&(generated.skeleton_coords<=generated.coords.max(0).values)).all())
 try:m.losses(coords[:1],shape[:1],material[:1],condition,rig=rig[:1],skeleton_coords=coords[-1:])
 except ValueError:pass
 else:raise AssertionError('Out-of-shape skeleton accepted')
 grid=torch.cartesian_prod(*(torch.arange(5,device=device) for _ in range(3)))
 shell=((grid>=1)&(grid<=3)).all(1)&((grid==1)|(grid==3)).any(1);center=(grid==2).all(1)
 solid=closed_shell_solid_mask(grid,shell);assert bool(solid[center].item()) and int(solid.sum())==27
 shell[(grid==torch.tensor([1,2,2],device=device)).all(1)]=False
 assert not bool(closed_shell_solid_mask(grid,shell)[center].item())
 report={'closed_shell_fill_passed':True,'open_shell_leak_detected':True,'scope':'Synthetic coupled occupancy CUDA integration probe; not a trained generator or geometric bone-containment benchmark','device':torch.cuda.get_device_name(0),'steps':8,'history':history,'adapter_gradient_nonzero':bool(adapter_grad),'generated_shape_count':len(generated.coords),'generated_skeleton_count':len(generated.skeleton_coords),'skeleton_inside_shape_aabb':True}
 Path('/kaggle/working/coupled-occupancy-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
