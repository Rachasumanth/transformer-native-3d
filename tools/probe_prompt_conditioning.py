"""T4 real pretrained encoder smoke probe; no alignment or generation claim."""
import hashlib,json
from pathlib import Path
import torch
from PIL import Image
from t3d.conditioning import PromptConditioner
from t3d.generation import ThreeStageFlow

def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 torch.manual_seed(13);device='cuda';im=Path('artifacts/coupled-real-pilot/cesiumman-view.png')
 conditioner=PromptConditioner.from_pretrained(image_model='facebook/dinov2-small',image_revision='ed25f3a31f01632728cabb09d1542f84ab7b0056',text_model='openai/clip-vit-base-patch32',text_revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',output_channels=32).to(device).train()
 image=conditioner.encode_image(Image.open(im).convert('RGB'));text=conditioner.encode_text('a small astronaut in a white spacesuit');assert image.shape[1]==text.shape[1]==32 and torch.isfinite(image).all() and torch.isfinite(text).all()
 (image.square().mean()+text.square().mean()).backward();assert conditioner.image_projection.weight.grad is not None and conditioner.text_projection.weight.grad is not None
 assert all(p.grad is None for p in conditioner.image_encoder.parameters()) and all(p.grad is None for p in conditioner.text_encoder.parameters())
 model=ThreeStageFlow(2,2,32,grid_size=(2,2,2),width=24,heads=2,depth=1,patch_size=4,separate_skeleton=True).to(device);coords=torch.tensor([[0,0,0],[1,1,1]],device=device);shape=torch.randn(2,2,device=device);mat=torch.randn(2,2,device=device)
 summaries={}
 for label,tokens in (('image',image.detach()),('text',text.detach())):
  values=model.losses(coords,shape,mat,tokens,skeleton_coords=coords[:1],drop_probability=0);assert all(torch.isfinite(v) for v in values.values());summaries[label]={k:float(v.detach()) for k,v in values.items()}
 report=dict(scope='One rendered image and one text prompt through frozen pretrained encoders and untrained flow; no alignment, training quality or generation claim',device=torch.cuda.get_device_name(0),image_sha256=hashlib.sha256(im.read_bytes()).hexdigest(),models=conditioner.model_provenance,image_tokens=list(image.shape),text_tokens=list(text.shape),backbones_frozen=True,projection_gradients=True,flow_losses=summaries)
 Path('/kaggle/working/prompt-conditioning-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
