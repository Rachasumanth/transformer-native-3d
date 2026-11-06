"""Frozen DINOv2 image and CLIP text token adapters for flow conditioning.

Hugging Face Transformers is optional and loaded only by from_pretrained.
Projection layers are trainable; foundation backbones remain frozen/eval.
One image or one prompt per call, matching the current one-asset DiT API.
"""
import torch
from torch import nn


class PromptConditioner(nn.Module):
    def __init__(self, image_encoder, text_encoder, image_processor, tokenizer, *, output_channels):
        super().__init__()
        if type(output_channels) is not int or output_channels<=0:raise ValueError('Positive output_channels required')
        self.image_encoder=image_encoder.requires_grad_(False).eval();self.text_encoder=text_encoder.requires_grad_(False).eval()
        self.image_processor=image_processor;self.tokenizer=tokenizer
        self.image_projection=nn.Linear(image_encoder.config.hidden_size,output_channels)
        self.text_projection=nn.Linear(text_encoder.config.hidden_size,output_channels)
        self.output_channels=output_channels

    @classmethod
    def from_pretrained(cls, *, image_model, text_model, image_revision, text_revision, output_channels, local_files_only=False):
        if not all(isinstance(x,str) and x for x in (image_model,text_model,image_revision,text_revision)):
            raise ValueError('Model identifiers and explicit revisions are required')
        from transformers import AutoImageProcessor,AutoTokenizer,CLIPTextModel,CLIPTextConfig,Dinov2Model
        opts=dict(local_files_only=local_files_only)
        ip=AutoImageProcessor.from_pretrained(image_model,revision=image_revision,**opts)
        tokenizer=AutoTokenizer.from_pretrained(text_model,revision=text_revision,**opts)
        image=Dinov2Model.from_pretrained(image_model,revision=image_revision,**opts)
        text_cfg=CLIPTextConfig.from_pretrained(text_model,revision=text_revision,**opts)
        text=CLIPTextModel.from_pretrained(text_model,config=text_cfg,revision=text_revision,**opts)
        obj=cls(image,text,ip,tokenizer,output_channels=output_channels)
        obj.model_provenance=dict(image_model=image_model,image_revision=image_revision,text_model=text_model,text_revision=text_revision)
        return obj

    def train(self, mode=True):
        super().train(mode);self.image_encoder.eval();self.text_encoder.eval();return self

    def encode_image(self,image):
        params=next(self.image_encoder.parameters());inputs=self.image_processor(images=image,return_tensors='pt')
        inputs={k:v.to(params.device) for k,v in inputs.items() if torch.is_tensor(v)}
        with torch.no_grad():tokens=self.image_encoder(**inputs).last_hidden_state
        if tokens.ndim!=3 or tokens.shape[0]!=1 or not tokens.shape[1] or not torch.isfinite(tokens).all():
            raise ValueError('Image encoder must return finite tokens for exactly one image')
        return self.image_projection(tokens[0].to(dtype=self.image_projection.weight.dtype))

    def encode_text(self,text):
        if not isinstance(text,str) or not text.strip():raise ValueError('Nonempty text prompt required')
        params=next(self.text_encoder.parameters());inputs=self.tokenizer(text,return_tensors='pt',padding=False,truncation=True)
        inputs={k:v.to(params.device) for k,v in inputs.items() if torch.is_tensor(v)}
        with torch.no_grad():tokens=self.text_encoder(**inputs).last_hidden_state
        if tokens.ndim!=3 or tokens.shape[0]!=1 or not tokens.shape[1] or not torch.isfinite(tokens).all():
            raise ValueError('Text encoder must return finite tokens for exactly one prompt')
        mask=inputs.get('attention_mask');tokens=tokens[0] if mask is None else tokens[0][mask[0].bool()]
        if not len(tokens):raise ValueError('Text prompt has no valid tokens')
        return self.text_projection(tokens.to(dtype=self.text_projection.weight.dtype))
