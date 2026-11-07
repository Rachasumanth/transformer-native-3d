"""Frozen DINOv2 image and CLIP text token adapters for flow conditioning.

Hugging Face Transformers is optional and loaded only by from_pretrained.
Projection layers are trainable; foundation backbones remain frozen/eval.
One image or one prompt per call, matching the current one-asset DiT API.
"""
import os
from pathlib import Path

import torch
from torch import nn


class PromptConditioner(nn.Module):
    def __init__(self, image_encoder, text_encoder, image_processor, tokenizer, *, output_channels):
        super().__init__()
        if type(output_channels) is not int or output_channels<=0:raise ValueError('Positive output_channels required')
        if text_encoder is None or tokenizer is None:raise ValueError('Text encoder and tokenizer are required')
        self.image_encoder=None if image_encoder is None else image_encoder.requires_grad_(False).eval()
        self.text_encoder=text_encoder.requires_grad_(False).eval()
        self.image_processor=image_processor;self.tokenizer=tokenizer
        self.image_projection=None
        if image_encoder is not None:
            self.image_projection=nn.Linear(image_encoder.config.hidden_size,output_channels)
        self.text_projection=nn.Linear(text_encoder.config.hidden_size,output_channels)
        self.output_channels=output_channels

    @classmethod
    def from_pretrained(cls, *, image_model=None, text_model, image_revision=None, text_revision, output_channels, include_image=True, local_files_only=False):
        if not isinstance(text_model,str) or not text_model or not isinstance(text_revision,str) or not text_revision:
            raise ValueError('Text model identifiers and explicit revisions are required')
        if include_image and (not isinstance(image_model,str) or not image_model or not isinstance(image_revision,str) or not image_revision):
            raise ValueError('Image model identifiers and explicit revisions are required when include_image is True')
        from transformers import AutoImageProcessor,AutoTokenizer,CLIPTextModel,CLIPTextConfig,Dinov2Model
        opts=dict(local_files_only=local_files_only)
        if include_image:
            if image_model is None or image_revision is None:
                raise ValueError('Image model identifiers are required when include_image is True')
            ip=AutoImageProcessor.from_pretrained(image_model,revision=image_revision,**opts)
            image=Dinov2Model.from_pretrained(image_model,revision=image_revision,**opts)
        else:
            ip=None;image=None
        tokenizer=AutoTokenizer.from_pretrained(text_model,revision=text_revision,**opts)
        text_cfg=CLIPTextConfig.from_pretrained(text_model,revision=text_revision,**opts)
        text=CLIPTextModel.from_pretrained(text_model,config=text_cfg,revision=text_revision,**opts)
        obj=cls(image,text,ip,tokenizer,output_channels=output_channels)
        obj.model_provenance=dict(
            image_model=image_model,image_revision=image_revision,
            text_model=str(text_model),text_revision=text_revision,
            include_image=include_image)
        return obj

    @classmethod
    def from_env(cls, *, output_channels):
        """Build the conditioner from pinned remote models or a local T3D_CLIP_DIR snapshot."""
        clip_dir=os.environ.get('T3D_CLIP_DIR')
        if clip_dir:
            local=Path(clip_dir)
            if not local.is_dir() or not (local/'config.json').exists():
                raise FileNotFoundError(f'T3D_CLIP_DIR snapshot missing config.json: {clip_dir}')
            return cls.from_pretrained(
                image_model=None, text_model=str(local), image_revision=None,
                text_revision='local', output_channels=output_channels, include_image=False)
        return cls.from_pretrained(
            image_model='facebook/dinov2-small',
            image_revision='ed25f3a31f01632728cabb09d1542f84ab7b0056',
            text_model='openai/clip-vit-base-patch32',
            text_revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
            output_channels=output_channels)

    def train(self, mode=True):
        super().train(mode);self.text_encoder.eval();return self

    def encode_image(self,image):
        if self.image_encoder is None or self.image_processor is None or self.image_projection is None:
            raise ValueError('Image conditioning was not loaded for this conditioner')
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
