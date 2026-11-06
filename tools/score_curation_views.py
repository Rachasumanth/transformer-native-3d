"""Score multi-view curation images with frozen CLIP; not a validated filter."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

MODEL = 'openai/clip-vit-base-patch32'
REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'


def prepare_image(im, *, crop_foreground=False):
    rgba = im.convert('RGBA')
    if crop_foreground:
        bounds = rgba.getchannel('A').getbbox()
        if bounds is None:
            raise ValueError('Image has no foreground')
        margin = max(1, round(max(bounds[2]-bounds[0], bounds[3]-bounds[1]) * .1))
        bounds = (max(0,bounds[0]-margin), max(0,bounds[1]-margin),
          min(rgba.width,bounds[2]+margin), min(rgba.height,bounds[3]+margin))
        rgba = rgba.crop(bounds)
    if crop_foreground:
        side = max(rgba.size)
        canvas = Image.new('RGBA', (side,side), (192,192,192,255))
        canvas.alpha_composite(rgba, ((side-rgba.width)//2, (side-rgba.height)//2))
    else:
        canvas = Image.new('RGBA', rgba.size, (192,192,192,255))
        canvas.alpha_composite(rgba)
    return canvas.convert('RGB')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--renders', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True, help='JSON object mapping category names to nonempty prompt lists')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--crop-foreground', action='store_true', help='Crop to alpha bounds with a 10 percent margin')
    args = parser.parse_args()
    root = args.renders.resolve(strict=True)
    raw_manifest = (root/'manifest.json').read_bytes()
    manifest = json.loads(raw_manifest)
    prompts_raw = args.prompts.read_bytes()
    categories = json.loads(prompts_raw)
    if not isinstance(categories, dict) or len(categories) < 2:
        raise ValueError('At least two prompt categories are required')
    if any(not isinstance(k,str) or not k.strip() or not isinstance(v,list) or not v or
           any(not isinstance(x,str) or not x.strip() for x in v) for k,v in categories.items()):
        raise ValueError('Categories need names and nonempty lists of prompt strings')
    views = manifest['views']
    if not views:
        raise ValueError('Render manifest has no views')
    images = []
    names = []
    for view in views:
        files = [(name, digest) for name,digest in view['files'].items() if name.endswith('-color.png')]
        if len(files) != 1:
            raise ValueError('Expected exactly one color image per view')
        name, digest = files[0]
        path = (root/name).resolve(strict=True)
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Render image path or hash mismatch')
        with Image.open(path) as im:
            images.append(prepare_image(im, crop_foreground=args.crop_foreground))
        names.append(name)
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(MODEL, revision=REVISION).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL, revision=REVISION)
    prompts = [prompt for group in categories.values() for prompt in group]
    with torch.inference_mode():
        image_inputs = processor(images=images, return_tensors='pt').to(args.device)
        text_inputs = processor(text=prompts, padding=True, truncation=True, return_tensors='pt').to(args.device)
        image_vectors = model.get_image_features(**image_inputs)
        text_vectors = model.get_text_features(**text_inputs)
        image_vectors = image_vectors / image_vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        text_vectors = text_vectors / text_vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        similarity = (image_vectors @ text_vectors.T).cpu().numpy()
    if not np.isfinite(similarity).all():
        raise ValueError('Nonfinite embedding similarities')
    columns = {}
    offset = 0
    for label, group in categories.items():
        columns[label] = similarity[:,offset:offset+len(group)].mean(axis=1)
        offset += len(group)
    result = dict(scope='Uncalibrated CLIP cosine similarities, not decisions or verified quality labels',
                  model=MODEL, revision=REVISION, device=str(args.device), crop_foreground=args.crop_foreground,
                  render_manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
                  prompts_sha256=hashlib.sha256(prompts_raw).hexdigest(),
                  categories=categories,
                  views=[dict(image=name, scores={k:float(v[i]) for k,v in columns.items()}) for i,name in enumerate(names)],
                  mean_scores={k:float(v.mean()) for k,v in columns.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
