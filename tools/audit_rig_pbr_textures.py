"""Report embedded PBR texture coverage before and after rigged GLB export."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from t3d.import_rig import load_rig_glb


def _material_report(material):
    def texture_stats(value):
        if value is None:return None
        rgb=value[...,:3]
        return {'shape':list(value.shape),'min':float(rgb.min()),'max':float(rgb.max()),
                'channel_std':rgb.std(axis=(0,1)).tolist(),
                'unique_rgb_count':int(len(np.unique(np.rint(rgb*255).astype(np.uint8).reshape(-1,3),axis=0)))}
    return {'base_color_factor':np.asarray(material.base_color).tolist(),
            'metallic_factor':float(material.metallic),'roughness_factor':float(material.roughness),
            'alpha_mode':material.alpha_mode,
            'base_color_texture':texture_stats(material.base_texture),
            'metallic_roughness_texture':texture_stats(material.metallic_roughness_texture)}


def audit(source_path, output_path):
    source=load_rig_glb(source_path,include_pbr=True)
    output=load_rig_glb(output_path,include_pbr=True)
    return {
        'source_sha256':hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        'output_sha256':hashlib.sha256(Path(output_path).read_bytes()).hexdigest(),
        'source_face_count':len(source.faces),'output_face_count':len(output.faces),
        'source_materials':[_material_report(m) for m in source.pbr_materials],
        'output_materials':[_material_report(m) for m in output.pbr_materials],
        'source_face_material_ids':np.unique(source.face_materials,return_counts=True)[0].tolist(),
        'output_face_material_ids':np.unique(output.face_materials,return_counts=True)[0].tolist(),
        'scope':'Structural texture retention and variation only; not a rendered material-fidelity metric.',
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_glb',type=Path)
    parser.add_argument('output_glb',type=Path)
    parser.add_argument('--json',type=Path)
    args=parser.parse_args()
    report=audit(args.source_glb,args.output_glb)
    rendered=json.dumps(report,indent=2)+'\n'
    if args.json:args.json.write_text(rendered)
    print(rendered,end='')


if __name__=='__main__':main()
