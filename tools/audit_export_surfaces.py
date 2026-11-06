"""Repeat bidirectional surface metrics across fixed RNG seeds for exported GLBs."""
import argparse
import hashlib
import json
from pathlib import Path

from t3d.import_rig import load_rig_glb
from t3d.metrics import surface_comparison


def audit(source_path, output_paths, *, samples=512, seeds=(0,1,2,3,4), tolerance=.01):
    source,_=load_rig_glb(source_path).normalized()
    results={}
    for output_path in output_paths:
        output=load_rig_glb(output_path)
        results[Path(output_path).name]={
            'sha256':hashlib.sha256(Path(output_path).read_bytes()).hexdigest(),
            'vertices':len(output.vertices),'faces':len(output.faces),
            'runs':[surface_comparison(source.vertices,source.faces,output.vertices,
                output.faces,samples=samples,seed=int(seed),tolerance=tolerance)
                for seed in seeds],
        }
    return {
        'source_sha256':hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        'samples_per_run':samples,'seeds':[int(seed) for seed in seeds],
        'tolerance':tolerance,
        'results':results,
        'scope':'Sampled geometric comparison; no rendered appearance or animated-pose metric.',
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_glb',type=Path)
    parser.add_argument('output_glb',type=Path,nargs='+')
    parser.add_argument('--samples',type=int,default=512)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    parser.add_argument('--tolerance',type=float,default=.01)
    parser.add_argument('--json',type=Path)
    args=parser.parse_args()
    report=audit(args.source_glb,args.output_glb,samples=args.samples,
                 seeds=args.seeds,tolerance=args.tolerance)
    rendered=json.dumps(report,indent=2)+'\n'
    if args.json:args.json.write_text(rendered)
    print(rendered,end='')


if __name__=='__main__':main()
