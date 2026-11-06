"""Limit an existing GLB's SkinAE weights to a chosen number of influences."""
import argparse
import hashlib
import json
from pathlib import Path

from t3d.glb import rewrite_skin_weights
from t3d.import_rig import load_rig_glb


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--max-influences',type=int,default=2)
    args=parser.parse_args()
    if args.input.resolve()==args.output.resolve():
        parser.error('input must differ from output')
    source=args.input.read_bytes();rig=load_rig_glb(args.input)
    result,audit=rewrite_skin_weights(source,rig.weights,
                                     max_influences=args.max_influences)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(result)
    audit.update(input=str(args.input),input_sha256=hashlib.sha256(source).hexdigest(),
                 output=str(args.output),output_sha256=hashlib.sha256(result).hexdigest())
    report=args.output.with_suffix(args.output.suffix+'.json')
    report.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
