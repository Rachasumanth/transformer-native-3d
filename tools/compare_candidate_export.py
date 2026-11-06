"""Compare a newly downloaded CesiumMan export with the archived v13 audit."""
import argparse
import json
from pathlib import Path
from tools.audit_posed_deformation import audit_posed_deformation


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('export',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    source=Path('artifacts/rigged-inputs/CesiumMan/CesiumMan.glb')
    candidate=audit_posed_deformation(source,args.export,
        animated_path=args.output_dir/'candidate-animated.glb')
    baseline=json.loads(Path('artifacts/posed-deformation/cesiumman-v13-posed-deformation.json').read_text())
    report=dict(scope='Single CesiumMan reconstructed export; asset used for checkpoint selection; surface metrics are not rendered quality',
        baseline=baseline['summary'],candidate=candidate.get('summary'),candidate_valid=candidate['valid'],
        baseline_sha256=baseline['output']['sha256'],candidate_sha256=candidate['output']['sha256'])
    (args.output_dir/'candidate-posed-deformation.json').write_text(json.dumps(candidate,indent=2)+'\n')
    (args.output_dir/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
