"""Compare two reconstructed voxel surfaces; not an upstream benchmark."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from t3d.container import load
from t3d.extraction import voxels_to_mesh
from t3d.metrics import surface_comparison


def main():
 p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--prediction',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=128);args=p.parse_args()
 reference=load(args.reference).asset;prediction=load(args.prediction).asset
 if reference.grid_size!=prediction.grid_size or not np.allclose(reference.aabb,prediction.aabb):raise ValueError('Grid metadata mismatch')
 rv,rf=voxels_to_mesh(reference);pv,pf=voxels_to_mesh(prediction)
 result=surface_comparison(rv,rf,pv,pf,samples=args.samples,seed=71,tolerance=.01)
 result.update(scope='Sampled surface comparison against prepared voxelized reference, not original source mesh or research benchmark',reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),prediction_sha256=hashlib.sha256(args.prediction.read_bytes()).hexdigest(),reference_triangles=len(rf),prediction_triangles=len(pf))
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
