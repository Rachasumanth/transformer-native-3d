"""Compare CPU BVH material projection with exhaustive traversal on source GLBs."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from t3d.import_glb import load_glb
from t3d.normalization import normalize_asset
from t3d.materials import closest_surface
from t3d.proximity import TriangleBVH


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('sources',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,default=Path('artifacts/proximity-benchmark.json'))
    args=parser.parse_args()
    results=[]
    for source in args.sources:
        mesh=load_glb(source)
        vertices,_,_=normalize_asset(mesh.vertices)
        vertices=np.asarray(vertices)
        rng=np.random.default_rng(917)
        points=vertices[rng.integers(0,len(vertices),256)]+rng.normal(0,.01,(256,3))
        start=time.perf_counter();ids,bc=closest_surface(points,vertices,mesh.faces,method='brute')
        brute_seconds=time.perf_counter()-start
        start=time.perf_counter();tree=TriangleBVH(vertices,mesh.faces)
        build_seconds=time.perf_counter()-start
        stats={}
        start=time.perf_counter();other_ids,other_bc=tree.query(points,diagnostics=stats)
        bvh_seconds=time.perf_counter()-start
        original=np.einsum('ni,nij->nj',bc,vertices[mesh.faces[ids]])
        projected=np.einsum('ni,nij->nj',other_bc,vertices[mesh.faces[other_ids]])
        max_position_error=float(np.abs(original-projected).max())
        max_distance_error=float(np.abs(np.linalg.norm(original-points,axis=1)-np.linalg.norm(projected-points,axis=1)).max())
        row=dict(source_sha256=mesh.source_sha256,source=str(source),faces=len(mesh.faces),queries=len(points),
            brute_seconds=brute_seconds,bvh_build_seconds=build_seconds,bvh_query_seconds=bvh_seconds,
            max_position_error=max_position_error,max_distance_error=max_distance_error,
            face_ids_equal=bool(np.array_equal(ids,other_ids)),**stats)
        results.append(row)
        print(json.dumps(row),flush=True)
        assert max_position_error<1e-10 and max_distance_error<1e-10
        assert np.array_equal(ids,other_ids),'source-face selection differs from exhaustive traversal'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(seed=917,results=results),indent=2)+'\n')


if __name__=='__main__':main()
