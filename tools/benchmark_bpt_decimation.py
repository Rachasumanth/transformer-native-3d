"""Measure explicit quadric simplification as a possible BPT data-prep step."""
import json
from pathlib import Path
import time
import numpy as np
import trimesh

from t3d.bpt import quantization_diagnostics
from t3d.import_glb import load_glb
from t3d.normalization import normalize_asset
from tools.audit_bpt_corpus import load_geometry_only_glb
from t3d.proximity import TriangleBVH


def fast_surface_comparison(reference_vertices, reference_faces, vertices, faces,
                            *, samples=128, seed=0):
    """Use the exact CPU BVH for a small, deterministic sampled comparison."""
    def sample(v, f, rng):
        tri = v[f]
        areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                        tri[:, 2] - tri[:, 0]), axis=1) / 2
        owners = rng.choice(len(f), size=samples, p=areas / areas.sum())
        uv = rng.random((samples, 2))
        root = np.sqrt(uv[:, 0])
        bary = np.column_stack((1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]))
        return np.einsum('ni,nij->nj', bary, tri[owners])

    reference = sample(reference_vertices, reference_faces, np.random.default_rng(seed))
    prediction = sample(vertices, faces, np.random.default_rng(seed + 1))
    ref_bvh = TriangleBVH(reference_vertices, reference_faces)
    pred_bvh = TriangleBVH(vertices, faces)

    def distances(points, bvh):
        ids, bary = bvh.query(points[:, :3])
        nearest = np.einsum('ni,nij->nj', bary, bvh.vertices[bvh.faces[ids]])
        return np.linalg.norm(nearest - points[:, :3], axis=1)

    recall = distances(reference, pred_bvh)
    precision = distances(prediction, ref_bvh)
    p, r = float(np.mean(precision <= .01)), float(np.mean(recall <= .01))
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return dict(mean_distance=float((precision.mean() + recall.mean()) / 2),
                fscore=2 * p * r / (p + r) if p + r else 0.,
                boundary_edges=int(np.count_nonzero(counts == 1)),
                nonmanifold_edges=int(np.count_nonzero(counts > 2)))


def load_static_geometry(path):
    try:
        source=load_glb(path)
        return source.vertices,source.faces,source.source_sha256
    except ValueError:
        return load_geometry_only_glb(path)


def main():
    root=Path(__file__).resolve().parents[1]
    candidate_root=root/'artifacts/bpt-candidates'
    cases=[]
    for name in ('BarramundiFish','BoomBox','SheenChair'):
        path=candidate_root/name/f'{name}.glb'
        vertices,faces,digest=load_static_geometry(path)
        source_vertices=np.clip(normalize_asset(vertices)[0],-.5,.5)
        source_mesh=trimesh.Trimesh(vertices=source_vertices,faces=faces,process=False)
        target_counts=sorted({int(x) for x in (20000,16000,12000,10000,8192,6000,4000,2000,1000)
                              if x<len(faces)},reverse=True)
        results=[]
        for target in target_counts:
            started=time.perf_counter()
            simplified=source_mesh.simplify_quadric_decimation(face_count=target,aggression=0)
            seconds=time.perf_counter()-started
            simp_vertices=np.asarray(simplified.vertices)
            simp_faces=np.asarray(simplified.faces,dtype=np.int64)
            simp_vertices=np.clip(normalize_asset(simp_vertices)[0],-.5,.5)
            diag=quantization_diagnostics(simp_vertices,simp_faces,bins=128)
            metrics=[fast_surface_comparison(source_vertices,faces,simp_vertices,simp_faces,
                                             samples=128,seed=0)]
            row=dict(target_faces=target,actual_faces=len(simp_faces),vertices=len(simp_vertices),
                simplification_seconds=seconds,quantization=diag,
                mean_surface_distance=float(np.mean([m['mean_distance'] for m in metrics])),
                mean_fscore=float(np.mean([m['fscore'] for m in metrics])),
                edge_topology=dict(boundary_edges=metrics[0]['boundary_edges'],
                                   nonmanifold_edges=metrics[0]['nonmanifold_edges']))
            results.append(row)
            print(name,row,flush=True)
        cases.append(dict(asset=name,source_sha256=digest,source_faces=len(faces),cases=results))
    report=dict(scope='three Khronos static candidates; explicit QEM simplification, source-relative sampled metrics',
        quantization='128-bin BPT preflight; source face removal is explicit via simplification and actual face count',
        cases=cases)
    out=root/'artifacts/bpt/decimation-audit.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Wrote {out}')


if __name__=='__main__':main()
