"""Explicit surface metrics for CPU integration checks, not research-suite aliases."""
from collections import Counter
import numpy as np
from .materials import closest_surface
from .sampling import sample_conditioning


def surface_comparison(reference_vertices, reference_faces, vertices, faces, *, samples=512, seed=0, tolerance=.01):
    """Area-sampled bidirectional point-to-triangle distances in input world units.

    mean_distance averages the two directional means (unsquared). F-score uses
    nearest-surface distances at tolerance. This is not the blueprint's named
    Mesh Distance or rendered Sharp Normal Error protocol.
    """
    if samples<=0 or tolerance<=0 or not np.isfinite(tolerance):
        raise ValueError('Samples and tolerance must be positive')
    if len(reference_faces)==0 or len(faces)==0:
        return dict(valid=False,reason='No reference or reconstructed triangles',samples=samples)
    for name,v,f in [('reference',reference_vertices,reference_faces),('prediction',vertices,faces)]:
        triangles=np.asarray(v)[np.asarray(f)]
        areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
        if np.any(areas==0) or not np.isfinite(areas).all():
            return dict(valid=False,reason=f'Degenerate or nonfinite {name} triangles',
                        degenerate_triangles=int(np.count_nonzero(areas==0)),samples=samples)
    reference,_=sample_conditioning(reference_vertices,reference_faces,uniform_count=samples,salient_count=0,seed=seed)
    prediction,_=sample_conditioning(vertices,faces,uniform_count=samples,salient_count=0,seed=seed+1)
    def distance(points,v,f):
        v,f=np.asarray(v),np.asarray(f)
        try:
            from .proximity import TriangleBVH
            ids,bary=TriangleBVH(v,f).query(points[:,:3])
        except Exception:
            ids,bary=closest_surface(points[:,:3],v,f)
        nearest=np.einsum('ni,nij->nj',bary,v[f[ids]])
        return np.linalg.norm(nearest-points[:,:3],axis=1)
    recall_dist=distance(reference,vertices,faces)
    precision_dist=distance(prediction,reference_vertices,reference_faces)
    precision=float(np.mean(precision_dist<=tolerance));recall=float(np.mean(recall_dist<=tolerance))
    # GLB UV seams duplicate vertex records. Weld by exact bind-pose position
    # before incidence counting so a seam does not masquerade as a boundary.
    welded_vertices,inverse=np.unique(np.asarray(vertices),axis=0,return_inverse=True)
    welded_faces=inverse[np.asarray(faces,dtype=np.int64)]
    edges=Counter(tuple(sorted((int(a),int(b)))) for tri in welded_faces
                   for a,b in zip(tri,np.roll(tri,-1)))
    boundary=[edge for edge,count in edges.items() if count==1]
    boundary_length=float(sum(np.linalg.norm(welded_vertices[a]-welded_vertices[b])
                              for a,b in boundary))
    return dict(valid=True,samples=samples,tolerance=tolerance,seed=seed,
                mean_distance=float((recall_dist.mean()+precision_dist.mean())/2),
                reference_to_prediction=float(recall_dist.mean()),prediction_to_reference=float(precision_dist.mean()),
                precision=precision,recall=recall,fscore=2*precision*recall/(precision+recall) if precision+recall else 0.,
                triangles=len(faces),boundary_edges=len(boundary),boundary_edge_length=boundary_length,
                mean_boundary_edge_length=boundary_length/len(boundary) if boundary else 0.,
                nonmanifold_edges=sum(n>2 for n in edges.values()))
