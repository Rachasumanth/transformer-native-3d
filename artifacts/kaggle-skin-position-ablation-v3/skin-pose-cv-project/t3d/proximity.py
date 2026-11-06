"""CPU triangle-AABB hierarchy for exact source-surface projection."""
import numpy as np
from .sampling import _mesh


class TriangleBVH:
    def __init__(self,vertices,faces,*,leaf_size=8):
        if type(leaf_size) is not int or leaf_size<1:raise ValueError('leaf_size must be positive')
        vertices,faces,_,_=_mesh(vertices,faces)
        self.vertices=vertices.copy();self.faces=faces.copy()
        triangles=self.vertices[self.faces]
        low=triangles.min(1);high=triangles.max(1)
        centers=(low+high)/2
        self.nodes=[]
        def build(ids):
            index=len(self.nodes)
            bounds=(low[ids].min(0),high[ids].max(0))
            self.nodes.append(None)
            if len(ids)<=leaf_size:
                self.nodes[index]=(bounds,None,None,np.sort(ids))
            else:
                axis=np.ptp(centers[ids],axis=0).argmax()
                order=ids[np.argsort(centers[ids,axis],kind='stable')]
                mid=len(order)//2
                left=build(order[:mid]);right=build(order[mid:])
                self.nodes[index]=(bounds,left,right,None)
            return index
        build(np.arange(len(faces)))

    def query(self,points,*,diagnostics=None):
        from .materials import _closest_surface_brute
        points=np.asarray(points,dtype=float)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
            raise ValueError('Queries must be finite (N,3)')
        ids=np.full(len(points),np.iinfo(np.int64).max,dtype=np.int64)
        bary=np.zeros((len(points),3))
        best=np.full(len(points),np.inf)
        evaluated=0
        stack=[(0,np.arange(len(points)))]
        while stack:
            index,queries=stack.pop()
            bounds,left,right,triangles=self.nodes[index]
            delta=np.maximum(np.maximum(bounds[0]-points[queries],points[queries]-bounds[1]),0)
            lower=(delta*delta).sum(1)
            # Conservative floating-point slack only affects traversal, never
            # the exact closest-distance or source-face tie comparisons.
            threshold=best[queries]+64*np.finfo(float).eps*np.maximum(1,best[queries])
            queries=queries[lower<=threshold]
            if not len(queries):continue
            if triangles is None:
                stack.append((right,queries));stack.append((left,queries))
                continue
            local,bc,distance=_closest_surface_brute(points[queries],self.vertices,self.faces[triangles],
                                                   return_distance=True)
            candidate=triangles[local]
            take=(distance<best[queries])|((distance==best[queries])&(candidate<ids[queries]))
            selected=queries[take]
            best[selected]=distance[take];ids[selected]=candidate[take];bary[selected]=bc[take]
            evaluated+=len(queries)*len(triangles)
        if diagnostics is not None:
            diagnostics.update(nodes=len(self.nodes),triangle_query_pairs=evaluated,
                               exhaustive_pairs=len(points)*len(self.faces))
        return ids,bary
