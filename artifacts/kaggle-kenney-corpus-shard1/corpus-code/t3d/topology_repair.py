"""Experimental, guarded exact-weld local topology repair.

Geometry-only: callers must transfer UVs, normals and skin data and verify
self-intersections and surface quality. It is not enabled by the exporter.
"""
from collections import Counter, defaultdict
import numpy as np
from .proximity import TriangleBVH


def repair_local_fans(vertices, faces, *, max_removed_fraction=0.03, allow_open=False):
    """Remove duplicate sheets and close local nonmanifold one-rings.

    Returns welded vertices, faces, and an audit including kept input face IDs.
    Exact welding merges touching sheets; use only where this is appropriate.
    Rejects open input by default. With allow_open=True, preserves existing
    boundaries and rejects any repair neighborhood touching their vertices.
    Also rejects large neighborhoods and non-simple repair boundaries.
    """
    v=np.asarray(vertices,dtype=np.float64);f=np.asarray(faces,dtype=np.int64)
    if (v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or not len(f)
            or not np.isfinite(v).all() or np.any(f<0) or np.any(f>=len(v))
            or type(allow_open) is not bool or not 0<max_removed_fraction<=1):raise ValueError('Invalid mesh or repair budget')
    u,inv=np.unique(v,axis=0,return_inverse=True);wf=inv[f]
    triangles=u[wf];norm=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);length=np.linalg.norm(norm,axis=1)
    if np.any(length==0):raise ValueError('Degenerate input triangle')
    norm/=length[:,None]
    def incidence(faces):
        result=defaultdict(list)
        for i,tri in enumerate(faces):
            for a,b in zip(tri,np.roll(tri,-1)):result[tuple(sorted((int(a),int(b))))].append(i)
        return result
    original=incidence(wf)
    original_boundary={edge for edge,ids in original.items() if len(ids)==1}
    if original_boundary and not allow_open:raise ValueError('Open input mesh is unsupported')
    protected={x for edge in original_boundary for x in edge}
    groups=defaultdict(list)
    for i,t in enumerate(wf):groups[tuple(sorted(map(int,t)))].append(i)
    remove=set()
    for group in groups.values():
        if len(group)<2:continue
        # Match the experimentally investigated rule; reject ambiguous overlap.
        anchor=group[0]
        for other in group[1:]:
            dot=float(np.dot(norm[anchor],norm[other]))
            if dot<-.999:remove.update((anchor,other))
            elif dot>.999:remove.add(other)
            else:raise ValueError('Ambiguous coincident face orientations')
    keep=np.array([i for i in range(len(f)) if i not in remove],dtype=np.int64)
    cleaned=wf[keep];edges=incidence(cleaned);bad=[ids for ids in edges.values() if len(ids)>2]
    if bad:
        region=set(i for ids in bad for i in ids)
        touched=set(int(x) for i in region for x in cleaned[i])
        region|={i for i,t in enumerate(cleaned) if any(int(x) in touched for x in t)}
    else:region=set()
    removed_ids=remove.union(int(keep[i]) for i in region)
    if any(int(vertex) in protected for i in removed_ids for vertex in wf[i]):
        raise ValueError('Repair neighborhood touches an existing boundary')
    if (len(remove)+len(region))/len(f)>max_removed_fraction:raise ValueError('Repair exceeds face removal budget')
    keep_after=np.array([i for i in range(len(cleaned)) if i not in region],dtype=np.int64)
    kept=cleaned[keep_after];edges=incidence(kept)
    if any(len(ids)>2 for ids in edges.values()):raise ValueError('Repair neighborhood leaves nonmanifold edges')
    current_boundary={edge for edge,ids in edges.items() if len(ids)==1}
    if not original_boundary.issubset(current_boundary):raise ValueError('Original boundary changed')
    boundary=sorted(current_boundary-original_boundary)
    if any(x in protected for edge in boundary for x in edge):raise ValueError('Patch intersects an existing boundary')
    adj=defaultdict(set)
    for a,b in boundary:adj[a].add(b);adj[b].add(a)
    if any(len(neighbors)!=2 for neighbors in adj.values()):raise ValueError('Boundary is not a set of simple loops')
    loops=[];seen=set()
    for start in sorted(adj):
        if start in seen:continue
        loop=[start];prev=None;cur=start
        while True:
            seen.add(cur);nxt=min(x for x in adj[cur] if x!=prev)
            if nxt==start:break
            if nxt in seen:raise ValueError('Boundary loops intersect')
            loop.append(nxt);prev,cur=cur,nxt
        loops.append(loop)
    bvh=TriangleBVH(v,f);new_vertices=list(u);new_faces=[]
    for loop in loops:
        point=np.mean(u[loop],axis=0)[None,:];ids,bary=bvh.query(point)
        center=np.einsum('ni,nij->nj',bary,v[f[ids]])[0];ci=len(new_vertices);new_vertices.append(center)
        for j,a in enumerate(loop):
            b=loop[(j+1)%len(loop)];face=kept[edges[tuple(sorted((a,b)))][0]]
            oriented=any(int(x)==a and int(y)==b for x,y in zip(face,np.roll(face,-1)))
            new_faces.append((b,a,ci) if oriented else (a,b,ci))
    outv=np.asarray(new_vertices);outf=np.vstack((kept,np.asarray(new_faces,dtype=np.int64).reshape(-1,3)))
    final=incidence(outf)
    final_boundary={edge for edge,ids in final.items() if len(ids)==1}
    if final_boundary!=original_boundary or any(len(ids)>2 for ids in final.values()):
        raise ValueError('Output changed existing boundaries or has nonmanifold edges')
    patch=outv[outf];area=np.linalg.norm(np.cross(patch[:,1]-patch[:,0],patch[:,2]-patch[:,0]),axis=1)
    if np.any(area==0):raise ValueError('Repair generated degenerate faces')
    return outv,outf,dict(removed_duplicate_faces=sorted(remove),removed_region_faces=keep[sorted(region)].tolist(),
        kept_input_faces=keep[keep_after].tolist(),loop_lengths=[len(x) for x in loops],added_faces=len(new_faces),
        input_faces=len(f),output_faces=len(outf),min_double_area=float(area.min()),
        near_degenerate_faces=int(np.count_nonzero(area<=1e-12)),
        exact_weld_boundary_edges=len(final_boundary),exact_weld_nonmanifold_edges=0)
