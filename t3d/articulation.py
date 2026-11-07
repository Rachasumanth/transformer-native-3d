"""Deterministic rig preprocessing; no learned SkinAE encoder or decoder here."""
from dataclasses import dataclass, replace
import operator
import numpy as np


def _points(points):
    a = np.array(points, dtype=float, copy=True)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError('Points must be finite (N,3)')
    return a


@dataclass(frozen=True)
class Skeleton:
    joints: np.ndarray
    parents: np.ndarray

    def __post_init__(self):
        joints = _points(self.joints)
        parents = np.array(self.parents, copy=True)
        if (not len(joints) or parents.shape != (len(joints),)
                or not np.issubdtype(parents.dtype, np.integer)
                or np.any(parents < -1) or np.any(parents >= len(joints))):
            raise ValueError('Skeleton needs joints and valid parent indices; roots use -1')
        # Iterative traversal permits arbitrary joint order and deep chains.
        finished = np.zeros(len(joints), dtype=bool)
        for start in range(len(joints)):
            current, path, seen = start, [], set()
            while current != -1 and not finished[current]:
                if current in seen:
                    raise ValueError('Skeleton parent graph contains a cycle')
                seen.add(current)
                path.append(current)
                current = int(parents[current])
            finished[path] = True
        joints.setflags(write=False)
        parents = parents.astype(np.int64)
        parents.setflags(write=False)
        object.__setattr__(self, 'joints', joints)
        object.__setattr__(self, 'parents', parents)

    @property
    def parent_relative_vectors(self):
        parent = np.where(self.parents < 0, np.arange(len(self.parents)), self.parents)
        return self.joints - self.joints[parent]


def skeleton_displacements(points, skeleton, *, chunk_size=4096):
    """Return (N,6) joint-point / parent-point vectors and nearest joint IDs.

    Roots refer to themselves as parent. Exact nearest-joint ties select the
    lowest joint index. Query chunks bound temporary distance storage by K*J.
    This follows the blueprint's nearest-joint convention, without confidence
    prediction or bone-volume support generation.
    """
    points = _points(points)
    chunk_size = operator.index(chunk_size)
    if chunk_size <= 0:
        raise ValueError('Chunk size must be positive')
    nearest = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), chunk_size):
        delta = points[start:start+chunk_size, None] - skeleton.joints[None]
        nearest[start:start+chunk_size] = np.argmin(np.einsum('nbi,nbi->nb', delta, delta), axis=1)
    parent = skeleton.parents[nearest]
    parent = np.where(parent < 0, nearest, parent)
    return np.concatenate((skeleton.joints[nearest]-points, skeleton.joints[parent]-points), axis=1), nearest


def infer_skeleton_from_field(points, field, *, tolerance, max_joints=512):
    """Cluster predicted joint/parent endpoints into a deterministic joint forest.

    ``points`` are the voxel centers in the same world coordinates used by the
    six-channel field. The first three channels point from each center to its
    nearest joint; the last three point to that joint's parent (or itself for a
    root). Endpoint clustering uses a tolerance-grid followed by neighboring
    cell radius checks. Conflicting parent observations are resolved by mode;
    cycles and oversized rigs are rejected rather than silently repaired.
    """
    points = _points(points)
    vectors = np.asarray(field, dtype=float)
    if (vectors.shape != (len(points), 6) or not np.isfinite(vectors).all()
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('Finite (N,6) fields and a positive finite tolerance are required')
    if type(max_joints) is not int or max_joints <= 0:
        raise ValueError('max_joints must be a positive integer')
    if not len(points):
        raise ValueError('At least one field sample is required')

    endpoints = (points[:, None, :] + vectors.reshape(-1, 2, 3)).reshape(-1, 3)
    scaled = endpoints / tolerance
    limit = np.iinfo(np.int64).max - 2
    if np.any(np.abs(scaled) > limit):
        raise ValueError('Endpoint coordinates exceed clustering range')
    cells = np.floor(scaled).astype(np.int64)
    cell_to_rows = {}
    row_cells = []
    for row, cell in enumerate(cells):
        key = tuple(int(x) for x in cell)
        index = cell_to_rows.get(key)
        if index is None:
            index = len(row_cells)
            cell_to_rows[key] = index
            row_cells.append([])
        row_cells[index].append(row)
    representatives = np.asarray([endpoints[rows].mean(axis=0) for rows in row_cells])

    parents = np.arange(len(row_cells), dtype=np.int64)
    sizes = np.ones(len(row_cells), dtype=np.int64)
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index
    for index, cell in enumerate(row_cells):
        key = tuple(int(x) for x in cells[cell[0]])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    other = cell_to_rows.get((key[0]+dx, key[1]+dy, key[2]+dz))
                    if other is None or other >= index:
                        continue
                    delta = representatives[index] - representatives[other]
                    if float(delta @ delta) <= tolerance*tolerance:
                        left, right = find(index), find(other)
                        if left != right:
                            if sizes[left] < sizes[right]:left, right = right, left
                            parents[right] = left
                            sizes[left] += sizes[right]

    roots = np.asarray([find(i) for i in range(len(row_cells))])
    row_cluster = np.empty(len(endpoints), dtype=np.int64)
    for cell_index, rows in enumerate(row_cells):
        row_cluster[rows] = roots[cell_index]
    unique_roots = np.unique(roots)
    centroids = np.asarray([endpoints[row_cluster == root].mean(axis=0) for root in unique_roots])
    order = np.lexsort((centroids[:, 2], centroids[:, 1], centroids[:, 0]))
    ordered_roots = unique_roots[order]
    if len(ordered_roots) > max_joints:
        raise ValueError(f'Inferred joint count exceeds max_joints={max_joints}')
    root_to_joint = {int(root): rank for rank, root in enumerate(ordered_roots)}
    endpoint_joint = np.asarray([root_to_joint[int(root)] for root in row_cluster], dtype=np.int64).reshape(-1, 2)
    joints = centroids[order]

    votes = [dict() for _ in joints]
    for child, parent in endpoint_joint:
        child, parent = int(child), int(parent)
        votes[child][parent] = votes[child].get(parent, 0) + 1
    inferred_parents = np.full(len(joints), -1, dtype=np.int64)
    for child, counts in enumerate(votes):
        if counts:
            # Highest support wins; ties use the stable coordinate ordering.
            winner = min(counts, key=lambda parent: (-counts[parent], parent))
            if winner != child:
                inferred_parents[child] = winner
    return Skeleton(joints, inferred_parents)


def sample_sparse_embeddings(points, coords, embeddings, *, grid_size, aabb):
    """Trilinearly sample four-channel sparse voxel features at CUDA points.

    Missing sparse corners are omitted and the remaining interpolation weights
    are renormalized. The returned support mask is false when all eight corners
    are absent; callers must handle that case instead of silently inventing a
    feature vector.
    """
    import torch
    if not isinstance(points, torch.Tensor) or not points.is_cuda:
        raise ValueError('points must be a CUDA tensor')
    if (points.ndim != 2 or points.shape[1] != 3 or not points.is_floating_point()
            or not torch.isfinite(points).all()):
        raise ValueError('points must be finite floating (N,3)')
    device=points.device
    coords=torch.as_tensor(coords,device=device)
    if coords.dtype not in (torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64,
                            torch.uint16,torch.uint32,torch.uint64):
        raise ValueError('Sparse coordinates must use an integer dtype')
    coords=coords.to(torch.int64)
    embeddings=torch.as_tensor(embeddings,device=device,dtype=points.dtype)
    bounds=torch.as_tensor(aabb,device=device,dtype=points.dtype)
    grid=torch.as_tensor(grid_size,device=device,dtype=torch.int64)
    if (coords.ndim!=2 or coords.shape[1]!=3 or not len(coords) or embeddings.shape!=(len(coords),4)
            or not torch.isfinite(embeddings).all() or bounds.shape!=(2,3)
            or not torch.isfinite(bounds).all() or not (bounds[1]>bounds[0]).all()
            or grid.shape!=(3,) or (grid<=0).any()
            or (coords<0).any() or (coords>=grid).any()):
        raise ValueError('Invalid sparse coordinates, embeddings, grid, or bounds')
    if len(torch.unique(coords,dim=0))!=len(coords):
        raise ValueError('Sparse coordinates must be unique')
    voxel=(bounds[1]-bounds[0])/grid.to(points.dtype)
    query=((points-bounds[0])/voxel)-.5
    query=torch.minimum(torch.maximum(query,torch.zeros_like(query)),(grid-1).to(points.dtype))
    lower=query.floor().to(torch.int64)
    upper=torch.minimum(lower+1,grid-1)
    fraction=query-lower.to(query.dtype)
    gz,gy=grid[2],grid[1]
    keys=coords[:,0].to(torch.int64)*(gy*gz)+coords[:,1].to(torch.int64)*gz+coords[:,2].to(torch.int64)
    keys,order=torch.sort(keys)
    sorted_embeddings=embeddings[order]
    result=torch.zeros((len(points),4),device=device,dtype=points.dtype)
    support=torch.zeros(len(points),device=device,dtype=points.dtype)
    for ix in (0,1):
        cx=upper[:,0] if ix else lower[:,0]
        wx=fraction[:,0] if ix else 1-fraction[:,0]
        for iy in (0,1):
            cy=upper[:,1] if iy else lower[:,1]
            wy=fraction[:,1] if iy else 1-fraction[:,1]
            for iz in (0,1):
                cz=upper[:,2] if iz else lower[:,2]
                wz=fraction[:,2] if iz else 1-fraction[:,2]
                weight=wx*wy*wz
                query_keys=cx*(gy*gz)+cy*gz+cz
                positions=torch.searchsorted(keys,query_keys.contiguous()).clamp(max=len(keys)-1)
                present=keys[positions]==query_keys
                active_weight=weight*present.to(weight.dtype)
                result+=sorted_embeddings[positions]*active_weight[:,None]
                support+=active_weight
    valid=support>0
    if valid.any():result[valid]/=support[valid,None]
    return result,valid


def _weights(weights, joint_count):
    w = np.asarray(weights, dtype=float)
    if (w.ndim != 2 or w.shape[1] != joint_count or not np.isfinite(w).all()
            or np.any(w < 0) or not np.allclose(w.sum(axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError('Weights must be finite nonnegative (N,J), each row summing to one')
    return w


def skin_embeddings(weights, joint_embeddings):
    """Weighted average of supplied (J,4) embeddings; never substitutes a trained encoder."""
    e = np.asarray(joint_embeddings, dtype=float)
    if e.ndim != 2 or e.shape[1] != 4 or not len(e) or not np.isfinite(e).all():
        raise ValueError('Joint embeddings must be finite (J,4)')
    return _weights(weights, len(e)) @ e


def transfer_skin_weights(points, vertices, faces, vertex_weights, *, method='auto'):
    """Barycentrically transfer source vertex weights at closest surface points.

    Uses TriangleBVH acceleration for exact surface projection.
    """
    vertices = _points(vertices)
    weights = np.asarray(vertex_weights, dtype=float)
    if weights.ndim != 2 or not weights.shape[1] or len(weights) != len(vertices):
        raise ValueError('Source weights must have shape (V,J) with J > 0')
    weights = _weights(weights, weights.shape[1])
    try:
        from .proximity import TriangleBVH
        bvh = TriangleBVH(vertices, faces)
        ids, bary = bvh.query(points)
    except Exception:
        from .materials import closest_surface
        ids, bary = closest_surface(points, vertices, faces)
    triangles = np.asarray(faces)[ids]
    return np.einsum('ni,nij->nj', bary, weights[triangles])


def attach_articulation(asset, skeleton, *, voxel_weights=None, joint_embeddings=None):
    """Sample skeleton field at voxel centers; optionally attach supplied SkinAE features.

    Voxel weights must already be transferred from the source rig. Roots use a
    duplicated displacement. No joints or weights are inferred from geometry.
    """
    centers = asset.aabb[0] + (asset.coords.astype(float) + .5) * asset.voxel_size
    field, _ = skeleton_displacements(centers, skeleton)
    if (voxel_weights is None) != (joint_embeddings is None):
        raise ValueError('Supply both weights and joint embeddings, or neither')
    values = dict(skeleton_field=field)
    if voxel_weights is not None:
        if np.asarray(joint_embeddings).shape != (len(skeleton.joints), 4):
            raise ValueError('Joint embedding count must match skeleton')
        embedding = skin_embeddings(voxel_weights, joint_embeddings)
        if len(embedding) != len(asset.coords):
            raise ValueError('Voxel weight row count must match active voxels')
        values['skin_embedding'] = embedding
    return replace(asset, **values)


def joint_deformation_matrices(bind_world, pose_world):
    """Return per-joint ``pose_world @ inv(bind_world)`` world deformations.

    Both inputs use column-vector affine matrices. This composes bind-relative
    motion without forward kinematics or skeleton inference.
    """
    bind = np.asarray(bind_world, dtype=float)
    pose = np.asarray(pose_world, dtype=float)
    for name, matrices in (('bind', bind), ('pose', pose)):
        if (matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or not len(matrices)
                or not np.isfinite(matrices).all()
                or not np.allclose(matrices[:, 3, :], [0, 0, 0, 1], rtol=0, atol=1e-12)):
            raise ValueError(f'{name.capitalize()} joint matrices must be finite affine (J,4,4)')
    if len(bind) != len(pose):
        raise ValueError('Bind and pose joint counts must match')
    try:
        inverse = np.linalg.inv(bind)
    except np.linalg.LinAlgError as error:
        raise ValueError('Bind joint matrices must be invertible') from error
    result = pose @ inverse
    if not np.isfinite(result).all():
        raise ValueError('Joint deformation produced nonfinite matrices')
    return result


def linear_blend_skinning(vertices, weights, joint_matrices):
    """Apply caller-supplied world_pose @ inverse_bind matrices to source vertices.

    Matrices use column-vector convention. This is a deformation check utility;
    forward kinematics and inverse bind computation remain the caller's task.
    """
    v = _points(vertices)
    matrices = np.asarray(joint_matrices, dtype=float)
    if (matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or not len(matrices)
            or not np.isfinite(matrices).all()
            or not np.allclose(matrices[:, 3, :], [0, 0, 0, 1], rtol=0, atol=1e-12)):
        raise ValueError('Joint matrices must be finite affine (J,4,4)')
    w = _weights(weights, len(matrices))
    if len(w) != len(v):
        raise ValueError('Weight row count must match vertices')
    homogeneous = np.column_stack((v, np.ones(len(v))))
    blended = np.einsum('nj,jab->nab', w, matrices)
    return np.einsum('nab,nb->na', blended, homogeneous)[:, :3]


def skeleton_confidence(points, skeleton, *, chunk_size=4096):
    """Nearest/second-nearest squared-distance confidence (AniGen equation 3).

    Returns a separate (N,) target without changing six-channel displacement
    storage. A singleton skeleton has confidence one; coincident nearest joints
    give zero confidence, including when the query coincides with both joints.
    Temporary distance storage is bounded by chunk_size times joint count.
    """
    points = _points(points)
    chunk_size = operator.index(chunk_size)
    if chunk_size <= 0:
        raise ValueError('Chunk size must be positive')
    confidence = np.ones(len(points), dtype=float)
    if len(skeleton.joints) == 1:
        return confidence
    for start in range(0, len(points), chunk_size):
        delta = points[start:start+chunk_size, None] - skeleton.joints[None]
        squared = np.einsum('nbi,nbi->nb', delta, delta)
        pair = np.partition(squared, 1, axis=1)[:, :2]
        first, second = pair.min(axis=1), pair.max(axis=1)
        ratio = np.ones(len(first), dtype=float)
        np.divide(first, second, out=ratio, where=second > 0)
        confidence[start:start+chunk_size] = np.clip(1-ratio, 0, 1)
    return confidence
