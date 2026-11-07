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


def infer_skeleton_from_field(points, field, *, tolerance, max_joints=512, min_votes=1):
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
    if type(min_votes) is not int or min_votes <= 0:
        raise ValueError('min_votes must be a positive integer')
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
    if min_votes > 1:
        root_counts = {int(r): int(np.sum(row_cluster == r)) for r in unique_roots}
        supported_roots = [r for r in unique_roots if root_counts[r] >= min_votes]
        if supported_roots and len(supported_roots) < len(unique_roots):
            sup_centroids = np.asarray([endpoints[row_cluster == r].mean(axis=0) for r in supported_roots])
            unsupported_mask = np.isin(row_cluster, supported_roots, invert=True)
            if np.any(unsupported_mask):
                dists = np.linalg.norm(endpoints[unsupported_mask, None] - sup_centroids[None, :], axis=-1)
                nearest_sup_idx = np.argmin(dists, axis=-1)
                row_cluster[unsupported_mask] = np.asarray(supported_roots)[nearest_sup_idx]
            unique_roots = np.asarray(supported_roots)
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


def check_bone_containment(skeleton, aabb=None, steps=10):
    if aabb is None:
        aabb = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    else:
        aabb = np.asarray(aabb, dtype=float)
    joints = np.asarray(skeleton.joints, dtype=float)
    parents = np.asarray(skeleton.parents, dtype=int)
    violations = 0
    total_segments = 0
    for child_idx, parent_idx in enumerate(parents):
        if parent_idx < 0:
            continue
        total_segments += 1
        c = joints[child_idx]
        p = joints[parent_idx]
        t = np.linspace(0, 1, steps)[:, None]
        pts = (1 - t) * c + t * p
        outside = (pts < aabb[0] - 1e-4) | (pts > aabb[1] + 1e-4)
        if np.any(outside):
            violations += 1
    return {
        'total_segments': total_segments,
        'violations': violations,
        'containment_rate': (total_segments - violations) / max(1, total_segments)
    }


def robust_infer_skeleton(centers, pred_field, voxel_size, aabb=None):
    """Multi-scale clustering recovery for neural predicted displacement fields."""
    base_tol = max(float(np.min(voxel_size)), 1e-4)
    candidates = []
    if len(centers) > 2048:
        eval_idx = np.linspace(0, len(centers) - 1, 2048, dtype=int)
        c_eval = centers[eval_idx]
        pf_eval = pred_field[eval_idx]
    else:
        c_eval = centers
        pf_eval = pred_field

    for tol_mult in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        for min_v in [2, 3, 4, 5, 6, 8, 10, 12]:
            try:
                eff_min_v = max(1, int(round(min_v * len(c_eval) / len(centers)))) if len(centers) > 2048 else min_v
                skel = infer_skeleton_from_field(
                    c_eval, pf_eval,
                    tolerance=base_tol * tol_mult,
                    min_votes=eff_min_v,
                    max_joints=128
                )
                if 2 <= len(skel.joints) <= 64:
                    if aabb is not None:
                        clamped_joints = np.clip(skel.joints, aabb[0], aabb[1])
                        skel = Skeleton(clamped_joints, skel.parents)
                    viol = 0
                    if aabb is not None:
                        c_info = check_bone_containment(skel, aabb)
                        viol = c_info['violations']
                    implied_field, _ = skeleton_displacements(c_eval, skel)
                    diff = pf_eval - implied_field
                    res = float(np.mean(np.linalg.norm(diff[:, :3], axis=-1) + np.linalg.norm(diff[:, 3:6], axis=-1)))
                    score = viol * 10.0 + res
                    candidates.append((score, skel))
            except Exception:
                continue

    if candidates:
        candidates.sort(key=lambda x: x[0])
        best = candidates[0][1]
        return refine_joints_medial_axis(best, centers, voxel_size)

    for tol_mult in [3.5, 4.0, 5.0]:
        for min_v in [2, 4]:
            try:
                eff_min_v = max(1, int(round(min_v * len(c_eval) / len(centers)))) if len(centers) > 2048 else min_v
                skel = infer_skeleton_from_field(
                    c_eval, pf_eval,
                    tolerance=base_tol * tol_mult,
                    min_votes=eff_min_v,
                    max_joints=128
                )
                if 1 <= len(skel.joints) <= 64:
                    if aabb is not None:
                        clamped_joints = np.clip(skel.joints, aabb[0], aabb[1])
                        skel = Skeleton(clamped_joints, skel.parents)
                    return refine_joints_medial_axis(skel, centers, voxel_size)
            except Exception:
                continue
    return None


def sample_sparse_embeddings(points, coords, embeddings, *, grid_size, aabb):
    """Trilinearly sample four-channel sparse voxel features at CUDA points.

    Missing sparse corners are omitted and the remaining interpolation weights
    are renormalized. The returned support mask is false when all eight corners
    are absent; callers must handle that case instead of silently inventing a
    feature vector.
    """
    import torch
    if not isinstance(points, torch.Tensor):
        raise ValueError('points must be a torch tensor')
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


def transfer_skin_weights(points, vertices, faces, vertex_weights):
    """Barycentrically transfer source vertex weights at closest surface points.

    Uses the CPU triangle projection reference; production BVH transfer is pending.
    """
    from .materials import closest_surface
    vertices = _points(vertices)
    weights = np.asarray(vertex_weights, dtype=float)
    if weights.ndim != 2 or not weights.shape[1] or len(weights) != len(vertices):
        raise ValueError('Source weights must have shape (V,J) with J > 0')
    weights = _weights(weights, weights.shape[1])
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


def refine_joints_medial_axis(skeleton, centers, voxel_size):
    """Snap each joint 30% toward the local centroid of its enclosing voxel cross-section.

    Identifies the local cross-section of voxels enclosing each joint perpendicular
    to the incident bone direction, isolates the connected component enclosing the joint,
    and shifts the joint 30% toward the cross-section centroid. Uses cKDTree for O(log N)
    neighborhood clustering.
    """
    if not isinstance(skeleton, Skeleton):
        raise ValueError('A validated Skeleton is required')
    centers_arr = _points(centers)
    if not len(centers_arr):
        return skeleton

    if hasattr(voxel_size, '__len__'):
        v_arr = np.asarray(voxel_size, dtype=float)
        if v_arr.size == 0 or np.any(v_arr <= 0) or not np.isfinite(v_arr).all():
            raise ValueError('voxel_size must be positive finite')
        v_scale = float(np.mean(v_arr))
    else:
        v_scale = float(voxel_size)
        if not np.isfinite(v_scale) or v_scale <= 0:
            raise ValueError('voxel_size must be positive finite')

    from scipy.spatial import cKDTree

    joints = np.array(skeleton.joints, dtype=float, copy=True)
    parents = np.asarray(skeleton.parents, dtype=int)
    refined_joints = np.empty_like(joints)

    for j in range(len(joints)):
        p = joints[j]
        # Determine bone direction vector
        if parents[j] >= 0:
            v_dir = p - joints[parents[j]]
        else:
            children = np.where(parents == j)[0]
            v_dir = joints[children[0]] - p if len(children) > 0 else np.zeros(3)

        v_norm = float(np.linalg.norm(v_dir))
        u = v_dir / v_norm if v_norm > 1e-6 else None
        dist = np.linalg.norm(centers_arr - p, axis=-1)

        if u is not None:
            t = (centers_arr - p) @ u
            slice_thickness = max(0.6 * v_scale, 1e-4)
            slice_mask = np.abs(t) <= slice_thickness
            if not np.any(slice_mask):
                slice_thickness = max(1.5 * v_scale, 1e-4)
                slice_mask = np.abs(t) <= slice_thickness

            if np.any(slice_mask):
                slice_centers = centers_arr[slice_mask]
                slice_dists = np.linalg.norm(slice_centers - p, axis=-1)
                start_idx = int(np.argmin(slice_dists))

                # Extract connected component enclosing the joint in the slice using cKDTree
                adj_thresh = 1.75 * v_scale
                slice_tree = cKDTree(slice_centers)
                visited = np.zeros(len(slice_centers), dtype=bool)
                queue = [start_idx]
                visited[start_idx] = True
                while queue:
                    curr = queue.pop()
                    nbrs = slice_tree.query_ball_point(slice_centers[curr], r=adj_thresh)
                    for n in nbrs:
                        if not visited[n]:
                            visited[n] = True
                            queue.append(n)

                comp = slice_centers[visited]
                # Project component points onto cross-section plane through p normal to u
                proj = comp - ((comp - p) @ u)[:, None] * u
                centroid = proj.mean(axis=0)
            else:
                k = min(8, len(centers_arr))
                near_idx = np.argpartition(dist, k - 1)[:k]
                centroid = centers_arr[near_idx].mean(axis=0)
        else:
            adj_thresh = 1.75 * v_scale
            near_idx = int(np.argmin(dist))
            tree_full = cKDTree(centers_arr)
            visited = np.zeros(len(centers_arr), dtype=bool)
            queue = [near_idx]
            visited[near_idx] = True
            while queue and np.sum(visited) < 128:
                curr = queue.pop()
                nbrs = tree_full.query_ball_point(centers_arr[curr], r=adj_thresh)
                for n in nbrs:
                    if not visited[n]:
                        visited[n] = True
                        queue.append(n)
            centroid = centers_arr[visited].mean(axis=0)

        # Snap 30% toward the local centroid of its enclosing voxel cross-section
        refined_joints[j] = p + 0.3 * (centroid - p)

    return Skeleton(refined_joints, skeleton.parents)


def _point_to_segment_distance(points, a, b, vertex_normals=None, chunk_size=4096):
    """Vectorized distance from (N,3) points to J line segments [a, b], with normal-visibility penalty."""
    ab = b - a
    ab_len_sq = np.sum(ab * ab, axis=-1)
    dists = np.empty((len(points), len(a)), dtype=np.float32)
    for start in range(0, len(points), chunk_size):
        p_chunk = points[start:start+chunk_size]
        ap = p_chunk[:, None, :] - a[None, :, :]
        dot = np.sum(ap * ab[None, :, :], axis=-1)
        denom = np.where(ab_len_sq > 1e-12, ab_len_sq, 1.0)
        t = np.clip(dot / denom[None, :], 0.0, 1.0)
        t = np.where(ab_len_sq[None, :] > 1e-12, t, 0.0)
        proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        diff = p_chunk[:, None, :] - proj
        d = np.linalg.norm(diff, axis=-1)
        if vertex_normals is not None:
            n_chunk = vertex_normals[start:start+chunk_size]
            d_safe = np.maximum(d, 1e-8)
            v_dir = diff / d_safe[:, :, None]
            # Normal-facing visibility penalty: if n_v . v_dir < -0.2, the bone faces
            # an exterior surface from across an air gap; apply a 3.0x distance penalty.
            dot_nv = np.sum(n_chunk[:, None, :] * v_dir, axis=-1)
            d = np.where(dot_nv < -0.2, d * 3.0, d)
        dists[start:start+chunk_size] = d.astype(np.float32)
    return dists


def compute_geometric_skin_weights(vertices, faces, skeleton, max_influences=4):
    """Calculate smooth geodesic/volumetric bone-segment skin weights conforming to glTF 2.0.

    Computes bone segment distances to each bone with normal-facing visibility penalties,
    smooths along the mesh topology via 3-step Laplacian diffusion on the vertex adjacency
    matrix to prevent air-gap bleeding, sorts descending, clamps to top max_influences (<= 4),
    zeroes weights below 0.005, and normalizes in float32 such that row sums equal exactly 1.0
    with w0 = 1.0 - sum(w1..w3) guaranteeing zero IEEE-754 drift.
    """
    if not isinstance(skeleton, Skeleton):
        raise ValueError('A validated Skeleton is required')
    v = _points(vertices)
    if type(max_influences) is not int or not 1 <= max_influences <= 16:
        raise ValueError('max_influences must be an integer between 1 and 16')
    J = len(skeleton.joints)
    if len(v) == 0:
        return np.empty((0, J), dtype=np.float32)
    if J == 1:
        return np.ones((len(v), 1), dtype=np.float32)

    # Compute vertex normals and mesh adjacency if faces are supplied
    v_normals = None
    adj_norm = None
    if faces is not None and len(faces) > 0:
        f_arr = np.asarray(faces, dtype=np.int64)
        if f_arr.ndim == 2 and f_arr.shape[1] == 3 and (f_arr >= 0).all() and (f_arr < len(v)).all():
            triangles = v[f_arr]
            face_norms = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            v_normals = np.zeros_like(v, dtype=np.float32)
            for corner in range(3):
                np.add.at(v_normals, f_arr[:, corner], face_norms)
            norm_lens = np.linalg.norm(v_normals, axis=1, keepdims=True)
            norm_lens[norm_lens == 0] = 1.0
            v_normals = (v_normals / norm_lens).astype(np.float32)

            # Build row-normalized mesh adjacency matrix
            import scipy.sparse as sp
            rows = np.concatenate([f_arr[:, 0], f_arr[:, 1], f_arr[:, 1], f_arr[:, 2], f_arr[:, 2], f_arr[:, 0]])
            cols = np.concatenate([f_arr[:, 1], f_arr[:, 0], f_arr[:, 2], f_arr[:, 1], f_arr[:, 0], f_arr[:, 2]])
            adj = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(v), len(v)))
            deg = np.array(adj.sum(axis=1)).flatten()
            deg[deg == 0] = 1.0
            d_inv = sp.diags(1.0 / deg)
            adj_norm = d_inv @ adj

    joints = skeleton.joints
    parents = skeleton.parents
    a_pts = np.empty_like(joints)
    b_pts = np.empty_like(joints)

    for j in range(J):
        b_pts[j] = joints[j]
        if parents[j] >= 0:
            a_pts[j] = joints[parents[j]]
        else:
            a_pts[j] = joints[j]

    dists = _point_to_segment_distance(v, a_pts, b_pts, vertex_normals=v_normals)

    lengths = np.linalg.norm(b_pts - a_pts, axis=-1)
    pos_lens = lengths[lengths > 1e-6]
    if len(pos_lens) > 0:
        sigma = float(np.mean(pos_lens))
    else:
        span = np.ptp(v, axis=0)
        sigma = float(np.linalg.norm(span)) / 4.0 if np.any(span > 0) else 1.0
    sigma = max(sigma, 1e-4)

    # Gaussian / inverse-distance falloff
    inv_sq = 1.0 / (dists + 1e-4) ** 2
    gauss = np.exp(-0.5 * (dists / sigma) ** 2)
    raw_weights = inv_sq * gauss

    # 3-step topological smoothing along connected mesh edges only (eliminates air gap bleeding)
    if adj_norm is not None:
        for _ in range(3):
            raw_weights = 0.5 * raw_weights + 0.5 * (adj_norm @ raw_weights)

    max_inf = min(max_influences, J)
    result = np.zeros((len(v), J), dtype=np.float32)

    for i in range(len(v)):
        row = raw_weights[i]
        row_sum = float(np.sum(row))
        if row_sum > 0:
            row = row / row_sum
        else:
            row = np.full(J, 1.0 / J, dtype=np.float32)

        order = np.argsort(-row, kind='stable')
        top_idx = order[:max_inf]
        top_w = row[top_idx].astype(np.float32)

        mask = top_w >= 0.005
        if not np.any(mask):
            mask[0] = True
        top_w = np.where(mask, top_w, 0.0).astype(np.float32)

        s = float(np.sum(top_w))
        if s > 0:
            top_w = (top_w / s).astype(np.float32)
        else:
            top_w[0] = 1.0

        if len(top_w) > 1:
            rest = np.sum(top_w[1:], dtype=np.float32)
            top_w[0] = np.float32(1.0) - rest
        else:
            top_w[0] = np.float32(1.0)

        result[i, top_idx] = top_w

    return result
