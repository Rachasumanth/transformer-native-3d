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
