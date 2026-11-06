"""Bounded Hermite QEF reference solver for one local voxel at a time.

The local cube is [0,1]^3. Minimize weighted squared unit-normal plane
residuals plus regularization_weight * ||x - mass_point||^2. Optional boundary
planes use the same representation. The mass point is the mean face intersection.
Weights multiply squared residuals (not the plane equations themselves).
"""
from itertools import product
import numpy as np


def _hermite(points, normals, name, *, allow_empty=False):
    p = np.asarray(points, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] != 3 or n.shape != p.shape
            or not np.isfinite(p).all() or not np.isfinite(n).all()
            or (not allow_empty and not len(p))):
        raise ValueError(f'{name} requires finite matching (N, 3) points and normals')
    # Scaling first avoids overflow/underflow while normalizing finite vectors.
    scale = np.max(np.abs(n), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError(f'{name} normals must be nonzero')
    unit = n / scale
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError(f'{name} intersections must be in the local unit voxel')
    return p, unit


def solve_qef(points, normals, *, boundary_points=None, boundary_normals=None,
              face_weight=1.0, boundary_weight=1.0, regularization_weight=0.1):
    """Return a bounded local dual vertex, including rank-deficient cases.

    Enumerates 27 active sets (interior, faces, edges, corners) and solves each
    free least-squares problem. This finds the box-constrained minimum rather
    than clipping an unconstrained solution. Equal objective values prefer the
    point closest to the face mass point. No SDF or inside/outside test is used.
    This consumes Hermite data; it does not find mesh/grid intersections.
    """
    p, n = _hermite(points, normals, 'Face')
    weights = np.asarray([face_weight, boundary_weight, regularization_weight], dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or face_weight <= 0:
        raise ValueError('Weights must be finite and nonnegative; face weight positive')
    if (boundary_points is None) != (boundary_normals is None):
        raise ValueError('Boundary points and normals must be supplied together')
    mass = p.mean(axis=0)
    matrices = [np.sqrt(face_weight) * n]
    rhs = [np.sqrt(face_weight) * np.einsum('ij,ij->i', n, p)]
    if boundary_points is not None:
        bp, bn = _hermite(boundary_points, boundary_normals, 'Boundary', allow_empty=True)
        matrices.append(np.sqrt(boundary_weight) * bn)
        rhs.append(np.sqrt(boundary_weight) * np.einsum('ij,ij->i', bn, bp))
    matrices.append(np.sqrt(regularization_weight) * np.eye(3))
    rhs.append(np.sqrt(regularization_weight) * mass)
    a, b = np.concatenate(matrices), np.concatenate(rhs)
    best = None
    for state in product((-1, 0, 1), repeat=3):
        free = np.array(state) == -1
        x = np.maximum(state, 0).astype(float)
        if np.any(free):
            # Solve relative to mass so null-space coordinates remain centered.
            x[free] = mass[free]
            delta = np.linalg.lstsq(a[:, free], b - a @ x, rcond=None)[0]
            x[free] += delta
        if np.any(x < -1e-12) or np.any(x > 1 + 1e-12):
            continue
        x = np.clip(x, 0, 1)
        residual = a @ x - b
        score = (float(residual @ residual), float(np.sum((x - mass)**2)))
        if best is None or score < best[0]:
            best = score, x
    return best[1]
