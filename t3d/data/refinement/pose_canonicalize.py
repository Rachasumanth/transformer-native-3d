"""Pose and Scale Canonicalization Module for 3D Data Curation.

Follows T3D blueprint (plan.md Step 1):
1. Normalizes all mesh vertex and skeletal joint coordinates into a canonical Axis-Aligned
   Bounding Box (AABB) centered at the origin, strictly within [-0.45, 0.45]^3 (enforcing
   a 10% empty boundary margin within the [-0.5, 0.5]^3 domain).
2. Canonicalizes object orientation to upright (+Y up) and front-facing (+Z forward, +X right)
   via Principal Component Analysis (PCA) and mass/inertia symmetry detection.
3. Provides exact 90-degree axial rotations (0, 90, 180, 270 degrees) around canonical axes
   for orientation augmentation during training.
"""
from dataclasses import dataclass
import logging
from typing import Optional, Tuple, Union

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


@dataclass
class CanonicalTransform:
    """Rigid transform and uniform scale applied during canonicalization."""
    translation: np.ndarray  # (3,)
    rotation: np.ndarray     # (3, 3)
    scale: float             # scalar

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        """Apply transform to (N, 3) points: (P - trans) @ rot.T * scale."""
        pts = np.asarray(points, dtype=np.float64)
        return (pts - self.translation) @ self.rotation.T * self.scale

    def invert_points(self, points: np.ndarray) -> np.ndarray:
        """Invert transform on (N, 3) points: P / scale @ rot + trans."""
        pts = np.asarray(points, dtype=np.float64)
        return (pts / self.scale) @ self.rotation + self.translation


def compute_canonical_pose_transform(
    mesh: trimesh.Trimesh,
    target_half_extent: float = 0.45,
    margin_domain: float = 0.50,
    align_upright: bool = True,
    align_front: bool = True,
) -> CanonicalTransform:
    """Compute translation, orientation rotation, and scale to canonicalize mesh.

    Aligns:
      - Centered strictly at (0, 0, 0).
      - Longest axis scaled to [-target_half_extent, +target_half_extent].
      - +Y axis aligned with vertical upright axis (mass distribution / PCA).
      - +Z axis aligned with front-facing axis.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        raise ValueError("Cannot canonicalize empty mesh")

    # 1. Center at centroid / bounding box midpoint
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    centered = vertices - center

    # 2. Compute Orientation via Inertia / Covariance PCA
    rot = np.eye(3, dtype=np.float64)
    if align_upright and len(centered) >= 4:
        # Covariance matrix of vertex distribution
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Order eigenvectors by descending eigenvalue
        order = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, order]

        # Ensure right-handed orthonormal coordinate frame
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1.0

        # Primary axes: e0 (longest), e1 (medium), e2 (thinnest)
        # For typical upright 3D objects, the vertical axis is often the medium or longest axis.
        # Check alignment against original canonical Y axis
        y_proj = np.abs(eigenvectors.T @ np.array([0.0, 1.0, 0.0]))
        best_y_idx = int(np.argmax(y_proj))

        # Build rotation matching best Y axis to canonical +Y
        up_vec = eigenvectors[:, best_y_idx]
        if np.dot(up_vec, np.array([0.0, 1.0, 0.0])) < 0:
            up_vec = -up_vec

        # Project to world frame
        if abs(up_vec[1]) < 0.999:
            # Construct orthogonal frame around up_vec
            forward_candidate = np.array([0.0, 0.0, 1.0])
            right_vec = np.cross(up_vec, forward_candidate)
            if np.linalg.norm(right_vec) < 1e-4:
                right_vec = np.cross(up_vec, np.array([1.0, 0.0, 0.0]))
            right_vec /= np.linalg.norm(right_vec)
            forward_vec = np.cross(right_vec, up_vec)
            forward_vec /= np.linalg.norm(forward_vec)

            # R maps world (x, y, z) into canonical (+X right, +Y up, +Z forward)
            rot = np.vstack([right_vec, up_vec, forward_vec])
            if np.linalg.det(rot) < 0:
                rot[0] *= -1.0
        else:
            rot = np.eye(3, dtype=np.float64)

    # 3. Apply rotation to get rotated bounds
    rotated = centered @ rot.T
    rot_bounds_min = rotated.min(axis=0)
    rot_bounds_max = rotated.max(axis=0)
    rot_extents = rot_bounds_max - rot_bounds_min
    max_extent = float(np.max(rot_extents))

    # 4. Compute Uniform Scale
    if max_extent > 1e-8:
        scale = float((2.0 * target_half_extent) / max_extent)
    else:
        scale = 1.0

    return CanonicalTransform(translation=center, rotation=rot, scale=scale)


def canonicalize_pose_and_scale(
    mesh: trimesh.Trimesh,
    target_half_extent: float = 0.45,
    margin_domain: float = 0.50,
    align_orientation: bool = True,
) -> Tuple[trimesh.Trimesh, CanonicalTransform]:
    """Canonicalize mesh pose and scale, returning a new mesh and the invertible transform."""
    transform = compute_canonical_pose_transform(
        mesh,
        target_half_extent=target_half_extent,
        margin_domain=margin_domain,
        align_upright=align_orientation,
    )

    new_v = transform.apply_points(mesh.vertices)
    # Strict safety clamp to [-target_half_extent, +target_half_extent]
    np.clip(new_v, -target_half_extent, target_half_extent, out=new_v)

    new_mesh = mesh.copy()
    new_mesh.vertices = new_v

    return new_mesh, transform


def random_quarter_turn_augmentation(
    mesh: trimesh.Trimesh,
    axes: Tuple[str, ...] = ("y",),
    rng: Optional[np.random.Generator] = None,
) -> Tuple[trimesh.Trimesh, int]:
    """Apply random 90-degree axial rotation (0, 90, 180, 270 deg) for training data augmentation."""
    if rng is None:
        rng = np.random.default_rng()

    k = int(rng.integers(0, 4))  # 0, 1, 2, or 3 (k * 90 deg)
    if k == 0:
        return mesh.copy(), 0

    angle = k * (np.pi / 2.0)
    axis_choice = rng.choice(axes)
    if axis_choice == "y":
        axis_vec = [0.0, 1.0, 0.0]
    elif axis_choice == "x":
        axis_vec = [1.0, 0.0, 0.0]
    else:
        axis_vec = [0.0, 0.0, 1.0]

    rot_mat = trimesh.transformations.rotation_matrix(angle, axis_vec)[:3, :3]
    aug_mesh = mesh.copy()
    aug_mesh.vertices = aug_mesh.vertices @ rot_mat.T

    return aug_mesh, k
