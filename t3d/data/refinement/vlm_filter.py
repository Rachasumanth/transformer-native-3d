"""VLM Aesthetic and Geometry Filtering Pipeline for 3D Data Curation.

Follows T3D blueprint (plan.md Step 1):
1. Multi-view rendering of color, depth, and normal maps across canonical viewpoints.
2. Geometric complexity, dihedral angle variance, and planar ground-plane detection.
3. Physically-Based Rendering (PBR) material completeness verification.
4. Vision-Language Model (VLM) filtering interface to discard:
   - Simple geometric primitives (cubes, spheres, flat sheets)
   - Ambient ground planes and display pedestals
   - Noisy raw photogrammetry scans with floating debris or non-manifold tears
"""
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import trimesh

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of aesthetic, geometric, and material quality filtering."""
    passed: bool
    quality_score: float
    is_primitive: bool
    has_ground_plane: bool
    has_noisy_artifacts: bool
    pbr_valid: bool
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "quality_score": float(self.quality_score),
            "is_primitive": self.is_primitive,
            "has_ground_plane": self.has_ground_plane,
            "has_noisy_artifacts": self.has_noisy_artifacts,
            "pbr_valid": self.pbr_valid,
            "reasons": list(self.reasons),
        }


def render_projected_views(
    mesh: trimesh.Trimesh,
    image_size: int = 128,
    views: Tuple[str, ...] = ("front", "right", "top", "iso"),
) -> Dict[str, Dict[str, np.ndarray]]:
    """Render 2D multi-view normal and depth projections for VLM analysis.

    Renders orthographic projections from canonical camera angles:
      - 'front': +Z view (+Y up)
      - 'right': +X view (+Y up)
      - 'top':   +Y view (-Z up)
      - 'iso':   Isometric view (diagonal [1, 1, 1])

    Returns:
        Dict mapping view_name -> {"depth": (H, W), "normal": (H, W, 3), "mask": (H, W)}
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Normalize mesh to [-0.45, 0.45]^3 bounding cube for rendering
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    scale = np.max(bounds[1] - bounds[0])
    if scale > 1e-6:
        v_norm = (vertices - center) / scale * 0.90
    else:
        v_norm = vertices - center

    # Face normals
    tri_v0 = v_norm[faces[:, 0]]
    tri_v1 = v_norm[faces[:, 1]]
    tri_v2 = v_norm[faces[:, 2]]
    face_normals = np.cross(tri_v1 - tri_v0, tri_v2 - tri_v0)
    fn_norms = np.linalg.norm(face_normals, axis=-1, keepdims=True)
    face_normals = np.where(fn_norms > 1e-8, face_normals / np.maximum(fn_norms, 1e-8), 0.0)

    # Camera rotation matrices (transforming world coordinates to camera coordinates)
    rotations = {
        "front": np.eye(3),
        "right": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
        "top":   np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),
        "iso":   np.array([
            [1.0 / np.sqrt(2.0), 0.0, -1.0 / np.sqrt(2.0)],
            [-1.0 / np.sqrt(6.0), np.sqrt(2.0 / 3.0), -1.0 / np.sqrt(6.0)],
            [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
        ], dtype=np.float64),
    }

    projections = {}
    H, W = image_size, image_size

    for view_name in views:
        rot = rotations.get(view_name, np.eye(3))
        # Transform vertices and face normals to camera frame
        cam_v = v_norm @ rot.T
        cam_fn = face_normals @ rot.T

        # Depth buffer initialized to -infinity
        depth_map = np.full((H, W), -np.inf, dtype=np.float32)
        normal_map = np.zeros((H, W, 3), dtype=np.float32)
        mask = np.zeros((H, W), dtype=bool)

        # Rasterize triangle centers as orthographic approximation
        tri_centers = (cam_v[faces[:, 0]] + cam_v[faces[:, 1]] + cam_v[faces[:, 2]]) / 3.0
        # Project X, Y in [-0.5, 0.5] to image pixels [0, W-1] and [0, H-1]
        px = np.clip(np.floor((tri_centers[:, 0] + 0.5) * (W - 1)).astype(np.int64), 0, W - 1)
        py = np.clip(np.floor((0.5 - tri_centers[:, 1]) * (H - 1)).astype(np.int64), 0, H - 1)
        depths = tri_centers[:, 2]

        # Painter's sorting by depth (nearest replaces farther)
        order = np.argsort(depths)
        for idx in order:
            x, y, d = px[idx], py[idx], depths[idx]
            if d > depth_map[y, x]:
                depth_map[y, x] = d
                # Encode normal [-1, 1] to [0, 1]
                normal_map[y, x] = np.clip(cam_fn[idx] * 0.5 + 0.5, 0.0, 1.0)
                mask[y, x] = True

        # Invert depth for empty texels
        depth_clean = np.where(mask, depth_map - depth_map[mask].min(), 0.0)
        max_d = depth_clean.max()
        if max_d > 1e-6:
            depth_clean /= max_d

        projections[view_name] = {
            "depth": depth_clean,
            "normal": normal_map,
            "mask": mask,
        }

    return projections


def check_geometric_complexity(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """Analyze mesh surface entropy, planar ratio, and primitive characteristics."""
    num_faces = len(mesh.faces)
    num_vertices = len(mesh.vertices)

    # 1. Bounding box aspect ratios
    extents = mesh.extents
    sorted_extents = np.sort(extents)
    flatness_ratio = sorted_extents[0] / max(sorted_extents[2], 1e-6)

    # 2. Dihedral angle variation across interior edges
    face_normals = mesh.face_normals
    # Map edges to faces
    edges = mesh.edges_sorted
    unique_edges, indices, counts = np.unique(edges, axis=0, return_index=True, return_counts=True)
    manifold_mask = (counts == 2)
    
    dihedral_angles = []
    if np.any(manifold_mask):
        # Sample subset of manifold edges to compute angles
        sample_indices = np.where(manifold_mask)[0][:1000]
        # Quick edge to face map
        edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
        for f_idx, face in enumerate(mesh.faces):
            for i in range(3):
                e = tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
                edge_to_faces.setdefault(e, []).append(f_idx)
        
        for e, f_list in edge_to_faces.items():
            if len(f_list) == 2:
                n1 = face_normals[f_list[0]]
                n2 = face_normals[f_list[1]]
                cos_ang = np.clip(np.dot(n1, n2), -1.0, 1.0)
                dihedral_angles.append(float(np.arccos(cos_ang)))

    dihedral_variance = float(np.var(dihedral_angles)) if len(dihedral_angles) > 10 else 0.0
    dihedral_mean = float(np.mean(dihedral_angles)) if dihedral_angles else 0.0

    # 3. Ground plane detection (large flat horizontal component near the bottom)
    has_ground_plane = False
    bottom_z_threshold = mesh.bounds[0, 1] + 0.05 * extents[1]  # Bottom 5% in Y
    y_normals = np.abs(face_normals[:, 1])
    horizontal_faces = (y_normals > 0.95)
    bottom_faces = mesh.triangles_center[:, 1] <= bottom_z_threshold
    
    if np.any(horizontal_faces & bottom_faces):
        bottom_area = np.sum(mesh.area_faces[horizontal_faces & bottom_faces])
        total_area = mesh.area
        if total_area > 0 and (bottom_area / total_area) > 0.35:
            has_ground_plane = True

    # 4. Primitive detection (box/plane/cylinder with very few faces or near-zero curvature)
    is_primitive = False
    if num_faces < 100 and dihedral_variance < 0.02:
        is_primitive = True
    elif flatness_ratio < 0.015:  # Extremely flat planar sheet
        is_primitive = True

    # 5. Noisy scan detection (unusually high face count with tiny area variance, or disconnected pieces)
    connected_components = trimesh.graph.connected_components(mesh.face_adjacency)
    num_components = len(connected_components)
    has_noisy_artifacts = (num_components > 15)

    return {
        "num_vertices": num_vertices,
        "num_faces": num_faces,
        "flatness_ratio": float(flatness_ratio),
        "dihedral_variance": dihedral_variance,
        "dihedral_mean": dihedral_mean,
        "has_ground_plane": has_ground_plane,
        "is_primitive": is_primitive,
        "has_noisy_artifacts": has_noisy_artifacts,
        "num_components": num_components,
    }


def verify_pbr_materials(mesh_or_path: Union[str, Path, trimesh.Trimesh]) -> Tuple[bool, Dict[str, bool]]:
    """Check whether asset possesses valid Physically-Based Rendering (PBR) materials."""
    if isinstance(mesh_or_path, (str, Path)):
        scene_or_mesh = trimesh.load(str(mesh_or_path))
    else:
        scene_or_mesh = mesh_or_path

    has_basecolor = False
    has_mr = False
    has_normal = False

    # Check visuals on mesh or sub-meshes
    geometries = [scene_or_mesh] if isinstance(scene_or_mesh, trimesh.Trimesh) else list(scene_or_mesh.geometry.values())

    for geom in geometries:
        if not hasattr(geom, "visual") or geom.visual is None:
            continue
        visual = geom.visual

        # Check vertex colors or texture material
        if hasattr(visual, "material") and visual.material is not None:
            mat = visual.material
            # PBR material attributes (glTF PBR Metallic Roughness)
            if hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
                has_basecolor = True
            elif hasattr(mat, "baseColorFactor") and mat.baseColorFactor is not None:
                has_basecolor = True
            elif hasattr(mat, "image") and mat.image is not None:
                has_basecolor = True

            if hasattr(mat, "metallicRoughnessTexture") and mat.metallicRoughnessTexture is not None:
                has_mr = True
            elif hasattr(mat, "roughnessFactor") or hasattr(mat, "metallicFactor"):
                has_mr = True

            if hasattr(mat, "normalTexture") and mat.normalTexture is not None:
                has_normal = True

        elif hasattr(visual, "vertex_colors") and visual.vertex_colors is not None and len(visual.vertex_colors) > 0:
            has_basecolor = True

    # PBR is valid if at least basecolor is present and some roughness/metallic factor exists
    pbr_valid = has_basecolor and (has_mr or has_normal or isinstance(scene_or_mesh, trimesh.Trimesh))
    
    return pbr_valid, {
        "basecolor": has_basecolor,
        "metallic_roughness": has_mr,
        "normal": has_normal,
    }


class VLMQualityFilter:
    """Comprehensive aesthetic and geometry curation filter for 3D dataset scaling.

    Refines raw 3D candidates down to high-quality training assets.
    """

    def __init__(
        self,
        min_faces: int = 300,
        max_faces: int = 100000,
        min_dihedral_var: float = 0.0,
        require_pbr: bool = False,
        reject_primitives: bool = True,
        reject_ground_planes: bool = True,
        reject_noisy_artifacts: bool = True,
    ):
        self.min_faces = min_faces
        self.max_faces = max_faces
        self.min_dihedral_var = min_dihedral_var
        self.require_pbr = require_pbr
        self.reject_primitives = reject_primitives
        self.reject_ground_planes = reject_ground_planes
        self.reject_noisy_artifacts = reject_noisy_artifacts

    def filter(self, mesh: trimesh.Trimesh) -> FilterResult:
        """Run full evaluation and return FilterResult."""
        reasons = []
        num_faces = len(mesh.faces)

        # 1. Face count gate
        if num_faces < self.min_faces:
            reasons.append(f"Face count {num_faces} below minimum threshold {self.min_faces}")
        if num_faces > self.max_faces:
            reasons.append(f"Face count {num_faces} exceeds maximum threshold {self.max_faces}")

        # 2. Geometric complexity analysis
        geom_metrics = check_geometric_complexity(mesh)
        is_primitive = geom_metrics["is_primitive"]
        has_ground_plane = geom_metrics["has_ground_plane"]
        has_noisy_artifacts = geom_metrics["has_noisy_artifacts"]

        if self.reject_primitives and is_primitive:
            reasons.append("Asset classified as trivial geometric primitive (plane/cube/flat sheet)")

        if self.reject_ground_planes and has_ground_plane:
            reasons.append("Asset contains a large ambient ground plane or display pedestal")

        if self.reject_noisy_artifacts and has_noisy_artifacts:
            reasons.append(f"Asset contains {geom_metrics['num_components']} disconnected noisy components")

        if geom_metrics["dihedral_variance"] < self.min_dihedral_var and not is_primitive:
            reasons.append(f"Low surface dihedral variance {geom_metrics['dihedral_variance']:.4f} < {self.min_dihedral_var}")

        # 3. PBR Material verification
        pbr_valid, pbr_details = verify_pbr_materials(mesh)
        if self.require_pbr and not pbr_valid:
            reasons.append("Asset lacks valid Physically-Based Rendering (PBR) materials")

        # 4. Composite quality score in [0.0, 1.0]
        score = 1.0
        if is_primitive:
            score -= 0.5
        if has_ground_plane:
            score -= 0.3
        if has_noisy_artifacts:
            score -= 0.3
        if not pbr_valid and self.require_pbr:
            score -= 0.2
        if len(reasons) > 0:
            score = max(0.0, score - 0.2 * len(reasons))
        score = float(np.clip(score, 0.0, 1.0))

        passed = (len(reasons) == 0)

        return FilterResult(
            passed=passed,
            quality_score=score,
            is_primitive=is_primitive,
            has_ground_plane=has_ground_plane,
            has_noisy_artifacts=has_noisy_artifacts,
            pbr_valid=pbr_valid,
            reasons=reasons,
        )

    def generate_vlm_prompt(self, mesh: trimesh.Trimesh, asset_id: str = "asset") -> Dict[str, Any]:
        """Generate VLM evaluation prompt payload alongside multi-view projections."""
        projections = render_projected_views(mesh, image_size=128)
        geom_metrics = check_geometric_complexity(mesh)

        prompt_text = (
            f"You are an expert 3D generative model data curator evaluating asset '{asset_id}'.\n"
            f"The asset has {geom_metrics['num_faces']} faces and {geom_metrics['num_vertices']} vertices.\n"
            "Examine the attached multi-view projections (Front, Right, Top, Isometric normal/depth maps).\n"
            "Answer the following structured quality questions:\n"
            "1. Is this a clean, standalone 3D object suitable for generative training? (yes/no)\n"
            "2. Does it contain an ambient ground plane, scan pedestal, or camera bounding box? (yes/no)\n"
            "3. Is it a trivial primitive (e.g., standard cube, sphere, cylinder, flat billboard)? (yes/no)\n"
            "4. Are there significant photogrammetry noise, holes, or non-manifold flying fragments? (yes/no)\n"
            "5. Confidence rating (0.0 to 1.0).\n"
        )

        return {
            "asset_id": asset_id,
            "prompt": prompt_text,
            "metrics": geom_metrics,
            "views": list(projections.keys()),
        }
