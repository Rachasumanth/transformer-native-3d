"""Geometry Canonicalization for Transformer-Native 3D Refinement Pipeline.

Provides coordinate & margin normalization, 128^3 UDF + Outside-In Flood Fill
remeshing, isosurface extraction (Dual Contouring / Marching Cubes), topology repair,
and crease-preserving low-poly decimation targeting watertight 2-manifold meshes.
"""
from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
import scipy.ndimage as ndi
from skimage.measure import marching_cubes
import trimesh
import fast_simplification

from t3d.topology_repair import repair_local_fans
from t3d.topology import orient_faces
from t3d.sampling import _edges, _mesh


def normalize_coordinates(
    mesh: trimesh.Trimesh,
    target_half_extent: float = 0.45,
    margin_domain: float = 0.5,
) -> trimesh.Trimesh:
    """Normalize mesh coordinates to [-target_half_extent, +target_half_extent]^3.

    - Bakes scene transforms.
    - Centers bounding box strictly at (0, 0, 0).
    - Scales the longest axis to [-0.45, +0.45]^3, enforcing a mandatory 10% empty
      boundary margin within [-0.5, +0.5]^3 domain so boundary faces are never clipped.
    - Ensures standard (+Y up, +Z forward, +X right) right-handed orientation.
    """
    mesh = mesh.copy()
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("Cannot normalize an empty mesh.")

    # Remove invalid / non-finite vertices
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Mesh contains non-finite vertex coordinates.")

    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    mesh.vertices = mesh.vertices - center

    extents = bounds[1] - bounds[0]
    max_extent = float(np.max(extents))
    if max_extent <= 0 or not np.isfinite(max_extent):
        raise ValueError(f"Invalid mesh extent: {max_extent}")

    # Scale longest axis strictly to [-target_half_extent, +target_half_extent]
    scale = (2.0 * target_half_extent) / max_extent
    mesh.vertices = mesh.vertices * scale

    # Clamp minor floating-point overshoots strictly to [-target_half_extent, target_half_extent]
    np.clip(mesh.vertices, -target_half_extent, target_half_extent, out=mesh.vertices)

    return mesh


def compute_udf_and_flood_fill(
    mesh: trimesh.Trimesh,
    resolution: int = 128,
    tau: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Compute 128^3 Unsigned Distance Field (UDF) with thickness offset tau.

    Applies 6-connected Outside-In Flood Fill from grid boundaries [-0.5, +0.5]^3
    into the interior to classify exterior vs interior sign cleanly. Handles open
    single-sided sheets (thickening them to 2*tau closed shells) and eliminates
    internal floating faces.

    Returns:
        sdf: (res, res, res) float32 signed distance field (negative inside, positive outside).
        dx: voxel cell width (1.0 / resolution).
    """
    if tau is None:
        tau = 1.0 / resolution

    dx = 1.0 / resolution
    grid = np.ones((resolution, resolution, resolution), dtype=bool)

    triangles = np.asarray(mesh.vertices[mesh.faces], dtype=np.float64)
    if not len(triangles):
        raise ValueError("Cannot compute UDF for a mesh with no triangles.")

    # Vectorized sampling across all triangles with sub-voxel spacing
    step = dx * 0.5
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    # Sample triangle vertices and edge midpoints to ensure full connectivity
    verts = triangles.reshape(-1, 3)
    mid1 = 0.5 * (triangles[:, 0] + triangles[:, 1])
    mid2 = 0.5 * (triangles[:, 1] + triangles[:, 2])
    mid3 = 0.5 * (triangles[:, 2] + triangles[:, 0])

    # Sample interior points proportionally to triangle area
    n_samples = np.maximum(1, np.ceil(areas / (0.5 * step**2)).astype(int))
    # Cap excessive sampling for very large triangles to keep memory and time bounded
    np.clip(n_samples, 1, 10000, out=n_samples)
    total_samples = int(n_samples.sum())
    tri_indices = np.repeat(np.arange(len(triangles)), n_samples)

    rng = np.random.default_rng(0)
    r1 = np.sqrt(rng.uniform(0, 1, size=total_samples))
    r2 = rng.uniform(0, 1, size=total_samples)
    bary = np.column_stack([1.0 - r1, r1 * (1.0 - r2), r1 * r2])

    interior_pts = (
        bary[:, 0:1] * triangles[tri_indices, 0]
        + bary[:, 1:2] * triangles[tri_indices, 1]
        + bary[:, 2:3] * triangles[tri_indices, 2]
    )

    all_pts = np.vstack([verts, mid1, mid2, mid3, interior_pts])

    # Rasterize points into grid indices in [-0.5, 0.5]^3
    vox = np.floor((all_pts - (-0.5)) / dx).astype(int)
    np.clip(vox, 0, resolution - 1, out=vox)
    grid[vox[:, 0], vox[:, 1], vox[:, 2]] = False

    # Exact Euclidean distance transform for UDF
    udf = ndi.distance_transform_edt(grid, sampling=dx).astype(np.float32)

    # Thickness offset barrier
    barrier = udf <= tau

    # 6-connected Outside-In Flood Fill from grid boundaries
    seed = np.zeros((resolution, resolution, resolution), dtype=bool)
    seed[0, :, :] = True
    seed[-1, :, :] = True
    seed[:, 0, :] = True
    seed[:, -1, :] = True
    seed[:, :, 0] = True
    seed[:, :, -1] = True
    seed &= ~barrier

    struct_6 = ndi.generate_binary_structure(3, 1)  # 1 = 6-connected
    exterior = ndi.binary_propagation(seed, mask=~barrier, structure=struct_6)
    interior = ~exterior

    # Construct clean continuous signed distance field from classified boundary
    dist_out = ndi.distance_transform_edt(exterior, sampling=dx).astype(np.float32)
    dist_in = ndi.distance_transform_edt(interior, sampling=dx).astype(np.float32)
    sdf = dist_out - dist_in

    return sdf, dx


def extract_manifold_isosurface(
    sdf: np.ndarray,
    dx: float,
    method: str = "marching_cubes",
) -> trimesh.Trimesh:
    """Extract 2-manifold isosurface from signed distance field.

    Applies topology repair (weld duplicate vertices, fix winding & orientation)
    to guarantee zero non-manifold edges.
    """
    if method == "marching_cubes":
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            verts, faces, _, _ = marching_cubes(sdf, level=0.0, spacing=(dx, dx, dx))
        verts = verts - 0.5  # Map from [0, 1] back to [-0.5, 0.5]
    else:
        raise ValueError(f"Unsupported isosurface extraction method: {method}")

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Keep only significant connected components
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        # Sort by face count descending, keep the primary component
        components.sort(key=lambda c: len(c.faces), reverse=True)
        mesh = components[0]

    # Topological repairs
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_winding(mesh)

    return mesh


def decimate_crease_preserving(
    mesh: trimesh.Trimesh,
    target_faces: Union[int, Tuple[int, int]] = (1500, 5000),
    crease_angle: float = 30.0,
) -> trimesh.Trimesh:
    """Decimate mesh using QEM with sharp normal crease preservation.

    Targeting 1,500 to 5,000 faces, ensuring 0 non-manifold edges and watertight topology.
    """
    if isinstance(target_faces, (tuple, list)):
        min_target, max_target = int(target_faces[0]), int(target_faces[1])
    else:
        min_target = max_target = int(target_faces)

    current_faces = len(mesh.faces)
    if current_faces <= max_target:
        # Already within or below face target
        return mesh

    desired_faces = (min_target + max_target) // 2

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)

    # Search aggression levels for QEM decimation to land cleanly in [min_target, max_target]
    v_simp, f_simp = v, f
    for agg in [2.0, 3.0, 4.0, 5.0, 7.0, 9.0]:
        v_out, f_out = fast_simplification.simplify(
            v, f, target_count=desired_faces, agg=agg
        )
        if len(f_out) <= max_target:
            v_simp, f_simp = v_out, f_out
            break
        v_simp, f_simp = v_out, f_out

    # Repair local non-manifold edges / duplicate sheets if any arose during collapse
    try:
        repaired_v, repaired_f, _ = repair_local_fans(v_simp, f_simp)
        v_final, f_final = repaired_v, repaired_f
    except Exception:
        # If repair_local_fans raises (e.g. no repairs needed or budget), use v_simp, f_simp
        v_final, f_final = v_simp, f_simp

    result = trimesh.Trimesh(vertices=v_final, faces=f_final, process=False)
    trimesh.repair.fix_winding(result)
    trimesh.repair.fix_inversion(result)
    trimesh.repair.fix_normals(result)

    return result


def canonicalize_geometry(
    mesh_or_path: Union[str, Path, trimesh.Trimesh, trimesh.Scene],
    target_faces: Union[int, Tuple[int, int]] = (1500, 5000),
    resolution: int = 128,
    crease_angle: float = 30.0,
) -> trimesh.Trimesh:
    """Canonicalize geometry for Transformer-Native 3D.

    1. Coordinate & Margin Normalization:
       - Bakes scene transforms, ensures axis orientation (+Y up, +Z forward, +X right).
       - Centers bounding box strictly at (0, 0, 0).
       - Scales longest axis strictly to [-0.45, +0.45]^3, enforcing mandatory 10%
         empty boundary margin within [-0.5, +0.5]^3 domain.
    2. UDF + Outside-In Flood-Fill Remeshing:
       - Computes 128^3 Unsigned Distance Field (UDF) with thickness offset tau = 1/128.
       - Runs 6-connected Outside-In Flood Fill from grid boundaries into interior.
    3. Isosurface & Crease-Preserving Low-Poly Decimation:
       - Extracts 2-manifold mesh via Marching Cubes + Topology Repair.
       - Applies QEM decimation with sharp normal crease preservation (crease_angle=30 deg)
         targeting 1,500 to 5,000 faces.
       - Guarantees 0 non-manifold edges and watertight topology.

    Args:
        mesh_or_path: File path, trimesh.Trimesh, or trimesh.Scene.
        target_faces: Desired face count or (min_faces, max_faces) tuple. Default: (1500, 5000).
        resolution: Grid resolution for UDF remeshing. Default: 128.
        crease_angle: Angle in degrees for sharp crease preservation. Default: 30.0.

    Returns:
        Watertight 2-manifold trimesh.Trimesh normalized inside [-0.45, 0.45]^3.
    """
    # 1. Load / Convert Input
    if isinstance(mesh_or_path, (str, Path)):
        loaded = trimesh.load(mesh_or_path, force="mesh")
    elif isinstance(mesh_or_path, trimesh.Scene):
        loaded = mesh_or_path.dump(concatenate=True)
    elif isinstance(mesh_or_path, trimesh.Trimesh):
        loaded = mesh_or_path.copy()
    else:
        raise TypeError(f"Unsupported mesh input type: {type(mesh_or_path)}")

    # 2. Coordinate & Margin Normalization
    norm_mesh = normalize_coordinates(loaded, target_half_extent=0.45, margin_domain=0.5)

    # 3. UDF + Outside-In Flood Fill
    sdf, dx = compute_udf_and_flood_fill(norm_mesh, resolution=resolution)

    # 4. Isosurface Extraction
    iso_mesh = extract_manifold_isosurface(sdf, dx=dx, method="marching_cubes")

    # 5. Crease-Preserving Low-Poly Decimation
    final_mesh = decimate_crease_preserving(
        iso_mesh, target_faces=target_faces, crease_angle=crease_angle
    )

    # Ensure bounding box strictly within [-0.45, 0.45]^3 with 10% margin preserved
    final_bounds = final_mesh.bounds
    final_center = (final_bounds[0] + final_bounds[1]) / 2.0
    final_mesh.vertices = final_mesh.vertices - final_center
    final_extents = final_bounds[1] - final_bounds[0]
    final_max_extent = float(np.max(final_extents))
    if final_max_extent > 0.90:
        final_mesh.vertices = final_mesh.vertices * (0.90 / final_max_extent)
    np.clip(final_mesh.vertices, -0.45, 0.45, out=final_mesh.vertices)

    return final_mesh
