"""Single-Atlas UV Unwrapping and PBR Raycast & Closest-Point Texture Baking.

Canonicalizes multi-material, vertex-colored, and textured 3D meshes into a
single guaranteed non-overlapping [0, 1]^2 UV atlas with complete PBR maps:
- Basecolor (RGB, debaked luminance clamped to [0.04, 0.96])
- Metallic-Roughness (2-channel: G=Roughness, B=Metallic; plus 3-channel glTF PBR)
- Normal map (RGB tangent-space or object-space)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import trimesh

from t3d.proximity import TriangleBVH
from t3d.materials import bilinear


# ITU-R BT.709 / sRGB relative luminance weights
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _extract_mesh_arrays(mesh: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Extract and validate (V, 3) vertices and (F, 3) triangle faces from various mesh representations."""
    if isinstance(mesh, (tuple, list)) and len(mesh) == 2:
        v, f = mesh
    elif hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        v = mesh.vertices
        f = mesh.faces
    elif isinstance(mesh, dict) and "vertices" in mesh and "faces" in mesh:
        v = mesh["vertices"]
        f = mesh["faces"]
    else:
        raise ValueError(f"Unsupported mesh representation: {type(mesh)}")

    vertices = np.asarray(v, dtype=np.float32)
    faces = np.asarray(f, dtype=np.uint32)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Mesh vertices must be (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Mesh faces must be (F, 3), got {faces.shape}")
    if len(vertices) < 3 or len(faces) < 1:
        raise ValueError(f"Mesh must have at least 3 vertices and 1 face, got {len(vertices)}V, {len(faces)}F")
    if not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices must contain only finite float values")
    if (faces >= len(vertices)).any():
        raise ValueError("Faces index out of bounds of vertex array")

    return vertices, faces


def unwrap_single_atlas(
    clean_mesh: Any,
    texture_res: int = 1024,
    padding: int = 4,
) -> Dict[str, Any]:
    """Unwrap a clean mesh into a single guaranteed non-overlapping [0, 1]^2 UV atlas.

    Uses `xatlas` to compute conformal charts and pack them into a normalized
    single atlas with the given padding and target resolution.

    Args:
        clean_mesh: A trimesh.Trimesh, object with vertices/faces, or (V, F) tuple.
        texture_res: Target texture resolution (e.g. 1024).
        padding: Padding in pixels between packed UV charts (default: 4px).

    Returns:
        Dictionary containing:
            - 'uv': (V_unwrapped, 2) float32 array in [0, 1]^2.
            - 'indices': (F, 3) uint32 triangle face array indexing into unwrapped vertices.
            - 'vmapping': (V_unwrapped,) uint32 mapping from unwrapped vertex to original vertex.
            - 'vertices': (V_unwrapped, 3) float32 unwrapped vertex positions in 3D.
            - 'faces': (F, 3) uint32 unwrapped faces (same as 'indices').
            - 'corner_uv': (F, 3, 2) float32 per-face corner UVs.
            - 'chart_count': int number of unwrapped charts.
            - 'atlas_width': int xatlas raster width.
            - 'atlas_height': int xatlas raster height.
            - 'mesh': trimesh.Trimesh with unwrapped geometry and TextureVisuals.
    """
    import xatlas

    if not isinstance(texture_res, int) or texture_res < 16 or texture_res > 8192:
        raise ValueError(f"texture_res must be an integer in [16, 8192], got {texture_res}")
    if not isinstance(padding, int) or padding < 0:
        raise ValueError(f"padding must be a non-negative integer, got {padding}")

    orig_vertices, orig_faces = _extract_mesh_arrays(clean_mesh)

    atlas = xatlas.Atlas()
    atlas.add_mesh(orig_vertices, orig_faces)

    chart_opts = xatlas.ChartOptions()
    pack_opts = xatlas.PackOptions()
    pack_opts.padding = int(padding)
    pack_opts.resolution = int(texture_res)

    atlas.generate(chart_options=chart_opts, pack_options=pack_opts)
    vmapping, indices, uvs = atlas.get_mesh(0)

    # Ensure UV bounds strictly in [0, 1]^2
    uvs = np.clip(np.asarray(uvs, dtype=np.float32), 0.0, 1.0)
    indices = np.asarray(indices, dtype=np.uint32)
    vmapping = np.asarray(vmapping, dtype=np.uint32)

    unwrapped_vertices = orig_vertices[vmapping]
    corner_uv = uvs[indices]

    visual = trimesh.visual.TextureVisuals(uv=uvs)
    mesh = trimesh.Trimesh(
        vertices=unwrapped_vertices,
        faces=indices,
        visual=visual,
        process=False,
    )

    return {
        "uv": uvs,
        "indices": indices,
        "vmapping": vmapping,
        "vertices": unwrapped_vertices,
        "faces": indices,
        "corner_uv": corner_uv,
        "chart_count": int(atlas.chart_count),
        "atlas_width": int(atlas.width),
        "atlas_height": int(atlas.height),
        "mesh": mesh,
    }


def debake_luminance(
    basecolor: np.ndarray,
    min_lum: float = 0.04,
    max_lum: float = 0.96,
) -> np.ndarray:
    """Clamp basecolor luminance to [min_lum, max_lum] to strip baked shadows and specular blowouts.

    Preserves chromaticity and hue by scaling RGB channels proportionally to luminance.

    Args:
        basecolor: (..., 3) array of RGB float values in [0, 1].
        min_lum: Minimum luminance threshold (default: 0.04).
        max_lum: Maximum luminance threshold (default: 0.96).

    Returns:
        (..., 3) array with luminance strictly clamped to [min_lum, max_lum].
    """
    rgb = np.asarray(basecolor, dtype=np.float32).copy()
    if rgb.shape[-1] < 3:
        raise ValueError(f"basecolor must have at least 3 channels, got {rgb.shape}")
    if min_lum < 0.0 or max_lum > 1.0 or min_lum > max_lum:
        raise ValueError(f"Invalid luminance bounds [{min_lum}, {max_lum}]")

    weights = LUMINANCE_WEIGHTS
    lum = np.sum(rgb[..., :3] * weights, axis=-1, keepdims=True)

    # Safe proportional scale factor
    scale = np.where(
        lum > 1e-6,
        np.clip(lum, min_lum, max_lum) / np.maximum(lum, 1e-6),
        1.0,
    )
    adjusted = rgb[..., :3] * scale

    # Pixels that were completely black (or near-zero luminance) receive minimum neutral luminance
    zero_mask = lum <= 1e-6
    adjusted = np.where(zero_mask, min_lum, adjusted)

    return np.clip(adjusted, 0.0, 1.0)


def pad_texture(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Pad uncovered atlas texels using exact nearest-neighbor Euclidean distance transform.

    Preserves all covered texel values exactly while extending chart borders
    into atlas background gaps to eliminate bilinear mipmap bleeding.

    Args:
        image: (H, W, C) or (H, W) array of texture values.
        valid_mask: (H, W) boolean mask of covered texels.

    Returns:
        Padded texture array of identical shape and dtype.
    """
    arr = np.asarray(image)
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.shape != arr.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match image spatial shape {arr.shape[:2]}")
    if mask.all() or not mask.any():
        return arr.copy()

    # Exact nearest neighbor query via Euclidean distance transform indices
    indices = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    padded = arr[indices[0], indices[1]]
    return padded


def compute_tangent_frames(
    vertices: np.ndarray,
    faces: np.ndarray,
    corner_uvs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-triangle tangent (T), bitangent (B), and normal (N) orthonormal frames.

    Args:
        vertices: (V, 3) float array of vertex positions.
        faces: (F, 3) int array of triangle indices.
        corner_uvs: (F, 3, 2) float array of UV coordinates per face corner.

    Returns:
        T: (F, 3) normalized tangent vectors.
        B: (F, 3) normalized bitangent vectors.
        N: (F, 3) normalized face normal vectors.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0

    uv0 = corner_uvs[:, 0]
    uv1 = corner_uvs[:, 1]
    uv2 = corner_uvs[:, 2]

    du1 = uv1[:, 0] - uv0[:, 0]
    dv1 = uv1[:, 1] - uv0[:, 1]
    du2 = uv2[:, 0] - uv0[:, 0]
    dv2 = uv2[:, 1] - uv0[:, 1]

    det = du1 * dv2 - du2 * dv1
    inv_det = np.where(np.abs(det) < 1e-12, 0.0, 1.0 / np.where(np.abs(det) < 1e-12, 1e-12, det))

    # Face normal
    N = np.cross(e1, e2)
    n_len = np.linalg.norm(N, axis=-1, keepdims=True)
    N = np.where(n_len > 1e-12, N / np.maximum(n_len, 1e-12), np.array([0.0, 0.0, 1.0], dtype=np.float32))

    # Raw tangents
    T_raw = (dv2[:, None] * e1 - dv1[:, None] * e2) * inv_det[:, None]
    B_raw = (-du2[:, None] * e1 + du1[:, None] * e2) * inv_det[:, None]

    # Gram-Schmidt orthogonalization for T against N
    T_ortho = T_raw - np.sum(T_raw * N, axis=-1, keepdims=True) * N
    t_len = np.linalg.norm(T_ortho, axis=-1, keepdims=True)

    # Deterministic fallback tangent if degenerate
    default_T = np.where(np.abs(N[:, :1]) < 0.9, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    default_T = default_T - np.sum(default_T * N, axis=-1, keepdims=True) * N
    default_T = default_T / np.maximum(np.linalg.norm(default_T, axis=-1, keepdims=True), 1e-12)

    T = np.where(t_len > 1e-6, T_ortho / np.maximum(t_len, 1e-6), default_T)

    # Bitangent from cross product with handedness
    cross_nt = np.cross(N, T)
    handedness = np.sign(np.sum(cross_nt * B_raw, axis=-1, keepdims=True))
    handedness = np.where(handedness == 0, 1.0, handedness)
    B = cross_nt * handedness

    return T, B, N


def rasterize_uv_atlas(
    vertices: np.ndarray,
    faces: np.ndarray,
    corner_uvs: np.ndarray,
    texture_res: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize UV triangles into an integer face map and compute surface samples.

    Args:
        vertices: (V, 3) unwrapped mesh vertices.
        faces: (F, 3) unwrapped mesh faces.
        corner_uvs: (F, 3, 2) per-corner UV coordinates in [0, 1]^2.
        texture_res: Atlas resolution (H = W = texture_res).

    Returns:
        valid_mask: (H, W) bool array.
        face_id_map: (H, W) int32 array (-1 for empty).
        P_clean: (K, 3) float32 surface positions for each valid texel.
        T_clean: (K, 3) float32 tangent vectors.
        B_clean: (K, 3) float32 bitangent vectors.
        N_clean: (K, 3) float32 normal vectors.
        texel_indices: (K, 2) int array of (y, x) coordinates for valid texels.
    """
    img = Image.new("I", (texture_res, texture_res), 0)
    draw = ImageDraw.Draw(img)

    pts = corner_uvs * float(texture_res)
    for i in range(len(faces)):
        poly = [
            (float(pts[i, 0, 0]), float(pts[i, 0, 1])),
            (float(pts[i, 1, 0]), float(pts[i, 1, 1])),
            (float(pts[i, 2, 0]), float(pts[i, 2, 1])),
        ]
        draw.polygon(poly, fill=i + 1)

    arr = np.array(img, dtype=np.int32)
    face_id_map = arr - 1

    # Ensure small / sub-pixel triangles are preserved at their centroids
    centroids = np.mean(pts, axis=1)
    cx = np.clip(np.round(centroids[:, 0]).astype(int), 0, texture_res - 1)
    cy = np.clip(np.round(centroids[:, 1]).astype(int), 0, texture_res - 1)
    empty_centroids = face_id_map[cy, cx] < 0
    if empty_centroids.any():
        idx_to_fill = np.flatnonzero(empty_centroids)
        face_id_map[cy[idx_to_fill], cx[idx_to_fill]] = idx_to_fill

    valid_mask = face_id_map >= 0
    y_coords, x_coords = np.nonzero(valid_mask)
    fids = face_id_map[y_coords, x_coords]

    # Pixel center UV coordinates
    u = (x_coords + 0.5) / float(texture_res)
    v = (y_coords + 0.5) / float(texture_res)
    p_uv = np.column_stack([u, v])

    # 2D barycentric coordinates in triangle UVs
    a = corner_uvs[fids, 0]
    b = corner_uvs[fids, 1]
    c = corner_uvs[fids, 2]

    denom = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    inv_denom = np.where(np.abs(denom) < 1e-12, 0.0, 1.0 / np.where(np.abs(denom) < 1e-12, 1e-12, denom))
    w0 = ((b[:, 1] - c[:, 1]) * (p_uv[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (p_uv[:, 1] - c[:, 1])) * inv_denom
    w1 = ((c[:, 1] - a[:, 1]) * (p_uv[:, 0] - c[:, 0]) + (a[:, 0] - c[:, 0]) * (p_uv[:, 1] - c[:, 1])) * inv_denom
    w2 = 1.0 - w0 - w1

    weights = np.column_stack([w0, w1, w2])
    weights = np.clip(weights, 0.0, 1.0)
    w_sum = weights.sum(axis=1, keepdims=True)
    weights = np.where(w_sum > 1e-12, weights / np.maximum(w_sum, 1e-12), np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]))

    # Compute 3D position on clean mesh
    v0 = vertices[faces[fids, 0]]
    v1 = vertices[faces[fids, 1]]
    v2 = vertices[faces[fids, 2]]
    P_clean = weights[:, 0:1] * v0 + weights[:, 1:2] * v1 + weights[:, 2:3] * v2

    # Compute tangent frames per face
    T_all, B_all, N_all = compute_tangent_frames(vertices, faces, corner_uvs)
    T_clean = T_all[fids]
    B_clean = B_all[fids]
    N_clean = N_all[fids]

    texel_indices = np.column_stack([y_coords, x_coords])

    return valid_mask, face_id_map, P_clean, T_clean, B_clean, N_clean, texel_indices


@dataclass
class RawAppearanceData:
    """Internal normalized representation of a raw mesh's geometry and appearance."""
    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: Optional[np.ndarray] = None
    face_colors: Optional[np.ndarray] = None
    corner_uv: Optional[np.ndarray] = None
    uv: Optional[np.ndarray] = None
    base_color_texture: Optional[np.ndarray] = None
    metallic_roughness_texture: Optional[np.ndarray] = None
    normal_texture: Optional[np.ndarray] = None
    base_color_factor: np.ndarray = None
    metallic_factor: float = 0.0
    roughness_factor: float = 0.5
    materials: Optional[List[Any]] = None
    face_materials: Optional[np.ndarray] = None


def _normalize_raw_mesh(raw_mesh: Any) -> RawAppearanceData:
    """Parse any raw mesh format (trimesh, Scene, ImportedMesh, dict) into RawAppearanceData."""
    if isinstance(raw_mesh, trimesh.Scene):
        # Flatten trimesh.Scene
        geometries = [raw_mesh.geometry[name] for name in raw_mesh.graph.nodes_geometry]
        if not geometries:
            raise ValueError("Scene contains no meshes")
        raw_mesh = trimesh.util.concatenate(raw_mesh.dump())

    # ImportedMesh from t3d.import_glb
    if hasattr(raw_mesh, "face_materials") and hasattr(raw_mesh, "materials"):
        vertices = np.asarray(raw_mesh.vertices, dtype=np.float32)
        faces = np.asarray(raw_mesh.faces, dtype=np.uint32)
        corner_uv = np.asarray(raw_mesh.corner_uv, dtype=np.float32) if hasattr(raw_mesh, "corner_uv") else None
        return RawAppearanceData(
            vertices=vertices,
            faces=faces,
            corner_uv=corner_uv,
            materials=raw_mesh.materials,
            face_materials=np.asarray(raw_mesh.face_materials, dtype=int),
        )

    # Standard trimesh.Trimesh
    if isinstance(raw_mesh, trimesh.Trimesh) or (hasattr(raw_mesh, "vertices") and hasattr(raw_mesh, "faces")):
        vertices = np.asarray(raw_mesh.vertices, dtype=np.float32)
        faces = np.asarray(raw_mesh.faces, dtype=np.uint32)
        visual = getattr(raw_mesh, "visual", None)

        vc = None
        fc = None
        uv = None
        b_tex = None
        mr_tex = None
        n_tex = None
        b_factor = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        metal = 0.0
        rough = 0.5

        if visual is not None:
            # Vertex colors
            if hasattr(visual, "vertex_colors") and visual.vertex_colors is not None and len(visual.vertex_colors):
                vc_raw = np.asarray(visual.vertex_colors, dtype=np.float32)
                if vc_raw.max() > 1.0:
                    vc_raw = vc_raw / 255.0
                vc = vc_raw[:, :3]
            # Face colors
            if hasattr(visual, "face_colors") and visual.face_colors is not None and len(visual.face_colors):
                fc_raw = np.asarray(visual.face_colors, dtype=np.float32)
                if fc_raw.max() > 1.0:
                    fc_raw = fc_raw / 255.0
                fc = fc_raw[:, :3]
            # UVs
            if hasattr(visual, "uv") and visual.uv is not None and len(visual.uv):
                uv = np.asarray(visual.uv, dtype=np.float32)

            # Material
            mat = getattr(visual, "material", None)
            if mat is not None:
                # Convert SimpleMaterial to PBR if necessary
                if hasattr(mat, "to_pbr") and not isinstance(mat, trimesh.visual.material.PBRMaterial):
                    try:
                        mat = mat.to_pbr()
                    except Exception:
                        pass

                # Base color texture
                b_img = getattr(mat, "baseColorTexture", getattr(mat, "image", None))
                if b_img is not None:
                    if isinstance(b_img, Image.Image):
                        b_tex = np.asarray(b_img.convert("RGBA"), dtype=np.float32) / 255.0
                    else:
                        b_tex = np.asarray(b_img, dtype=np.float32)
                        if b_tex.max() > 1.0:
                            b_tex = b_tex / 255.0

                # Metallic roughness texture
                mr_img = getattr(mat, "metallicRoughnessTexture", None)
                if mr_img is not None:
                    if isinstance(mr_img, Image.Image):
                        mr_tex = np.asarray(mr_img.convert("RGBA"), dtype=np.float32) / 255.0
                    else:
                        mr_tex = np.asarray(mr_img, dtype=np.float32)
                        if mr_tex.max() > 1.0:
                            mr_tex = mr_tex / 255.0

                # Normal texture
                norm_img = getattr(mat, "normalTexture", None)
                if norm_img is not None:
                    if isinstance(norm_img, Image.Image):
                        n_tex = np.asarray(norm_img.convert("RGB"), dtype=np.float32) / 255.0
                    else:
                        n_tex = np.asarray(norm_img, dtype=np.float32)
                        if n_tex.max() > 1.0:
                            n_tex = n_tex / 255.0

                # Factors
                bf = getattr(mat, "baseColorFactor", getattr(mat, "diffuse", None))
                if bf is not None:
                    bf = np.asarray(bf, dtype=np.float32)
                    if bf.max() > 1.0:
                        bf = bf / 255.0
                    b_factor = bf[:4] if len(bf) >= 4 else np.append(bf[:3], 1.0)

                mf = getattr(mat, "metallicFactor", None)
                if mf is not None:
                    metal = float(mf)
                rf = getattr(mat, "roughnessFactor", None)
                if rf is not None:
                    rough = float(rf)

        return RawAppearanceData(
            vertices=vertices,
            faces=faces,
            vertex_colors=vc,
            face_colors=fc,
            uv=uv,
            base_color_texture=b_tex,
            metallic_roughness_texture=mr_tex,
            normal_texture=n_tex,
            base_color_factor=b_factor,
            metallic_factor=metal,
            roughness_factor=rough,
        )

    raise ValueError(f"Unsupported raw mesh format: {type(raw_mesh)}")


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB in [0, 1] to sRGB."""
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * (np.maximum(linear, 0.0) ** (1.0 / 2.4)) - 0.055,
    )


def sample_appearance(
    raw_data: RawAppearanceData,
    raw_face_ids: np.ndarray,
    raw_bary: np.ndarray,
    T_clean: np.ndarray,
    B_clean: np.ndarray,
    N_clean: np.ndarray,
    normal_space: str = "tangent",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample basecolor, metallic-roughness, and normal map at closest surface projections.

    Args:
        raw_data: RawAppearanceData instance.
        raw_face_ids: (K,) face indices on raw mesh.
        raw_bary: (K, 3) barycentric weights on corresponding raw faces.
        T_clean: (K, 3) tangent vectors on clean mesh.
        B_clean: (K, 3) bitangent vectors on clean mesh.
        N_clean: (K, 3) normal vectors on clean mesh.
        normal_space: 'tangent' or 'object'.

    Returns:
        basecolor: (K, 3) linear RGB in [0, 1].
        metallic_roughness: (K, 2) array where channel 0 = Roughness, channel 1 = Metallic.
        normal_rgb: (K, 3) encoded normal map in [0, 1].
    """
    K = len(raw_face_ids)
    v_raw = raw_data.vertices
    f_raw = raw_data.faces

    # 1. Basecolor initialization
    has_color_source = (
        (raw_data.materials is not None and raw_data.face_materials is not None)
        or (raw_data.vertex_colors is not None)
        or (raw_data.face_colors is not None)
        or (raw_data.base_color_texture is not None)
    )
    initial_color = 1.0 if has_color_source else 0.7
    basecolor = np.full((K, 3), initial_color, dtype=np.float32)
    metallic = np.full(K, raw_data.metallic_factor, dtype=np.float32)
    roughness = np.full(K, raw_data.roughness_factor, dtype=np.float32)

    # Base color factor modulation
    if raw_data.base_color_factor is not None:
        basecolor *= raw_data.base_color_factor[:3]

    # ImportedMesh multi-material path
    if raw_data.materials is not None and raw_data.face_materials is not None:
        face_mat_ids = raw_data.face_materials[raw_face_ids]
        corner_uv = raw_data.corner_uv
        query_uv = (
            raw_bary[:, 0:1] * corner_uv[raw_face_ids, 0]
            + raw_bary[:, 1:2] * corner_uv[raw_face_ids, 1]
            + raw_bary[:, 2:3] * corner_uv[raw_face_ids, 2]
        )
        for i, mat in enumerate(raw_data.materials):
            sel = np.flatnonzero(face_mat_ids == i)
            if not len(sel):
                continue
            col = np.tile(mat.base_color[:3], (len(sel), 1))
            met = np.full(len(sel), mat.metallic, dtype=np.float32)
            rou = np.full(len(sel), mat.roughness, dtype=np.float32)

            if mat.base_texture is not None:
                tex = bilinear(mat.base_texture, query_uv[sel], wrap=mat.wrap, srgb=True)
                col *= tex[:, :3]
            if mat.metallic_roughness_texture is not None:
                tex = bilinear(mat.metallic_roughness_texture, query_uv[sel], wrap=mat.wrap)
                rou *= tex[:, 1]
                met *= tex[:, 2]

            basecolor[sel] = col
            metallic[sel] = met
            roughness[sel] = rou
    else:
        # Vertex colors
        if raw_data.vertex_colors is not None:
            vc = raw_data.vertex_colors
            tri_vc = vc[f_raw[raw_face_ids]]
            interp_vc = (
                raw_bary[:, 0:1] * tri_vc[:, 0]
                + raw_bary[:, 1:2] * tri_vc[:, 1]
                + raw_bary[:, 2:3] * tri_vc[:, 2]
            )
            basecolor *= interp_vc[:, :3]

        # Face colors
        elif raw_data.face_colors is not None:
            basecolor *= raw_data.face_colors[raw_face_ids, :3]

        # UV Texture sampling
        if raw_data.uv is not None and (raw_data.base_color_texture is not None or raw_data.metallic_roughness_texture is not None):
            tri_uv = raw_data.uv[f_raw[raw_face_ids]]
            interp_uv = (
                raw_bary[:, 0:1] * tri_uv[:, 0]
                + raw_bary[:, 1:2] * tri_uv[:, 1]
                + raw_bary[:, 2:3] * tri_uv[:, 2]
            )

            if raw_data.base_color_texture is not None:
                sampled_bc = bilinear(raw_data.base_color_texture, interp_uv, wrap="repeat", srgb=True)
                basecolor *= sampled_bc[:, :3]

            if raw_data.metallic_roughness_texture is not None:
                sampled_mr = bilinear(raw_data.metallic_roughness_texture, interp_uv, wrap="repeat")
                roughness *= sampled_mr[:, 1]
                metallic *= sampled_mr[:, 2]

    # Clip values into valid physical ranges
    basecolor = np.clip(basecolor, 0.0, 1.0)
    roughness = np.clip(roughness, 0.0, 1.0)
    metallic = np.clip(metallic, 0.0, 1.0)
    metallic_roughness = np.column_stack([roughness, metallic])

    # 2. Raw Normal computation
    e1_raw = v_raw[f_raw[:, 1]] - v_raw[f_raw[:, 0]]
    e2_raw = v_raw[f_raw[:, 2]] - v_raw[f_raw[:, 0]]
    f_norm_raw = np.cross(e1_raw, e2_raw)
    f_norm_len = np.linalg.norm(f_norm_raw, axis=-1, keepdims=True)
    f_norm_raw = np.where(f_norm_len > 1e-12, f_norm_raw / np.maximum(f_norm_len, 1e-12), np.array([0.0, 0.0, 1.0]))

    N_raw = f_norm_raw[raw_face_ids]

    if normal_space == "tangent":
        # Project raw normal into clean mesh's tangent frame (T, B, N)
        n_x = np.sum(T_clean * N_raw, axis=-1)
        n_y = np.sum(B_clean * N_raw, axis=-1)
        n_z = np.sum(N_clean * N_raw, axis=-1)
        n_ts = np.column_stack([n_x, n_y, n_z])
        ts_len = np.linalg.norm(n_ts, axis=-1, keepdims=True)
        n_ts = np.where(ts_len > 1e-12, n_ts / np.maximum(ts_len, 1e-12), np.array([0.0, 0.0, 1.0]))
        normal_rgb = 0.5 * n_ts + 0.5
    elif normal_space == "object":
        # Direct object-space normal encoding
        normal_rgb = 0.5 * N_raw + 0.5
    else:
        raise ValueError(f"Unsupported normal_space '{normal_space}'; expected 'tangent' or 'object'")

    normal_rgb = np.clip(normal_rgb, 0.0, 1.0)

    return basecolor, metallic_roughness, normal_rgb


def bake_single_atlas_pbr(
    clean_mesh: Any,
    raw_mesh: Any,
    texture_res: int = 1024,
    padding: int = 4,
    normal_space: str = "tangent",
    pad_empty_texels: bool = True,
    query_mode: str = "closest",
) -> Dict[str, Any]:
    """Bake multi-material textures and vertex colors from raw_mesh onto clean_mesh's single UV atlas.

    Performs:
    1. Single-atlas UV unwrapping with xatlas at the given resolution with 4px padding.
    2. Rasterization of clean mesh UVs to sample texel positions and tangent frames.
    3. BVH closest-point projection queries onto the raw mesh surface using TriangleBVH.
    4. PBR attribute transfer (Basecolor, Metallic-Roughness, Normal map).
    5. Luminance clamping of basecolor to [0.04, 0.96] to strip baked shadows and specular blowouts.
    6. Nearest-neighbor atlas dilation into uncovered gaps to prevent mipmap bleed.

    Args:
        clean_mesh: Clean target geometry (trimesh, mesh object, or (V, F)).
        raw_mesh: Original source mesh with materials/colors (trimesh, Scene, ImportedMesh).
        texture_res: Atlas resolution (H = W = texture_res, default 1024).
        padding: Chart padding in pixels (default: 4px).
        normal_space: 'tangent' (default) or 'object'.
        pad_empty_texels: Whether to pad uncovered texels using distance transform (default: True).
        query_mode: 'closest' (default) using TriangleBVH.

    Returns:
        Dictionary containing:
            - 'uv': (V_unwrapped, 2) float32 array in [0, 1]^2.
            - 'indices': (F, 3) uint32 triangle face array.
            - 'vmapping': (V_unwrapped,) uint32 mapping to original vertices.
            - 'basecolor': (H, W, 3) float32 array in [0, 1].
            - 'base_color': alias to 'basecolor'.
            - 'metallic_roughness': (H, W, 2) float32 array (G=Roughness, B=Metallic).
            - 'metallic_roughness_3ch': (H, W, 3) float32 array (glTF RGB: [0, Roughness, Metallic]).
            - 'normal': (H, W, 3) float32 array in [0, 1].
            - 'basecolor_image': PIL.Image.Image in 'RGB' mode.
            - 'base_color_image': alias to 'basecolor_image'.
            - 'metallic_roughness_image': PIL.Image.Image in 'RGB' mode.
            - 'normal_image': PIL.Image.Image in 'RGB' mode.
            - 'valid_mask': (H, W) bool array.
            - 'mesh': trimesh.Trimesh with unwrapped UVs and geometry.
            - 'atlas_info': dict with unwrapping metadata.
    """
    if normal_space not in ("tangent", "object"):
        raise ValueError(f"normal_space must be 'tangent' or 'object', got '{normal_space}'")
    if query_mode not in ("closest", "raycast"):
        raise ValueError(f"query_mode must be 'closest' or 'raycast', got '{query_mode}'")

    # 1. Single-Atlas UV Unwrapping with xatlas
    atlas_data = unwrap_single_atlas(clean_mesh, texture_res=texture_res, padding=padding)
    clean_v = atlas_data["vertices"]
    clean_f = atlas_data["faces"]
    corner_uv = atlas_data["corner_uv"]

    # 2. Rasterize UV atlas
    (
        valid_mask,
        face_id_map,
        P_clean,
        T_clean,
        B_clean,
        N_clean,
        texel_indices,
    ) = rasterize_uv_atlas(clean_v, clean_f, corner_uv, texture_res)

    if not valid_mask.any():
        raise ValueError("UV atlas rasterization yielded zero covered texels")

    # 3. Parse and query raw mesh geometry with TriangleBVH
    raw_data = _normalize_raw_mesh(raw_mesh)
    bvh = TriangleBVH(raw_data.vertices, raw_data.faces)

    # Perform BVH query
    raw_face_ids, raw_bary = bvh.query(P_clean)

    # 4. Sample appearance attributes
    basecolor_samples, mr_samples, normal_samples = sample_appearance(
        raw_data=raw_data,
        raw_face_ids=raw_face_ids,
        raw_bary=raw_bary,
        T_clean=T_clean,
        B_clean=B_clean,
        N_clean=N_clean,
        normal_space=normal_space,
    )

    # 5. Lighting / Specular De-baking: Clamp luminance to [0.04, 0.96]
    basecolor_samples = debake_luminance(basecolor_samples, min_lum=0.04, max_lum=0.96)

    # 6. Assemble 2D texture maps
    H = W = texture_res
    basecolor_map = np.zeros((H, W, 3), dtype=np.float32)
    mr_map = np.zeros((H, W, 2), dtype=np.float32)
    mr_map[..., 0] = 0.5  # default roughness in background
    normal_map = np.full((H, W, 3), [0.5, 0.5, 1.0], dtype=np.float32)

    y_idx = texel_indices[:, 0]
    x_idx = texel_indices[:, 1]

    basecolor_map[y_idx, x_idx] = basecolor_samples
    mr_map[y_idx, x_idx] = mr_samples
    normal_map[y_idx, x_idx] = normal_samples

    # 7. Apply atlas padding to uncovered texels if requested
    if pad_empty_texels:
        basecolor_map = pad_texture(basecolor_map, valid_mask)
        mr_map = pad_texture(mr_map, valid_mask)
        normal_map = pad_texture(normal_map, valid_mask)

    # 8. Create standard 3-channel metallic-roughness (glTF: R=0, G=Roughness, B=Metallic)
    mr_3ch = np.zeros((H, W, 3), dtype=np.float32)
    mr_3ch[..., 1] = mr_map[..., 0]  # Green = Roughness
    mr_3ch[..., 2] = mr_map[..., 1]  # Blue = Metallic

    # 9. Create PIL images (uint8)
    bc_srgb = linear_to_srgb(basecolor_map)
    bc_uint8 = np.clip(np.round(bc_srgb * 255.0), 0, 255).astype(np.uint8)
    mr_uint8 = np.clip(np.round(mr_3ch * 255.0), 0, 255).astype(np.uint8)
    norm_uint8 = np.clip(np.round(normal_map * 255.0), 0, 255).astype(np.uint8)

    basecolor_image = Image.fromarray(bc_uint8, mode="RGB")
    mr_image = Image.fromarray(mr_uint8, mode="RGB")
    normal_image = Image.fromarray(norm_uint8, mode="RGB")

    return {
        "uv": atlas_data["uv"],
        "indices": atlas_data["indices"],
        "vmapping": atlas_data["vmapping"],
        "basecolor": basecolor_map,
        "base_color": basecolor_map,
        "metallic_roughness": mr_map,
        "metallic_roughness_3ch": mr_3ch,
        "normal": normal_map,
        "basecolor_image": basecolor_image,
        "base_color_image": basecolor_image,
        "metallic_roughness_image": mr_image,
        "normal_image": normal_image,
        "valid_mask": valid_mask,
        "mesh": atlas_data["mesh"],
        "atlas_info": {
            "chart_count": atlas_data["chart_count"],
            "atlas_width": atlas_data["atlas_width"],
            "atlas_height": atlas_data["atlas_height"],
            "covered_texels": int(valid_mask.sum()),
            "total_texels": int(H * W),
            "coverage_ratio": float(valid_mask.sum()) / float(H * W),
        },
    }
