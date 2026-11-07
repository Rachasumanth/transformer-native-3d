"""Fast CPU geometry ops for FORGE-3D corpora and inference.

All meshes are normalized into [-0.5, 0.5]^3. Grids are cell-centred: cell i
along an axis is centred at -0.5 + (i + 0.5) / res.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
import zipfile
import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

from .constants import GRID, FINE, MAX_FACES, N_SURFACE, SANITIZE_FACES, pack_bits

try:
    import fast_simplification
except ImportError:  # pragma: no cover - optional accelerated path
    fast_simplification = None


# ---------------------------------------------------------------- mesh loading

def _geometry_meshes(scene):
    if isinstance(scene, trimesh.Trimesh):
        return [scene]
    out = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        geom = scene.geometry.get(name)
        if not isinstance(geom, trimesh.Trimesh) or not len(geom.faces):
            continue
        mesh = geom.copy()
        mesh.apply_transform(transform)
        if np.linalg.det(transform[:3, :3]) < 0:
            mesh.invert()
        out.append(mesh)
    return out


def load_mesh(path):
    """Load any supported mesh into one trimesh.Trimesh (textures dropped)."""
    suffix = Path(path).suffix.lower()
    if suffix == '.npz':
        data = np.load(path)
        keys = set(data.files)
        if {'vertices', 'facets'} <= keys:
            return trimesh.Trimesh(vertices=np.asarray(data['vertices'], dtype=np.float64),
                                   faces=np.asarray(data['facets'], dtype=np.int64),
                                   process=False)
        raise ValueError(f'unsupported npz keys: {sorted(keys)}')
    scene = trimesh.load(str(path), force='scene', process=False)
    meshes = _geometry_meshes(scene)
    if not meshes:
        raise ValueError('no triangle geometry found')
    return trimesh.util.concatenate(meshes)


def _bilinear_tex(tex, uv):
    tex = np.asarray(tex, dtype=np.float32)
    height, width = tex.shape[:2]
    x = np.clip(uv[:, 0], 0.0, 1.0) * (width - 1)
    y = np.clip(uv[:, 1], 0.0, 1.0) * (height - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    top = tex[y0, x0] * (1 - wx) + tex[y0, x1] * wx
    bottom = tex[y1, x0] * (1 - wx) + tex[y1, x1] * wx
    return top * (1 - wy) + bottom * wy


def _as_image_array(image, mode):
    if image is None:
        return None
    array = np.asarray(image.convert(mode), dtype=np.float32) / 255.0
    return array


@dataclass
class RichPBR:
    """Per-face PBR factors plus shared textures for one concatenated mesh."""
    faces: np.ndarray
    base_factor: np.ndarray
    metallic: np.ndarray
    roughness: np.ndarray
    face_tex: np.ndarray
    base_textures: list = field(default_factory=list)
    mr_textures: list = field(default_factory=list)
    corner_uv: np.ndarray | None = None
    vertex_colors: np.ndarray | None = None
    has_texture: bool = False

    def sample(self, face_ids, bary):
        """Return (M, 6) uint8-able values: sRGB rgb, metal, rough, alpha in [0,1]."""
        face_ids = np.asarray(face_ids, dtype=np.int64)
        bary = np.asarray(bary, dtype=np.float64)
        color = self.base_factor[face_ids, :3].astype(np.float64).copy()
        alpha = self.base_factor[face_ids, 3].astype(np.float64).copy()
        metal = self.metallic[face_ids].astype(np.float64).copy()
        rough = self.roughness[face_ids].astype(np.float64).copy()
        if self.corner_uv is not None:
            uv = np.einsum('mi,mij->mj', bary, self.corner_uv[face_ids])
            uv[:, 1] = 1.0 - uv[:, 1]
            tex_ids = self.face_tex[face_ids]
            for tex_id in np.unique(tex_ids):
                selected = tex_ids == tex_id
                base = self.base_textures[tex_id]
                if base is not None:
                    sampled = _bilinear_tex(base, uv[selected])
                    color[selected] *= sampled[:, :3]
                    if sampled.shape[1] > 3:
                        alpha[selected] *= sampled[:, 3]
                mr = self.mr_textures[tex_id] if tex_id < len(self.mr_textures) else None
                if mr is not None:
                    sampled = _bilinear_tex(mr, uv[selected])
                    rough[selected] *= sampled[:, 1]
                    metal[selected] *= sampled[:, 2]
        if self.vertex_colors is not None and self.has_texture:
            corners = self.faces[face_ids]
            color = np.einsum('mi,mic->mc', bary, self.vertex_colors[corners, :3])
        return np.concatenate([np.clip(color, 0, 1), np.clip(metal, 0, 1)[:, None],
                               np.clip(rough, 0, 1)[:, None], np.clip(alpha, 0, 1)[:, None]],
                              axis=1)


def _npz_face_count(path):
    try:
        with zipfile.ZipFile(path) as archive:
            name = next((item for item in archive.namelist()
                         if item.endswith('facets.npy')), None)
            if name is None:
                return None
            with archive.open(name) as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(handle)
                else:
                    shape, _, _ = np.lib.format.read_array_header_2_0(handle)
            return int(shape[0])
    except (OSError, ValueError):
        return None


def _glb_face_count(path):
    try:
        with open(path, 'rb') as handle:
            header = handle.read(12)
            if len(header) < 12 or header[:4] != b'glTF':
                return None
            length, kind = np.frombuffer(handle.read(8), dtype='<u4')
            if kind != 0x4E4F534A:
                return None
            document = json.loads(handle.read(int(length)))
    except (OSError, ValueError):
        return None
    faces = 0
    for mesh in document.get('meshes', []):
        for primitive in mesh.get('primitives', []):
            index = primitive.get('indices')
            if index is not None:
                faces += int(document['accessors'][index]['count']) // 3
    return faces


def _declared_gltf_faces(path):
    try:
        document = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    faces = 0
    for mesh in document.get('meshes', []):
        for primitive in mesh.get('primitives', []):
            index = primitive.get('indices')
            if index is not None:
                faces += int(document['accessors'][index]['count']) // 3
    return faces


def load_rich(path):
    """Load a scene with per-face PBR factors/textures for supervision."""
    suffix = Path(path).suffix.lower()
    if suffix == '.gltf':
        declared = _declared_gltf_faces(path)
        if declared is not None and declared > MAX_FACES:
            raise ValueError(f'mesh too dense: {declared} faces')
    elif suffix == '.glb':
        declared = _glb_face_count(path)
        if declared is not None and declared > MAX_FACES:
            raise ValueError(f'mesh too dense: {declared} faces')
    if suffix == '.npz':
        declared = _npz_face_count(path)
        if declared is not None and declared > MAX_FACES:
            raise ValueError(f'mesh too dense: {declared} faces')
        mesh = load_mesh(path)
        return mesh, RichPBR(mesh.faces, np.ones((len(mesh.faces), 4), dtype=np.float32),
                             np.zeros(len(mesh.faces)), np.ones(len(mesh.faces)),
                             np.zeros(len(mesh.faces), dtype=np.int64),
                             [None], [None], vertex_colors=None)
    scene = trimesh.load(str(path), force='scene', process=False)
    meshes = _geometry_meshes(scene)
    if not meshes:
        raise ValueError('no triangle geometry found')
    vertices, faces, corner_uvs = [], [], []
    base_factors, metallics, roughnesses, face_tex = [], [], [], []
    base_textures, mr_textures, texture_keys = [], [], {}
    vertex_colors = None
    has_vertex_texture = False
    offset = 0
    for mesh in meshes:
        if len(mesh.faces) > MAX_FACES:
            raise ValueError(f'mesh too dense: {len(mesh.faces)} faces')
        visual = mesh.visual
        material = getattr(visual, 'material', None)
        uv = getattr(visual, 'uv', None)
        if uv is not None:
            uv = np.asarray(uv, dtype=np.float64)
            if uv.shape != (len(mesh.vertices), 2) or not np.isfinite(uv).all():
                uv = None
        base = np.ones((len(mesh.faces), 4), dtype=np.float64)
        metal = np.ones(len(mesh.faces), dtype=np.float64)
        rough = np.ones(len(mesh.faces), dtype=np.float64)
        tex_id = np.zeros(len(mesh.faces), dtype=np.int64)
        if isinstance(material, trimesh.visual.material.PBRMaterial):
            factor = getattr(material, 'baseColorFactor', None)
            if factor is not None:
                factor = np.asarray(factor, dtype=np.float64).reshape(-1)
                if factor.max() > 1.5:
                    factor = factor / 255.0
                base = np.tile(factor[:4], (len(mesh.faces), 1))
            metallic = getattr(material, 'metallicFactor', None)
            roughness = getattr(material, 'roughnessFactor', None)
            metal = np.full(len(mesh.faces), 1.0 if metallic is None else float(metallic))
            rough = np.full(len(mesh.faces), 1.0 if roughness is None else float(roughness))
            base_tex = _as_image_array(getattr(material, 'baseColorTexture', None), 'RGBA')
            mr_tex = _as_image_array(getattr(material, 'metallicRoughnessTexture', None), 'RGB')
            key = (id(getattr(material, 'baseColorTexture', None)),
                   id(getattr(material, 'metallicRoughnessTexture', None)))
            if key not in texture_keys:
                texture_keys[key] = len(base_textures)
                base_textures.append(base_tex)
                mr_textures.append(mr_tex)
            tex_id = np.full(len(mesh.faces), texture_keys[key], dtype=np.int64)
            if uv is None:
                uv = np.zeros((len(mesh.vertices), 2), dtype=np.float64)
        elif isinstance(material, trimesh.visual.material.SimpleMaterial):
            color = np.asarray(material.baseColor, dtype=np.float64).reshape(-1)
            if color.max() > 1.5:
                color = color / 255.0
            base = np.tile(np.pad(color[:4], (0, max(0, 4 - len(color[:4]))),
                                  constant_values=1.0)[:4], (len(mesh.faces), 1))
            metal = np.zeros(len(mesh.faces))
            rough = np.full(len(mesh.faces), 0.8)
            base_textures.append(None)
            mr_textures.append(None)
            tex_id = np.full(len(mesh.faces), len(base_textures) - 1, dtype=np.int64)
        else:
            color = getattr(visual, 'vertex_colors', None)
            if color is None:
                color = np.full((len(mesh.vertices), 4), 200, dtype=np.uint8)
            color = np.asarray(color, dtype=np.float64)
            if color.max() > 1.5:
                color = color / 255.0
            if color.shape[1] == 3:
                color = np.concatenate([color, np.ones((len(color), 1))], axis=1)
            if getattr(visual, 'defined', False):
                has_vertex_texture = True
            if vertex_colors is None:
                vertex_colors = np.zeros((0, 4), dtype=np.float64)
            vertex_colors = np.concatenate([vertex_colors, color], axis=0)
            base = np.ones((len(mesh.faces), 4))
            metal = np.zeros(len(mesh.faces))
            rough = np.full(len(mesh.faces), 0.8)
            base_textures.append(None)
            mr_textures.append(None)
            tex_id = np.full(len(mesh.faces), len(base_textures) - 1, dtype=np.int64)
            if uv is None:
                uv = np.zeros((len(mesh.vertices), 2), dtype=np.float64)
        if uv is not None:
            corner_uvs.append(uv[mesh.faces])
        vertices.append(np.asarray(mesh.vertices, dtype=np.float64))
        faces.append(np.asarray(mesh.faces, dtype=np.int64) + offset)
        base_factors.append(base)
        metallics.append(metal)
        roughnesses.append(rough)
        face_tex.append(tex_id)
        offset += len(mesh.vertices)
    merged = trimesh.Trimesh(vertices=np.concatenate(vertices), faces=np.concatenate(faces),
                             process=False)
    uvs = np.concatenate(corner_uvs) if corner_uvs else None
    has_texture = (any(t is not None for t in base_textures)
                   or any(t is not None for t in mr_textures)
                   or has_vertex_texture)
    rich = RichPBR(merged.faces.copy(), np.concatenate(base_factors),
                   np.concatenate(metallics), np.concatenate(roughnesses),
                   np.concatenate(face_tex), base_textures, mr_textures, uvs,
                   vertex_colors, has_texture=has_texture)
    return merged, rich


# ------------------------------------------------------------------ sanitation

def _orient_mesh(mesh):
    """Orient each connected component outward using signed volume."""
    components = mesh.split(only_watertight=False)
    fixed = []
    for component in components:
        if len(component.faces) and component.volume < 0:
            component.invert()
        fixed.append(component)
    if len(fixed) == 1:
        return fixed[0]
    return trimesh.util.concatenate(fixed)

def sanitize_mesh(mesh, *, max_faces=SANITIZE_FACES):
    mesh = mesh.copy()
    mesh.process(validate=True)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces) or not len(mesh.vertices):
        raise ValueError('mesh is empty after sanitation')
    if len(mesh.faces) > max_faces:
        mesh = decimate_mesh(mesh, max_faces)
    mesh = _orient_mesh(mesh)
    extent = mesh.bounds[1] - mesh.bounds[0]
    if not np.isfinite(extent).all() or extent.max() <= 0:
        raise ValueError('mesh has degenerate bounds')
    return mesh


def normalize_mesh(mesh, *, half_extent=0.45):
    lo, hi = mesh.bounds
    center = (lo + hi) / 2.0
    extent = float((hi - lo).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError('degenerate mesh extent')
    scale = 2.0 * half_extent / extent
    mesh = mesh.copy()
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)
    return mesh, center, scale


def transform_rich(rich, center, scale):
    """Apply the same normalization to PBR sampling (textures are per-face)."""
    return rich


def decimate_mesh(mesh, target_faces):
    target_faces = max(4, int(target_faces))
    if fast_simplification is not None:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        try:
            vertices, faces = fast_simplification.simplify(vertices, faces,
                                                           target_count=target_faces)
        except TypeError:
            reduction = 1.0 - target_faces / max(1, len(faces))
            vertices, faces = fast_simplification.simplify(vertices, faces,
                                                           target_reduction=reduction)
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # Dependency-free vertex-cluster fallback: quantize to a grid fine enough
    # to keep roughly the requested face count.
    extent = mesh.bounds[1] - mesh.bounds[0]
    for cells in (256, 128, 96, 64, 48, 32, 24, 16, 12, 8):
        pitch = float(extent.max()) / cells
        quantized = np.round(mesh.vertices / pitch).astype(np.int64)
        _, inverse, counts = np.unique(quantized, axis=0, return_inverse=True,
                                       return_counts=True)
        faces = inverse[mesh.faces]
        keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
        if keep.sum() <= target_faces:
            new_vertices = np.zeros((len(counts), 3))
            np.add.at(new_vertices, inverse, mesh.vertices)
            new_vertices /= np.maximum(counts[:, None], 1)
            return trimesh.Trimesh(vertices=new_vertices, faces=faces[keep], process=False)
    return mesh


# ------------------------------------------------------------------ voxel grids

def _voxel_centers(mesh, pitch):
    grid = mesh.voxelized(pitch=pitch, method='subdivide')
    return np.asarray(grid.points, dtype=np.float64)


def solid_occupancy(mesh, res=GRID, *, close=1):
    """Cell-centred solid occupancy grid with hole closing and fill."""
    pitch = 1.0 / res
    centers = _voxel_centers(mesh, pitch)
    index = np.floor((centers + 0.5) / pitch).astype(np.int64)
    index = np.clip(index, 0, res - 1)
    grid = np.zeros((res, res, res), dtype=bool)
    grid[index[:, 0], index[:, 1], index[:, 2]] = True
    if close:
        grid = ndimage.binary_closing(grid, structure=np.ones((3, 3, 3), dtype=bool),
                                      iterations=int(close))
    grid = ndimage.binary_fill_holes(grid)
    return grid


def occupancy_bits(mesh, res=GRID):
    return pack_bits(solid_occupancy(mesh, res))


# ------------------------------------------------------------------ sampling

def sample_surface(mesh, count):
    points, face_ids = trimesh.sample.sample_surface(mesh, count)
    normals = mesh.face_normals[face_ids]
    return (np.asarray(points, np.float64), np.asarray(normals, np.float64),
            np.asarray(face_ids, np.int64))


def build_sdf_queries(mesh, *, n_surface=N_SURFACE, fine=FINE, rng=None):
    """Surface points + normal offsets, both precisely defined for SDF labels."""
    rng = rng or np.random.default_rng(0)
    points, normals, _ = sample_surface(mesh, n_surface)
    offsets = rng.uniform(-2.5, 2.5, size=(n_surface, 1)) / fine
    offset_points = points + normals * offsets
    coords = np.concatenate([points, offset_points], axis=0)
    query_normals = np.concatenate([normals, normals], axis=0)
    return coords, query_normals


def _closest_chunk(triangles, tree, points, k):
    _, candidates = tree.query(points, k=k)
    candidates = np.asarray(candidates).reshape(len(points), k)
    a = triangles[candidates]
    b = a[:, :, 1] - a[:, :, 0]
    c = a[:, :, 2] - a[:, :, 0]
    p = points[:, None, :] - a[:, :, 0]
    d1 = np.einsum('nki,nki->nk', b, p)
    d2 = np.einsum('nki,nki->nk', c, p)
    bp = points[:, None, :] - a[:, :, 1]
    d3 = np.einsum('nki,nki->nk', b, bp)
    d4 = np.einsum('nki,nki->nk', c, bp)
    cp = points[:, None, :] - a[:, :, 2]
    d5 = np.einsum('nki,nki->nk', b, cp)
    d6 = np.einsum('nki,nki->nk', c, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    safe = np.abs(denom) > 1e-18
    v = np.where(safe, vb / np.where(safe, denom, 1.0), 0.0)
    w = np.where(safe, vc / np.where(safe, denom, 1.0), 0.0)
    closest = a[:, :, 0] + b * v[..., None] + c * w[..., None]

    closest = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a[:, :, 0], closest)
    closest = np.where(((d3 >= 0) & (d4 <= d3))[..., None], a[:, :, 1], closest)
    edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    vab = d1 / np.maximum(d1 - d3, 1e-18)
    closest = np.where(edge_ab[..., None], a[:, :, 0] + b * vab[..., None], closest)
    closest = np.where(((d6 >= 0) & (d5 <= d6))[..., None], a[:, :, 2], closest)
    edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    wac = d2 / np.maximum(d2 - d6, 1e-18)
    closest = np.where(edge_ac[..., None], a[:, :, 0] + c * wac[..., None], closest)
    edge_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    wbc = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-18)
    closest = np.where(edge_bc[..., None], a[:, :, 1] +
                       (a[:, :, 2] - a[:, :, 1]) * wbc[..., None], closest)

    delta = closest - a[:, :, 0]
    norm_b = np.maximum(np.einsum('nki,nki->nk', b, b), 1e-18)
    norm_c = np.maximum(np.einsum('nki,nki->nk', c, c), 1e-18)
    vv = np.einsum('nki,nki->nk', delta, b) / norm_b
    ww = np.einsum('nki,nki->nk', delta, c) / norm_c
    diff = closest - points[:, None, :]
    squared = np.einsum('nki,nki->nk', diff, diff)
    best = np.argmin(squared, axis=1)
    rows = np.arange(len(points))
    face_ids = candidates[rows, best]
    bary = np.stack([1.0 - vv[rows, best] - ww[rows, best],
                     vv[rows, best], ww[rows, best]], axis=1)
    bary = np.clip(bary, 0.0, 1.0)
    bary /= np.maximum(bary.sum(axis=1, keepdims=True), 1e-12)
    return closest[rows, best], np.sqrt(np.maximum(squared[rows, best], 0.0)), \
        face_ids.astype(np.int64), bary


def _closest_triangles(mesh, points, k=24, chunk=4096):
    """Nearest triangle among candidates around k nearest surface samples."""
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if not len(triangles):
        raise ValueError('mesh has no triangles')
    n_samples = min(max(6 * len(triangles), 4096), 600000)
    sample_points, sample_faces = trimesh.sample.sample_surface(mesh, n_samples)
    sample_points = np.asarray(sample_points, dtype=np.float64)
    sample_faces = np.asarray(sample_faces, dtype=np.int64)
    tree = cKDTree(sample_points)
    k = min(k, len(sample_points))
    parts = []
    for i in range(0, len(points), chunk):
        chunk_points = points[i:i + chunk]
        _, nearest = tree.query(chunk_points, k=k)
        nearest = np.asarray(nearest).reshape(len(chunk_points), k)
        face_rows = sample_faces[nearest]
        unique = [np.unique(row) for row in face_rows]
        padded = np.full((len(unique), max(len(u) for u in unique)), -1, np.int64)
        for row, u in enumerate(unique):
            padded[row, :len(u)] = u
        parts.append(_closest_chunk(triangles, chunk_points, padded))
    if len(parts) == 1:
        return parts[0]
    return tuple(np.concatenate([part[j] for part in parts], axis=0)
                 for j in range(4))


def _closest_chunk(triangles, points, face_rows):
    a = triangles[face_rows]
    valid = face_rows >= 0
    a = np.where(valid[..., None, None], a, 0.0)
    b = a[:, :, 1] - a[:, :, 0]
    c = a[:, :, 2] - a[:, :, 0]
    p = points[:, None, :] - a[:, :, 0]
    d1 = np.einsum('nki,nki->nk', b, p)
    d2 = np.einsum('nki,nki->nk', c, p)
    bp = points[:, None, :] - a[:, :, 1]
    d3 = np.einsum('nki,nki->nk', b, bp)
    d4 = np.einsum('nki,nki->nk', c, bp)
    cp = points[:, None, :] - a[:, :, 2]
    d5 = np.einsum('nki,nki->nk', b, cp)
    d6 = np.einsum('nki,nki->nk', c, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    safe = np.abs(denom) > 1e-18
    v = np.where(safe, vb / np.where(safe, denom, 1.0), 0.0)
    w = np.where(safe, vc / np.where(safe, denom, 1.0), 0.0)
    closest = a[:, :, 0] + b * v[..., None] + c * w[..., None]

    closest = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a[:, :, 0], closest)
    closest = np.where(((d3 >= 0) & (d4 <= d3))[..., None], a[:, :, 1], closest)
    edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    vab = d1 / np.maximum(d1 - d3, 1e-18)
    closest = np.where(edge_ab[..., None], a[:, :, 0] + b * vab[..., None], closest)
    closest = np.where(((d6 >= 0) & (d5 <= d6))[..., None], a[:, :, 2], closest)
    edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    wac = d2 / np.maximum(d2 - d6, 1e-18)
    closest = np.where(edge_ac[..., None], a[:, :, 0] + c * wac[..., None], closest)
    edge_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    wbc = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-18)
    closest = np.where(edge_bc[..., None], a[:, :, 1] +
                       (a[:, :, 2] - a[:, :, 1]) * wbc[..., None], closest)

    delta = closest - a[:, :, 0]
    norm_b = np.maximum(np.einsum('nki,nki->nk', b, b), 1e-18)
    norm_c = np.maximum(np.einsum('nki,nki->nk', c, c), 1e-18)
    vv = np.einsum('nki,nki->nk', delta, b) / norm_b
    ww = np.einsum('nki,nki->nk', delta, c) / norm_c
    diff = closest - points[:, None, :]
    squared = np.where(valid, np.einsum('nki,nki->nk', diff, diff), np.inf)
    best = np.argmin(squared, axis=1)
    rows = np.arange(len(points))
    face_ids = face_rows[rows, best]
    bary = np.stack([1.0 - vv[rows, best] - ww[rows, best],
                     vv[rows, best], ww[rows, best]], axis=1)
    bary = np.clip(bary, 0.0, 1.0)
    bary /= np.maximum(bary.sum(axis=1, keepdims=True), 1e-12)
    return closest[rows, best], np.sqrt(np.maximum(squared[rows, best], 0.0)), \
        face_ids.astype(np.int64), bary


def sdf_at(mesh, points):
    """Signed distance via KD-tree triangle candidates and face-normal sign."""
    points = np.asarray(points, dtype=np.float64)
    closest, distances, face_ids, _ = _closest_triangles(mesh, points)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)[face_ids]
    delta = points - closest
    sign = np.sign(np.einsum('ij,ij->i', delta, normals))
    sign[sign == 0] = 1.0
    return (distances * sign).astype(np.float64)


def closest_face_bary(mesh, points):
    """Closest triangle and barycentric coordinates for arbitrary points."""
    _, _, face_ids, bary = _closest_triangles(mesh, points)
    return face_ids, bary


# ------------------------------------------------------------------ reconstruction

def sdf_to_mesh(sdf_grid, res=FINE, level=0.0):
    sdf = np.asarray(sdf_grid, dtype=np.float32)
    vertices, faces, _, _ = measure.marching_cubes(
        sdf, level=level, spacing=(1.0 / res, 1.0 / res, 1.0 / res),
        gradient_direction='ascent')
    vertices = vertices + (-0.5 + 0.5 / res)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def postprocess_mesh(mesh, *, target_faces=None, smooth_iters=2):
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if target_faces and len(mesh.faces) > target_faces:
        mesh = decimate_mesh(mesh, target_faces)
    if smooth_iters:
        trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=int(smooth_iters))
    return _orient_mesh(mesh)


# ------------------------------------------------------------------ metrics

def chamfer_distance(points_a, points_b):
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    da, _ = tree_b.query(points_a)
    db, _ = tree_a.query(points_b)
    return 0.5 * (np.mean(da ** 2) + np.mean(db ** 2)), 0.5 * (np.mean(da) + np.mean(db))


def fscore(points_a, points_b, threshold):
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    da, _ = tree_b.query(points_a)
    db, _ = tree_a.query(points_b)
    precision = np.mean(db < threshold)
    recall = np.mean(da < threshold)
    denom = precision + recall
    return float(2 * precision * recall / denom) if denom > 0 else 0.0


def sample_points(mesh, count):
    return sample_surface(mesh, count)[0]
