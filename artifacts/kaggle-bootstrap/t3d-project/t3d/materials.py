"""CPU reference PBR projection/sampling; textures are decoded float arrays.

Color textures use sRGB RGB and linear alpha; returned voxel RGB is linear.
Texture arrays are top-row first with UV (0,0) at the top-left (glTF convention).
Only bilinear repeat/clamp sampling is provided; no mipmaps or texture transforms.
"""
from dataclasses import dataclass, replace
import numpy as np
from .sampling import _mesh


def _bounded(value, name):
    a = np.array(value, dtype=float, copy=True)
    if not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1):
        raise ValueError(f'{name} must contain finite values in [0,1]')
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class PBRMaterial:
    base_color: tuple = (1., 1., 1., 1.)
    metallic: float = 1.
    roughness: float = 1.
    base_texture: np.ndarray | None = None
    metallic_roughness_texture: np.ndarray | None = None
    wrap: str = 'repeat'
    alpha_mode: str = 'OPAQUE'
    alpha_cutoff: float = .5

    def __post_init__(self):
        color = _bounded(self.base_color, 'base color')
        if color.shape != (4,):
            raise ValueError('Base color factor must be RGBA')
        object.__setattr__(self, 'base_color', color)
        for name in ('metallic', 'roughness', 'alpha_cutoff'):
            if np.asarray(getattr(self, name)).ndim != 0:
                raise ValueError(f'{name} must be scalar')
            _bounded(getattr(self, name), name)
        if self.wrap not in ('repeat', 'clamp') or self.alpha_mode not in ('OPAQUE', 'MASK', 'BLEND'):
            raise ValueError('Unsupported wrap or alpha mode')
        for name in ('base_texture', 'metallic_roughness_texture'):
            texture = getattr(self, name)
            if texture is None:
                continue
            texture = _bounded(texture, name)
            if texture.ndim != 3 or texture.shape[2] not in (3, 4) or min(texture.shape[:2]) == 0:
                raise ValueError('Textures must be nonempty HWC RGB/RGBA arrays')
            object.__setattr__(self, name, texture)


def bilinear(texture, uv, *, wrap='repeat', srgb=False):
    texture = np.asarray(texture, dtype=float)
    uv = np.asarray(uv, dtype=float)
    if (texture.ndim != 3 or texture.shape[2] not in (3, 4) or min(texture.shape[:2]) == 0
            or not np.isfinite(texture).all() or uv.ndim != 2 or uv.shape[1] != 2
            or not np.isfinite(uv).all() or wrap not in ('repeat', 'clamp')):
        raise ValueError('Invalid texture, UV coordinates, or wrap')
    if srgb:
        texture = texture.copy()
        rgb = texture[..., :3]
        texture[..., :3] = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055)**2.4)
    h, w = texture.shape[:2]
    uv = uv % 1 if wrap == 'repeat' else np.clip(uv, 0, 1)
    p = uv * [w, h] - .5
    low = np.floor(p).astype(np.int64)
    frac = p - low
    result = np.zeros((len(uv), texture.shape[2]))
    for dx in (0, 1):
        for dy in (0, 1):
            xy = low + [dx, dy]
            xy = xy % [w, h] if wrap == 'repeat' else np.clip(xy, [0, 0], [w-1, h-1])
            weight = (frac[:, 0] if dx else 1-frac[:, 0]) * (frac[:, 1] if dy else 1-frac[:, 1])
            result += texture[xy[:, 1], xy[:, 0]] * weight[:, None]
    return result


def closest_surface(points, vertices, faces):
    """Exact triangle projection reference, O(points*faces); return face IDs/barycentrics."""
    v, f, _, _ = _mesh(vertices, faces)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Queries must be finite (N,3)')
    ids = np.zeros(len(points), dtype=np.int64)
    bary = np.zeros((len(points), 3))
    best = np.full(len(points), np.inf)
    for i, tri in enumerate(v[f]):
        e0, e1 = tri[1] - tri[0], tri[2] - tri[0]
        projected = np.linalg.lstsq(np.column_stack((e0, e1)), (points-tri[0]).T, rcond=None)[0].T
        plane = np.column_stack((1-projected.sum(axis=1), projected))
        candidates = [plane]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edge = tri[b] - tri[a]
            t = np.clip((points-tri[a]) @ edge / (edge @ edge), 0, 1)
            bc = np.zeros((len(points), 3))
            bc[:, a], bc[:, b] = 1-t, t
            candidates.append(bc)
        for j, bc in enumerate(candidates):
            distance = np.sum((bc @ tri - points)**2, axis=1)
            if j == 0:
                distance[np.any(bc < 0, axis=1)] = np.inf
            take = distance < best
            best[take], ids[take], bary[take] = distance[take], i, bc[take]
    return ids, bary


def sample_pbr(asset, vertices, faces, materials, face_materials, *, corner_uv=None):
    """Attach six-channel linear RGB/metallic/roughness/alpha to a VoxelAsset.

    Per-corner UVs (F,3,2) preserve seams. Projection uses each dual vertex and
    the exact closest source triangle. No learned material values are invented.
    """
    vertices, faces, _, _ = _mesh(vertices, faces)
    mapping = np.asarray(face_materials)
    if (mapping.shape != (len(faces),) or not np.issubdtype(mapping.dtype, np.integer)
            or np.any(mapping < 0) or np.any(mapping >= len(materials))):
        raise ValueError('Each face needs a valid integer material index')
    if not all(isinstance(m, PBRMaterial) for m in materials):
        raise ValueError('Materials must be PBRMaterial objects')
    textured = any(m.base_texture is not None or m.metallic_roughness_texture is not None for m in materials)
    uv = None if corner_uv is None else np.asarray(corner_uv, dtype=float)
    if textured and uv is None:
        raise ValueError('Textured materials require per-corner UVs')
    if uv is not None and (uv.shape != (len(faces), 3, 2) or not np.isfinite(uv).all()):
        raise ValueError('UVs must be finite (F,3,2)')
    face_ids, bary = closest_surface(asset.world_vertices, vertices, faces)
    query_uv = None if uv is None else np.einsum('ni,nij->nj', bary, uv[face_ids])
    values = np.empty((len(asset.coords), 6))
    for i, material in enumerate(materials):
        selected = np.flatnonzero(mapping[face_ids] == i)
        color = np.tile(material.base_color, (len(selected), 1))
        metal = np.full(len(selected), material.metallic)
        rough = np.full(len(selected), material.roughness)
        if material.base_texture is not None:
            tex = bilinear(material.base_texture, query_uv[selected], wrap=material.wrap, srgb=True)
            color[:, :3] *= tex[:, :3]
            if tex.shape[1] == 4:
                color[:, 3] *= tex[:, 3]
        if material.metallic_roughness_texture is not None:
            tex = bilinear(material.metallic_roughness_texture, query_uv[selected], wrap=material.wrap)
            metal *= tex[:, 2]
            rough *= tex[:, 1]
        if material.alpha_mode == 'OPAQUE':
            color[:, 3] = 1
        elif material.alpha_mode == 'MASK':
            color[:, 3] = color[:, 3] >= material.alpha_cutoff
        values[selected] = np.column_stack((color[:, :3], metal, rough, color[:, 3]))
    return replace(asset, pbr=np.clip(values, 0, 1))
