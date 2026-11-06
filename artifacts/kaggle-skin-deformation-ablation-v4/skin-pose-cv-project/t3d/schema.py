"""Validated uncompressed Track A tensors; no learned fields are fabricated."""
from dataclasses import dataclass
import numpy as np
from .serialization import serialization_order


def _finite(value, shape, name, bounded=False):
    a = np.array(value, dtype=np.float64, copy=True)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    if bounded and (np.any(a < 0) or np.any(a > 1)):
        raise ValueError(f'{name} must lie in [0, 1]')
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class VoxelAsset:
    coords: np.ndarray
    dual_vertices: np.ndarray
    intersected_flag: np.ndarray
    grid_size: tuple[int, int, int]
    aabb: np.ndarray
    split_weight: np.ndarray | None = None
    pbr: np.ndarray | None = None
    skeleton_field: np.ndarray | None = None
    skin_embedding: np.ndarray | None = None

    def __post_init__(self):
        grid = np.asarray(self.grid_size)
        if (grid.shape != (3,) or not np.issubdtype(grid.dtype, np.integer)
                or np.any(grid <= 0) or np.any(grid > 65536)):
            raise ValueError('Grid size must be three integers in [1, 65536]')
        object.__setattr__(self, 'grid_size', tuple(int(x) for x in grid))
        coords = np.array(self.coords, copy=True)
        if (coords.ndim != 2 or coords.shape[1] != 3
                or not np.issubdtype(coords.dtype, np.integer)
                or np.any(coords < 0) or np.any(coords >= grid)):
            raise ValueError('Coordinates must be integer (L, 3) within the grid')
        depth = max(1, (int(grid.max()) - 1).bit_length())
        serialization_order(coords, depth=depth)
        coords = coords.astype(np.uint16)
        coords.setflags(write=False)
        object.__setattr__(self, 'coords', coords)
        n = len(coords)
        bounds = _finite(self.aabb, (2, 3), 'aabb')
        if np.any(bounds[1] <= bounds[0]):
            raise ValueError('AABB must have positive extent on every axis')
        object.__setattr__(self, 'aabb', bounds)
        flags = np.array(self.intersected_flag, copy=True)
        if flags.shape != (n, 3) or flags.dtype != np.bool_:
            raise ValueError('Intersection flags must be boolean (L, 3)')
        flags.setflags(write=False)
        object.__setattr__(self, 'intersected_flag', flags)
        for name, channels, bounded in [('dual_vertices', 3, True), ('split_weight', 1, False),
                                        ('pbr', 6, True), ('skeleton_field', 6, False),
                                        ('skin_embedding', 4, False)]:
            value = getattr(self, name)
            if value is None:
                if name == 'dual_vertices':
                    raise ValueError('Dual vertices are required')
                continue
            object.__setattr__(self, name, _finite(value, (n, channels), name, bounded))

    @property
    def voxel_size(self):
        return (self.aabb[1] - self.aabb[0]) / np.asarray(self.grid_size)

    @property
    def world_vertices(self):
        return self.aabb[0] + (self.coords + self.dual_vertices) * self.voxel_size

    def canonical(self, *, transposed=False, curve='z'):
        """Sort every aligned attribute together, preserving row correspondence."""
        depth = max(1, (max(self.grid_size) - 1).bit_length())
        order, _ = serialization_order(self.coords, depth=depth, transposed=transposed, curve=curve)
        values = {name: (None if getattr(self, name) is None else getattr(self, name)[order])
                  for name in ('coords', 'dual_vertices', 'intersected_flag', 'split_weight',
                               'pbr', 'skeleton_field', 'skin_embedding')}
        return VoxelAsset(**values, grid_size=self.grid_size, aabb=self.aabb)
