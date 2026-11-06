"""Experimental T3DV v2 raw profile. See docs/container.md for byte layout."""
from dataclasses import dataclass
from pathlib import Path
import struct
import numpy as np
from .schema import VoxelAsset
from .serialization import serialization_order
from .container_io import atomic_write, seal, verify_and_strip

_HEADER = struct.Struct('<4sI6f3I3f4I28H')
_FIELDS = [('dual_vertices', 3), ('intersected_flag', 3), ('split_weight', 1),
           ('base_color', 3), ('metallic', 1), ('roughness', 1), ('alpha', 1),
           ('skeleton_field', 6), ('skin_embedding', 4)]
_ABSENT = 65535
_CURVES = ('z', 'tz', 'h', 'th')


def _align(n):
    return (n + 63) // 64 * 64


def _bpt(tokens):
    if tokens is None:
        return np.empty(0, dtype='<u2')
    a = np.asarray(tokens)
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) or np.any(a < 0) or np.any(a >= 5120):
        raise ValueError('BPT stream must be integer tokens in [0, 5120)')
    return a.astype('<u2')


@dataclass(frozen=True)
class RawContainer:
    asset: VoxelAsset
    bpt_tokens: np.ndarray
    curve: str = 'z'


def dumps(asset: VoxelAsset, *, bpt_tokens=None, curve='z'):
    if curve not in _CURVES:
        raise ValueError('Unknown serialization curve')
    asset = asset.canonical(curve='h' if curve in ('h', 'th') else 'z',
                            transposed=curve in ('tz', 'th'))
    tokens = _bpt(bpt_tokens)
    n = len(asset.coords)
    if max(n, len(tokens)) >= 2**32:
        raise ValueError('Counts exceed uint32')
    slices, arrays, width = [], [], 0
    for name, channels in _FIELDS:
        if name in ('base_color', 'metallic', 'roughness', 'alpha'):
            ranges = dict(base_color=(0, 3), metallic=(3, 4), roughness=(4, 5), alpha=(5, 6))
            lo, hi = ranges[name]
            value = None if asset.pbr is None else asset.pbr[:, lo:hi]
        else:
            value = getattr(asset, name)
        if value is None:
            slices.extend([_ABSENT, _ABSENT])
        else:
            slices.extend([width, width + channels])
            arrays.append(value)
            width += channels
    # Storage precision must not silently produce infinities or collapsed bounds.
    with np.errstate(over='ignore', invalid='ignore'):
        attrs = np.concatenate(arrays, axis=1).astype('<f2')
        bounds = asset.aabb.astype('<f4')
        size = ((bounds[1].astype(float) - bounds[0]) / asset.grid_size).astype('<f4')
    if (not np.isfinite(attrs).all() or not np.isfinite(bounds).all()
            or np.any(bounds[1] <= bounds[0]) or not np.isfinite(size).all() or np.any(size <= 0)):
        raise ValueError('Asset cannot be represented at container precision')
    # Profile 3 is raw profile 1 with a SHA-256 integrity trailer.
    layout = slices + [width, 3, _CURVES.index(curve), 64] + [0]*6
    header = _HEADER.pack(b'T3DV', 2, *bounds.ravel(), *asset.grid_size, *size,
                          n, 0, len(tokens), 0, *layout)
    data = bytearray(header)
    for block in (asset.coords.astype('<u2').tobytes(), attrs.tobytes(), tokens.tobytes()):
        data.extend(bytes(_align(len(data)) - len(data)))
        data.extend(block)
    return seal(bytes(data))


def loads(data):
    data = memoryview(data)
    if len(data) < 128:
        raise ValueError('Truncated T3DV header')
    fields = _HEADER.unpack(data[:128])
    if fields[:2] != (b'T3DV', 2):
        raise ValueError('Unsupported T3DV magic or version')
    bounds = np.array(fields[2:8]).reshape(2, 3)
    grid = tuple(fields[8:11])
    size = np.array(fields[11:14])
    n, latents, count_bpt, joints = fields[14:18]
    layout = fields[18:]
    width = layout[18]
    profile = layout[19]
    if profile not in (1, 3):
        raise ValueError('Unsupported raw profile, curve, reserved fields or nonraw counts')
    if profile == 3:
        data = verify_and_strip(data)
    if (layout[20] not in range(4)
            or layout[21:] != (64, 0, 0, 0, 0, 0, 0) or latents or joints):
        raise ValueError('Unsupported raw profile, curve, reserved fields or nonraw counts')
    if not 6 <= width <= 23:
        raise ValueError('Invalid attribute width')
    if (not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]) or min(grid) <= 0
            or max(grid) > 65536 or not np.isfinite(size).all() or np.any(size <= 0)):
        raise ValueError('Invalid grid geometry')
    if not np.allclose(size, (bounds[1] - bounds[0]) / grid, rtol=2e-7, atol=0):
        raise ValueError('Voxel size disagrees with AABB/grid')
    cursor = 128
    def block(length):
        nonlocal cursor
        start = _align(cursor)
        end = start + length
        if end > len(data) or any(data[cursor:start]):
            raise ValueError('Truncated payload or nonzero alignment padding')
        cursor = end
        return data[start:end]
    coords = np.frombuffer(block(n*6), dtype='<u2').reshape(n, 3)
    attrs = np.frombuffer(block(n*width*2), dtype='<f2').reshape(n, width)
    tokens = np.frombuffer(block(count_bpt*2), dtype='<u2').copy()
    if cursor != len(data):
        raise ValueError('Unexpected trailing payload')
    if not np.isfinite(attrs).all():
        raise ValueError('Nonfinite raw attributes')
    values, expected = {}, 0
    for i, (name, channels) in enumerate(_FIELDS):
        lo, hi = layout[2*i:2*i+2]
        if (lo, hi) == (_ABSENT, _ABSENT):
            if i < 2:
                raise ValueError('Missing mandatory geometry attributes')
            values[name] = None
            continue
        if lo != expected or hi != lo + channels or hi > width:
            raise ValueError('Invalid or overlapping attribute slices')
        values[name] = attrs[:, lo:hi].astype(float)
        expected = hi
    if expected != width:
        raise ValueError('Unclaimed attribute channels')
    pbr_parts = [values.pop(name) for name in ('base_color', 'metallic', 'roughness', 'alpha')]
    if any(p is None for p in pbr_parts) and not all(p is None for p in pbr_parts):
        raise ValueError('PBR channels must be present together')
    flags = values['intersected_flag']
    if not np.all((flags == 0) | (flags == 1)):
        raise ValueError('Edge flags must be exactly 0 or 1')
    values['intersected_flag'] = flags.astype(bool)
    asset = VoxelAsset(coords=coords, grid_size=grid, aabb=bounds,
                       pbr=None if pbr_parts[0] is None else np.concatenate(pbr_parts, axis=1), **values)
    curve = _CURVES[layout[20]]
    order, _ = serialization_order(coords, depth=max(1, (max(grid)-1).bit_length()),
                                   curve='h' if curve in ('h', 'th') else 'z',
                                   transposed=curve in ('tz', 'th'))
    if not np.array_equal(order, np.arange(n)):
        raise ValueError('Coordinates are not in strict declared curve order')
    _bpt(tokens)
    tokens.setflags(write=False)
    return RawContainer(asset, tokens, curve)


def load(path):
    return loads(Path(path).read_bytes())


def save(path, asset, *, bpt_tokens=None, curve='z'):
    """Write only after fully encoding and validating the in-memory payload."""
    data = dumps(asset, bpt_tokens=bpt_tokens, curve=curve)
    loads(data)
    atomic_write(path, data)
