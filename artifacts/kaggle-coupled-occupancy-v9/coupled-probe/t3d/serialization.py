"""Unsigned 64-bit Morton keys and indirect sparse feature permutations.

Convention: x occupies bits 0,3,...; y bits 1,4,...; z bits 2,5,... .
TZ exchanges x and y before interleaving. These conventions are explicit local
choices; they do not imply byte compatibility with upstream .vxz files.
"""
import operator
import numpy as np


def _depth(depth):
    depth = operator.index(depth)
    if not 1 <= depth <= 21:
        raise ValueError('Depth must be in [1, 21] for 64-bit 3D keys')
    return depth


def morton_keys(coords, *, depth, batch=None, transposed=False):
    depth = _depth(depth)
    c = np.asarray(coords)
    if c.ndim != 2 or c.shape[1] != 3 or not np.issubdtype(c.dtype, np.integer):
        raise ValueError('Coordinates must be integer (N, 3)')
    if np.any(c < 0) or np.any(c >= (1 << depth)):
        raise ValueError('Coordinates exceed serialization depth')
    c = c.astype(np.uint64)
    if transposed:
        c = c[:, [1, 0, 2]]
    keys = np.zeros(len(c), dtype=np.uint64)
    for bit in range(depth):
        for axis in range(3):
            keys |= ((c[:, axis] >> bit) & 1) << (3 * bit + axis)
    if batch is not None:
        b = np.asarray(batch)
        if b.shape != (len(c),) or not np.issubdtype(b.dtype, np.integer):
            raise ValueError('Batch must be an integer vector of length N')
        if np.any(b < 0) or np.any(b >= (1 << (64 - 3 * depth))):
            raise ValueError('Batch index exceeds remaining key bits')
        keys |= b.astype(np.uint64) << (3 * depth)
    return keys


def morton_decode(keys, *, depth, transposed=False):
    depth = _depth(depth)
    keys = np.asarray(keys)
    if keys.ndim != 1 or not np.issubdtype(keys.dtype, np.integer) or np.any(keys < 0):
        raise ValueError('Keys must be a nonnegative integer vector')
    keys = keys.astype(np.uint64)
    c = np.zeros((len(keys), 3), dtype=np.uint64)
    for bit in range(depth):
        for axis in range(3):
            c[:, axis] |= ((keys >> (3 * bit + axis)) & 1) << bit
    if transposed:
        c = c[:, [1, 0, 2]]
    return c, keys >> (3 * depth)


def hilbert_keys(coords, *, depth, batch=None, transposed=False):
    """CPU Hilbert reference via hilbertcurve 2.x; same batch/depth limits as Z."""
    from hilbertcurve.hilbertcurve import HilbertCurve
    depth = _depth(depth)
    validated = morton_keys(coords, depth=depth, batch=batch)
    c = np.asarray(coords)
    if transposed:
        c = c[:, [1, 0, 2]]
    spatial = np.asarray(HilbertCurve(depth, 3).distances_from_points(c.tolist()), dtype=np.uint64)
    return spatial | ((validated >> (3 * depth)) << (3 * depth))


def hilbert_decode(keys, *, depth, transposed=False):
    from hilbertcurve.hilbertcurve import HilbertCurve
    depth = _depth(depth)
    _, batch = morton_decode(keys, depth=depth)  # validates integer range/shape
    spatial = np.asarray(keys, dtype=np.uint64) & np.uint64((1 << (3 * depth)) - 1)
    coords = np.asarray(HilbertCurve(depth, 3).points_from_distances(spatial.tolist()),
                        dtype=np.uint64).reshape(-1, 3)
    if transposed:
        coords = coords[:, [1, 0, 2]]
    return coords, batch


def serialization_order(coords, *, depth, batch=None, transposed=False, curve='z'):
    """Return order and inverse; features[order][inverse] restores input order.

    Only integer mappings are produced; feature tensors are not copied here.
    Duplicate coordinates within a batch are rejected for sparse-grid identity.
    """
    if curve not in ('z', 'h'):
        raise ValueError('Curve must be z or h; use transposed=True for TZ/TH')
    encode = morton_keys if curve == 'z' else hilbert_keys
    keys = encode(coords, depth=depth, batch=batch, transposed=transposed)
    order = np.argsort(keys, kind='stable')
    if np.any(keys[order][1:] == keys[order][:-1]):
        raise ValueError('Duplicate sparse coordinates within a batch')
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return order, inverse


def serialization_orders(coords, *, depth, batch=None):
    """Four order/inverse mappings for successive attention layers (Z,TZ,H,TH)."""
    return {name: serialization_order(coords, depth=depth, batch=batch,
                                     curve=curve, transposed=transposed)
            for name, curve, transposed in [('z', 'z', False), ('tz', 'z', True),
                                            ('h', 'h', False), ('th', 'h', True)]}
