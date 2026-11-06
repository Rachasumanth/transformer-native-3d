"""Exact right-handed axial quarter-turn augmentation for points and normals."""
import operator
from .normalization import _points


def quarter_turn(points, turns, *, axis='y'):
    """Rotate around the origin; apply equally to mesh, joints and normals."""
    if axis not in ('x', 'y', 'z'):
        raise ValueError('Axis must be x, y or z')
    turns = operator.index(turns) % 4
    result = _points(points)
    for _ in range(turns):
        if axis == 'x':
            result = tuple((x, -z, y) for x, y, z in result)
        elif axis == 'y':
            result = tuple((z, y, -x) for x, y, z in result)
        else:
            result = tuple((-y, x, z) for x, y, z in result)
    return result
