"""Step 1: invertible, aspect-preserving mesh and skeleton normalization."""
from dataclasses import dataclass
import math


def _points(values):
    result = tuple(tuple(float(v) for v in row) for row in values)
    if any(len(row) != 3 or not all(math.isfinite(v) for v in row) for row in result):
        raise ValueError('Points must have three finite coordinates')
    return result


@dataclass(frozen=True)
class CanonicalTransform:
    center: tuple[float, float, float]
    scale: float

    def __post_init__(self):
        center = _points([self.center])[0]
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Scale must be finite and positive')
        object.__setattr__(self, 'center', center)

    def apply(self, points):
        return tuple(tuple((p[a] - self.center[a]) * self.scale for a in range(3))
                     for p in _points(points))

    def inverse(self, points):
        return tuple(tuple(p[a] / self.scale + self.center[a] for a in range(3))
                     for p in _points(points))


def normalize_asset(vertices, joints=(), *, half_extent=0.5):
    """Fit the mesh into [-half_extent, half_extent]^3; transform joints identically.

    Skeleton outliers are preserved, not clipped. Pose inference is a separate
    preprocessing stage. Mesh bounds determine scale to keep geometry consistent
    between rigged and unrigged copies of the same asset.
    """
    vertices, joints = _points(vertices), _points(joints)
    if not vertices:
        raise ValueError('A mesh must contain vertices')
    if not math.isfinite(half_extent) or half_extent <= 0:
        raise ValueError('Half extent must be finite and positive')
    lo = tuple(min(p[a] for p in vertices) for a in range(3))
    hi = tuple(max(p[a] for p in vertices) for a in range(3))
    center = tuple(lo[a] / 2 + hi[a] / 2 for a in range(3))
    extent = max(hi[a] - lo[a] for a in range(3))
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError('Mesh must have finite, nonzero extent')
    transform = CanonicalTransform(center, 2 * half_extent / extent)
    return transform.apply(vertices), transform.apply(joints), transform
