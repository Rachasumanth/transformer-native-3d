"""Device-resident dual-grid extraction using sorted sparse coordinate keys.

Requires PyTorch. This reference GPU path uses binary search, not the planned
custom CUDA hash table. Winding is provisional, as in the CPU extractor.
"""
import torch


def voxels_to_mesh(coords, dual_vertices, intersected_flag, grid_size, aabb,
                   split_weight=None):
    """Extract world vertices and triangles without transferring payloads to CPU.

    Rows retain input order. Missing transverse neighbors and out-of-grid queries
    are excluded before generating triangles. Integer keys use signed int64;
    the grid volume must fit. Outputs remain on the input device. Metadata checks
    synchronize small scalar predicates; no production latency is claimed.
    """
    device = coords.device
    if coords.dtype != torch.int64 or coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('coords must be int64 (L,3)')
    n = len(coords)
    if dual_vertices.shape != (n, 3) or not dual_vertices.is_floating_point():
        raise ValueError('dual_vertices must be floating (L,3)')
    if intersected_flag.shape != (n, 3) or intersected_flag.dtype != torch.bool:
        raise ValueError('intersected_flag must be boolean (L,3)')
    tensors = [dual_vertices, intersected_flag]
    if split_weight is not None:
        if split_weight.shape != (n, 1) or not split_weight.is_floating_point():
            raise ValueError('split_weight must be floating (L,1)')
        tensors.append(split_weight)
    if any(x.device != device for x in tensors):
        raise ValueError('all payloads must share a device')
    import math
    if len(grid_size) != 3 or any(type(x) is not int or x <= 0 for x in grid_size):
        raise ValueError('grid_size must contain three positive integers')
    if math.prod(grid_size) > torch.iinfo(torch.int64).max:
        raise ValueError('grid volume exceeds int64 key range')
    grid = torch.tensor(grid_size, device=device, dtype=torch.int64)
    bounds = torch.as_tensor(aabb, device=device, dtype=dual_vertices.dtype)
    if bounds.shape != (2, 3) or not torch.isfinite(bounds).all() or not (bounds[1] > bounds[0]).all():
        raise ValueError('aabb must have finite strictly increasing bounds')
    if not torch.isfinite(dual_vertices).all() or not ((dual_vertices >= 0) & (dual_vertices <= 1)).all():
        raise ValueError('dual vertices must be finite local offsets in [0,1]')
    if split_weight is not None and not torch.isfinite(split_weight).all():
        raise ValueError('split weights must be finite')
    if not ((coords >= 0) & (coords < grid)).all():
        raise ValueError('coordinates outside grid')
    strides = torch.tensor([grid_size[1]*grid_size[2], grid_size[2], 1], device=device)
    keys = (coords * strides).sum(-1)
    sorted_keys, row_ids = keys.sort()
    if n > 1 and (sorted_keys[1:] == sorted_keys[:-1]).any():
        raise ValueError('duplicate coordinates')
    vertices = bounds[0] + (coords.to(dual_vertices.dtype) + dual_vertices) * ((bounds[1]-bounds[0])/grid)
    empty = torch.empty((0, 3), device=device, dtype=torch.int64)
    if n == 0:
        return vertices, empty
    marked = intersected_flag.nonzero()
    neighbors = torch.tensor([
        [[0,0,0],[0,0,1],[0,1,1],[0,1,0]],
        [[0,0,0],[1,0,0],[1,0,1],[0,0,1]],
        [[0,0,0],[0,1,0],[1,1,0],[1,0,0]],
    ], device=device)
    query = coords[marked[:,0], None, :] + neighbors[marked[:,1]]
    in_bounds = ((query >= 0) & (query < grid)).all(-1)
    query_keys = (query * strides).sum(-1)
    slots = torch.searchsorted(sorted_keys, query_keys).clamp(max=n-1)
    valid = (in_bounds & (sorted_keys[slots] == query_keys)).all(-1)
    quads = row_ids[slots[valid]]
    splits = torch.tensor([[[0,1,2],[0,2,3]], [[0,1,3],[3,1,2]]], device=device)
    if split_weight is not None:
        weights = split_weight[quads,0]
        scale = weights.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        weights = weights / scale
        choice = (weights[:,0]*weights[:,2] <= weights[:,1]*weights[:,3]).long()
    else:
        # (Q, split, triangle, corner, xyz), with exact atan2 angle criterion.
        points = vertices[quads[:,splits]]
        edges_a = points.roll(-1, dims=-2) - points
        edges_b = points.roll(-2, dims=-2) - points
        area = torch.linalg.vector_norm(torch.linalg.cross(edges_a, edges_b, dim=-1), dim=-1)
        dot = (edges_a*edges_b).sum(-1)
        angles = torch.atan2(area, dot)
        scores = angles.amin(dim=(-1,-2))
        scores = scores.masked_fill((area == 0).any(dim=(-1,-2)), -torch.inf)
        choice = (scores[:,0] <= scores[:,1]).long()
    faces = torch.gather(quads[:,None,:].expand(-1,2,-1), 2, splits[choice]).reshape(-1,3)
    return vertices, faces
