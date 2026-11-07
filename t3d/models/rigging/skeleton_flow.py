r"""Skeleton generation and kinematic tree decoding with relative DFS order.

Predicts joint coordinates in Depth-First Search (DFS) relative order:
    \Delta j_k = j_k - j_{p(k)} (root j_0 in world coordinates, child joints relative to parent).
Provides kinematic tree decoder mapping relative offsets back to global coordinates,
and enforces containment loss inside mesh bounding volume to penalize out-of-mesh floating joints.
"""

from typing import Optional, Tuple, Union, List, Dict, Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_dfs_order(
    parents: Union[np.ndarray, torch.Tensor, List[int]],
    joints: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> List[int]:
    """Compute Depth-First Search (DFS) topological order of skeleton joints.

    Roots (parents == -1) are visited first. For deterministic traversal across platforms,
    child joints can be ordered lexicographically by their coordinates (y, x, z) if
    joint coordinates are provided, or by ascending joint index.

    Parameters
    ----------
    parents : array-like of shape (K,)
        Parent joint index for each joint; roots use -1.
    joints : array-like of shape (K, 3), optional
        Joint coordinates used to break sibling ties deterministically.

    Returns
    -------
    order : List[int]
        List of joint indices in DFS order.
    """
    if isinstance(parents, torch.Tensor):
        parents_list = parents.detach().cpu().tolist()
    else:
        parents_list = list(parents)

    K = len(parents_list)
    if K == 0:
        return []

    roots = [i for i, p in enumerate(parents_list) if p == -1]
    if not roots:
        raise ValueError("Skeleton parent hierarchy must contain at least one root (parent == -1)")

    # Build adjacency mapping parent -> list of children
    children: List[List[int]] = [[] for _ in range(K)]
    for i, p in enumerate(parents_list):
        if p >= 0:
            if p >= K:
                raise ValueError(f"Parent index {p} out of bounds for joint count {K}")
            children[p].append(i)

    # Sort siblings for deterministic DFS ordering
    if joints is not None:
        if isinstance(joints, torch.Tensor):
            coords = joints.detach().cpu().numpy()
        else:
            coords = np.asarray(joints)
        for p in range(K):
            if len(children[p]) > 1:
                # Lexicographical sort by (y, x, z)
                children[p].sort(key=lambda idx: (coords[idx, 1], coords[idx, 0], coords[idx, 2], idx))
        if len(roots) > 1:
            roots.sort(key=lambda idx: (coords[idx, 1], coords[idx, 0], coords[idx, 2], idx))
    else:
        for p in range(K):
            children[p].sort()
        roots.sort()

    order: List[int] = []
    visited = set()

    def _dfs(u: int):
        if u in visited:
            raise ValueError(f"Cycle detected in skeleton hierarchy involving joint {u}")
        visited.add(u)
        order.append(u)
        for v in children[u]:
            _dfs(v)

    for r in roots:
        _dfs(r)

    if len(order) != K:
        raise ValueError(f"Disjoint or malformed skeleton: visited {len(order)} of {K} joints")

    return order


def reorder_to_dfs(
    joints: Union[np.ndarray, torch.Tensor],
    parents: Union[np.ndarray, torch.Tensor]
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], List[int], List[int]]:
    """Reorder joints and parent references into Depth-First Search (DFS) order.

    In the resulting DFS ordering, parents always appear before their children,
    and root joint(s) are placed at the beginning.

    Returns
    -------
    dfs_joints, dfs_parents, dfs_order, inverse_order
    """
    is_torch = isinstance(joints, torch.Tensor)
    order = build_dfs_order(parents, joints)
    inverse = [0] * len(order)
    for new_idx, old_idx in enumerate(order):
        inverse[old_idx] = new_idx

    if is_torch:
        device = joints.device
        dfs_joints = joints[order]
        old_parents = parents.detach().cpu().tolist() if isinstance(parents, torch.Tensor) else list(parents)
        dfs_parents_list = [-1 if old_parents[old_idx] == -1 else inverse[old_parents[old_idx]] for old_idx in order]
        dfs_parents = torch.tensor(dfs_parents_list, device=device, dtype=torch.int64)
    else:
        dfs_joints = np.asarray(joints)[order]
        old_parents = list(parents)
        dfs_parents_list = [-1 if old_parents[old_idx] == -1 else inverse[old_parents[old_idx]] for old_idx in order]
        dfs_parents = np.array(dfs_parents_list, dtype=np.int64)

    return dfs_joints, dfs_parents, order, inverse


def compute_ancestor_matrix(
    parents: Union[np.ndarray, torch.Tensor, List[int]],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    r"""Compute the ancestor accumulation matrix A for kinematic tree decoding.

    A[k, i] = 1.0 if joint i is an ancestor of joint k (including i == k), and 0.0 otherwise.
    Given relative DFS offsets \Delta j, the global coordinates are j = A @ \Delta j.
    """
    if isinstance(parents, torch.Tensor):
        parents_list = parents.detach().cpu().tolist()
        if device is None:
            device = parents.device
    else:
        parents_list = list(parents)

    K = len(parents_list)
    A = torch.eye(K, dtype=dtype, device=device)
    for k in range(K):
        p = parents_list[k]
        seen = {k}
        while p >= 0:
            if p in seen:
                raise ValueError(f"Cycle detected in skeleton hierarchy at joint {p}")
            seen.add(p)
            A[k, p] = 1.0
            p = parents_list[p]
    return A


def global_to_relative_joints(
    joints: Union[np.ndarray, torch.Tensor],
    parents: Union[np.ndarray, torch.Tensor, List[int]]
) -> Union[np.ndarray, torch.Tensor]:
    r"""Convert global joint coordinates into relative DFS offsets:
        \Delta j_k = j_k - j_{p(k)} (root j_0 in world coordinates, child joints relative to parent).

    Parameters
    ----------
    joints : array-like of shape (..., K, 3)
        Global joint positions.
    parents : array-like of shape (K,)
        Parent joint indices; root has -1.

    Returns
    -------
    rel_joints : array-like of shape (..., K, 3)
        Relative joint offsets.
    """
    is_torch = isinstance(joints, torch.Tensor)
    if is_torch:
        parents_t = torch.as_tensor(parents, device=joints.device, dtype=torch.int64)
        if parents_t.shape != (joints.shape[-2],):
            raise ValueError(f"Parents length {parents_t.shape} must match joint count {joints.shape[-2]}")
        safe_parents = torch.where(parents_t < 0, torch.arange(len(parents_t), device=joints.device), parents_t)
        parent_coords = joints[..., safe_parents, :]
        is_root = (parents_t < 0).unsqueeze(-1)
        return torch.where(is_root, joints, joints - parent_coords)
    else:
        joints_np = np.asarray(joints, dtype=float)
        parents_np = np.asarray(parents, dtype=np.int64)
        if parents_np.shape != (joints_np.shape[-2],):
            raise ValueError(f"Parents length {parents_np.shape} must match joint count {joints_np.shape[-2]}")
        safe_parents = np.where(parents_np < 0, np.arange(len(parents_np)), parents_np)
        parent_coords = joints_np[..., safe_parents, :]
        is_root = (parents_np < 0)[..., None]
        return np.where(is_root, joints_np, joints_np - parent_coords)


def relative_to_global_joints(
    rel_joints: Union[np.ndarray, torch.Tensor],
    parents: Union[np.ndarray, torch.Tensor, List[int]]
) -> Union[np.ndarray, torch.Tensor]:
    r"""Decode relative DFS joint offsets back to global world coordinates using forward kinematics.

    Parameters
    ----------
    rel_joints : array-like of shape (..., K, 3)
        Relative joint offsets (\Delta j).
    parents : array-like of shape (K,)
        Parent joint indices; root has -1.

    Returns
    -------
    global_joints : array-like of shape (..., K, 3)
        Global joint positions.
    """
    is_torch = isinstance(rel_joints, torch.Tensor)
    if is_torch:
        A = compute_ancestor_matrix(parents, device=rel_joints.device, dtype=rel_joints.dtype)
        # Supports arbitrary leading batch dimensions: (..., K, 3)
        if rel_joints.ndim == 2:
            return torch.matmul(A, rel_joints)
        else:
            return torch.einsum('ij,...jc->...ic', A, rel_joints)
    else:
        rel_np = np.asarray(rel_joints, dtype=float)
        parents_list = list(parents)
        K = len(parents_list)
        A_np = np.eye(K, dtype=float)
        for k in range(K):
            p = parents_list[k]
            while p >= 0:
                A_np[k, p] = 1.0
                p = parents_list[p]
        if rel_np.ndim == 2:
            return A_np @ rel_np
        else:
            return np.einsum('ij,...jc->...ic', A_np, rel_np)


class KinematicTreeDecoder(nn.Module):
    r"""Kinematic tree decoder mapping relative joint offsets \Delta j back to global coordinates.

    Supports fixed or dynamic skeleton hierarchies with fully vectorized and
    differentiable forward kinematics via ancestor accumulation matrix.
    """

    def __init__(self, parents: Optional[Union[torch.Tensor, List[int]]] = None):
        super().__init__()
        self.register_buffer('parents', None if parents is None else torch.as_tensor(parents, dtype=torch.int64))
        self.register_buffer('ancestor_matrix', None if parents is None else compute_ancestor_matrix(parents))

    def set_parents(self, parents: Union[torch.Tensor, List[int], np.ndarray]):
        parents_t = torch.as_tensor(parents, dtype=torch.int64)
        A = compute_ancestor_matrix(parents_t)
        self.parents = parents_t
        self.ancestor_matrix = A

    def forward(
        self,
        rel_joints: torch.Tensor,
        parents: Optional[Union[torch.Tensor, List[int]]] = None
    ) -> torch.Tensor:
        r"""Forward kinematics mapping \Delta j -> j.

        Parameters
        ----------
        rel_joints : torch.Tensor of shape (..., K, 3)
        parents : Optional[torch.Tensor], optional
            If provided, overrides registered skeleton parent structure.

        Returns
        -------
        global_joints : torch.Tensor of shape (..., K, 3)
        """
        if parents is not None:
            A = compute_ancestor_matrix(parents, device=rel_joints.device, dtype=rel_joints.dtype)
        elif self.ancestor_matrix is not None:
            A = self.ancestor_matrix.to(device=rel_joints.device, dtype=rel_joints.dtype)
        else:
            raise ValueError("No parents provided and decoder has no pre-registered skeleton parents")

        if rel_joints.ndim == 2:
            return torch.matmul(A, rel_joints)
        else:
            return torch.einsum('ij,...jc->...ic', A, rel_joints)

    def inverse(
        self,
        global_joints: torch.Tensor,
        parents: Optional[Union[torch.Tensor, List[int]]] = None
    ) -> torch.Tensor:
        r"""Inverse kinematics mapping j -> \Delta j."""
        p = parents if parents is not None else self.parents
        if p is None:
            raise ValueError("No parents provided and decoder has no pre-registered skeleton parents")
        return global_to_relative_joints(global_joints, p)


def containment_loss(
    joints: torch.Tensor,
    mesh_bounds: Optional[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    mesh_vertices: Optional[torch.Tensor] = None,
    margin: float = 0.0,
    loss_type: str = 'l2'
) -> torch.Tensor:
    """Enforce containment inside mesh bounding volume to penalize out-of-mesh floating joints.

    Parameters
    ----------
    joints : torch.Tensor of shape (..., K, 3)
        Global joint positions.
    mesh_bounds : Tuple[torch.Tensor, torch.Tensor] or torch.Tensor, optional
        (min_bound, max_bound) each of shape (..., 3) or tensor of shape (..., 2, 3).
    mesh_vertices : torch.Tensor of shape (..., M, 3), optional
        Mesh vertices used to determine bounding volume dynamically if mesh_bounds not given.
    margin : float, default 0.0
        Margin beyond which joints are penalized.
    loss_type : str, default 'l2'
        'l2' (squared penalty) or 'l1' (absolute penalty).

    Returns
    -------
    loss : torch.Tensor scalar
        Containment penalty. Identically 0.0 when all joints reside strictly within volume.
    """
    if mesh_bounds is None and mesh_vertices is None:
        raise ValueError("Either mesh_bounds or mesh_vertices must be provided to containment_loss")

    if mesh_bounds is not None:
        if isinstance(mesh_bounds, (tuple, list)):
            if len(mesh_bounds) == 6:
                b_min = torch.as_tensor([mesh_bounds[0], mesh_bounds[2], mesh_bounds[4]], device=joints.device, dtype=joints.dtype)
                b_max = torch.as_tensor([mesh_bounds[1], mesh_bounds[3], mesh_bounds[5]], device=joints.device, dtype=joints.dtype)
            else:
                b_min = torch.as_tensor(mesh_bounds[0], device=joints.device, dtype=joints.dtype)
                b_max = torch.as_tensor(mesh_bounds[1], device=joints.device, dtype=joints.dtype)
        else:
            b = torch.as_tensor(mesh_bounds, device=joints.device, dtype=joints.dtype)
            b_min, b_max = b[..., 0, :], b[..., 1, :]
    else:
        # Compute bounds directly from vertices
        b_min = mesh_vertices.min(dim=-2).values
        b_max = mesh_vertices.max(dim=-2).values

    # Expand dimensions for broadcasting with (..., K, 3)
    if b_min.ndim < joints.ndim:
        b_min = b_min.unsqueeze(-2)
        b_max = b_max.unsqueeze(-2)

    under_violation = F.relu(b_min - margin - joints)
    over_violation = F.relu(joints - (b_max + margin))

    if loss_type == 'l2':
        penalty = under_violation.square() + over_violation.square()
    elif loss_type == 'l1':
        penalty = under_violation + over_violation
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    return penalty.sum(dim=-1).mean()


class SkeletonFlowModel(nn.Module):
    r"""Flow matching joint generation model with kinematic tree forward decoding.

    Predicts relative DFS offsets \Delta j conditioned on mesh geometry features,
    integrates velocity field via continuous-time flow, and enforces containment loss.
    """

    def __init__(
        self,
        joint_count: int,
        cond_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 3,
        parents: Optional[Union[torch.Tensor, List[int]]] = None
    ):
        super().__init__()
        self.joint_count = joint_count
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.decoder = KinematicTreeDecoder(parents=parents)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Joint input projection: \Delta j has 3 channels per joint
        self.joint_in = nn.Linear(3, hidden_dim)
        self.cond_in = nn.Linear(cond_dim, hidden_dim)

        # Transformer / MLP layers
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                )
            )
        self.layers = nn.ModuleList(layers)

        # Output velocity / delta prediction
        self.velocity_out = nn.Linear(hidden_dim, 3)

    def _embed_time(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding."""
        if t.ndim == 0:
            t = t.unsqueeze(0)
        half_dim = self.hidden_dim // 2
        freqs = torch.exp(-torch.arange(half_dim, device=t.device, dtype=torch.float32) * (np.log(10000.0) / (half_dim - 1)))
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.time_mlp(embedding)

    def forward(
        self,
        rel_joints: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        r"""Predict flow velocity v = d(\Delta j)/dt.

        Parameters
        ----------
        rel_joints : torch.Tensor of shape (B, K, 3) or (K, 3)
        t : torch.Tensor of shape (B,) or scalar in [0, 1]
        cond : torch.Tensor of shape (B, cond_dim) or (cond_dim,)

        Returns
        -------
        velocity : torch.Tensor of shape matching rel_joints
        """
        unbatched = rel_joints.ndim == 2
        if unbatched:
            rel_joints = rel_joints.unsqueeze(0)
            if t.ndim == 0:
                t = t.unsqueeze(0)
            if cond.ndim == 1:
                cond = cond.unsqueeze(0)
        else:
            if cond.ndim == 1:
                cond = cond.unsqueeze(0)

        # Ensure cond matches batch size
        if cond.ndim == 2 and cond.shape[0] != rel_joints.shape[0]:
            cond = cond.expand(rel_joints.shape[0], -1)

        B, K, _ = rel_joints.shape
        h_joint = self.joint_in(rel_joints)  # (B, K, hidden)
        h_cond = self.cond_in(cond)          # (B, hidden)
        if h_cond.ndim == 2:
            h_cond = h_cond.unsqueeze(1)    # (B, 1, hidden)
        h_time = self._embed_time(t).unsqueeze(1)  # (B, 1, hidden)

        h = h_joint + h_cond + h_time
        for layer in self.layers:
            h = h + layer(h)

        v = self.velocity_out(h)
        if unbatched:
            v = v.squeeze(0)
        return v

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        steps: int = 10,
        parents: Optional[Union[torch.Tensor, List[int]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate relative and global joint positions via ODE integration.

        Returns
        -------
        global_joints, rel_joints
        """
        unbatched = cond.ndim == 1
        if unbatched:
            cond = cond.unsqueeze(0)
        B = cond.shape[0]
        device = cond.device

        # Prior sample at t=0
        x_t = torch.randn(B, self.joint_count, 3, device=device)
        dt = 1.0 / steps

        for i in range(steps):
            t_val = torch.full((B,), i * dt, device=device)
            v = self.forward(x_t, t_val, cond)
            x_t = x_t + v * dt

        global_joints = self.decoder(x_t, parents=parents)
        if unbatched:
            global_joints = global_joints.squeeze(0)
            x_t = x_t.squeeze(0)
        return global_joints, x_t

    def compute_training_loss(
        self,
        gt_global_joints: torch.Tensor,
        cond: torch.Tensor,
        parents: Union[torch.Tensor, List[int]],
        mesh_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        containment_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Compute flow matching loss and mesh containment loss."""
        # Convert GT global coordinates to relative DFS offsets
        gt_rel = self.decoder.inverse(gt_global_joints, parents=parents)

        is_unbatched = gt_rel.ndim == 2
        if is_unbatched:
            gt_rel = gt_rel.unsqueeze(0)
            if cond.ndim == 1:
                cond = cond.unsqueeze(0)
        elif cond.ndim == 1:
            cond = cond.unsqueeze(0)

        B = gt_rel.shape[0]
        device = gt_rel.device
        t = torch.rand(B, device=device)
        noise = torch.randn_like(gt_rel)

        # Rectified flow interpolation
        t_expanded = t[:, None, None]
        x_t = (1.0 - t_expanded) * noise + t_expanded * gt_rel
        target_v = gt_rel - noise

        pred_v = self.forward(x_t, t, cond)
        flow_loss = F.mse_loss(pred_v, target_v)

        # Estimate clean relative offsets: \hat{rel} = x_t + (1 - t) * pred_v
        pred_rel = x_t + (1.0 - t_expanded) * pred_v
        pred_global = self.decoder(pred_rel, parents=parents)

        loss_dict = {'flow_loss': flow_loss}
        if mesh_bounds is not None:
            c_loss = containment_loss(pred_global, mesh_bounds=mesh_bounds)
            loss_dict['containment_loss'] = c_loss
            loss_dict['total_loss'] = flow_loss + containment_weight * c_loss
        else:
            loss_dict['total_loss'] = flow_loss

        return loss_dict
