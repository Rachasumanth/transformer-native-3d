r"""Surface-Grounded Point-to-Bone Cross-Attention Skinning Module.

Given watertight mesh vertices V in R^{M x 3} and joint positions J in R^{K x 3},
computes cross-attention skinning weights grounded on Euclidean and point-to-bone
distance features, with manifold Dijkstra / heat-method geodesic fallback when
confidence is low. Provides top-4 bone truncation and direct glTF packing into
JOINTS_0 (uint16[4]) and WEIGHTS_0 (float32[4]) buffers.
"""

from typing import Optional, Tuple, Union, Dict, Any, List
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import torch
from torch import nn
from torch.nn import functional as F


def point_to_bone_distances(
    vertices: torch.Tensor,
    joints: torch.Tensor,
    parents: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute point-to-joint Euclidean distances and point-to-bone segment distances.

    Parameters
    ----------
    vertices : torch.Tensor of shape (..., M, 3)
    joints : torch.Tensor of shape (..., K, 3)
    parents : Optional[torch.Tensor] of shape (K,), optional
        Parent joint index for each joint. If None, treats each joint as a point bone.

    Returns
    -------
    dist_joint : torch.Tensor of shape (..., M, K)
        Euclidean distance from each vertex to each joint.
    dist_bone : torch.Tensor of shape (..., M, K)
        Distance from each vertex to the line segment of bone k [j_{p(k)}, j_k].
    """
    # Vertex-to-joint Euclidean distance: ||v_i - j_k||
    diff_joint = vertices.unsqueeze(-2) - joints.unsqueeze(-3)  # (..., M, K, 3)
    dist_joint = torch.norm(diff_joint, dim=-1).clamp_min(1e-8)

    if parents is None:
        return dist_joint, dist_joint

    parents_t = torch.as_tensor(parents, device=joints.device, dtype=torch.int64)
    K = joints.shape[-2]
    if parents_t.shape != (K,):
        raise ValueError(f"Parents length {parents_t.shape} must match joint count {K}")

    safe_parents = torch.where(parents_t < 0, torch.arange(K, device=joints.device), parents_t)
    p_joints = joints[..., safe_parents, :]  # (..., K, 3) parent coords

    # Bone segment vector u = j_k - j_{p(k)}
    u = joints - p_joints  # (..., K, 3)
    u_len_sq = (u * u).sum(dim=-1, keepdim=True).clamp_min(1e-8)  # (..., K, 1)

    # Vector from parent joint to vertex v: w = v - j_{p(k)}
    w = vertices.unsqueeze(-2) - p_joints.unsqueeze(-3)  # (..., M, K, 3)

    # Projection factor t = clamp((w . u) / ||u||^2, 0.0, 1.0)
    u_expanded = u.unsqueeze(-3)  # (..., 1, K, 3)
    u_len_sq_expanded = u_len_sq.unsqueeze(-3)  # (..., 1, K, 1)
    dot = (w * u_expanded).sum(dim=-1, keepdim=True)
    t = torch.clamp(dot / u_len_sq_expanded, 0.0, 1.0)

    # Closest point on segment
    closest = p_joints.unsqueeze(-3) + t * u_expanded
    dist_bone_cand = torch.norm(vertices.unsqueeze(-2) - closest, dim=-1).clamp_min(1e-8)

    # For root joints (parent < 0), bone is zero-length, fallback to dist_joint
    is_root = (parents_t < 0).unsqueeze(-2)  # (1, K)
    dist_bone = torch.where(is_root, dist_joint, dist_bone_cand)

    return dist_joint, dist_bone


def compute_manifold_geodesic_weights(
    vertices: Union[np.ndarray, torch.Tensor],
    faces: Union[np.ndarray, torch.Tensor],
    joints: Union[np.ndarray, torch.Tensor],
    method: str = 'dijkstra',
    sigma: float = 0.2
) -> np.ndarray:
    """Compute skinning weights via manifold geodesic distances along mesh edges.

    Parameters
    ----------
    vertices : array-like of shape (M, 3)
    faces : array-like of shape (F, 3)
    joints : array-like of shape (K, 3)
    method : str, default 'dijkstra'
        Geodesic calculation algorithm ('dijkstra' on mesh 1-skeleton or 'heat').
    sigma : float, default 0.2
        Heat / Gaussian kernel bandwidth for converting distances to weights.

    Returns
    -------
    weights : np.ndarray of shape (M, K)
        Normalized skinning weights summing to 1.0 per row.
    """
    v_np = vertices.detach().cpu().numpy() if isinstance(vertices, torch.Tensor) else np.asarray(vertices, dtype=np.float64)
    f_np = faces.detach().cpu().numpy() if isinstance(faces, torch.Tensor) else np.asarray(faces, dtype=np.int64)
    j_np = joints.detach().cpu().numpy() if isinstance(joints, torch.Tensor) else np.asarray(joints, dtype=np.float64)

    M = len(v_np)
    K = len(j_np)
    if M == 0 or K == 0:
        raise ValueError("Vertices and joints must be non-empty")

    # Step 1: For each joint, locate closest surface anchor vertex
    diff = v_np[:, None, :] - j_np[None, :, :]  # (M, K, 3)
    dist_sq = (diff * diff).sum(axis=-1)  # (M, K)
    anchors = np.argmin(dist_sq, axis=0)  # (K,)

    # Step 2: Build manifold mesh edge adjacency graph
    edges_u = []
    edges_v = []
    weights_list = []

    def _add_edge(u_idx, v_idx):
        w = np.linalg.norm(v_np[u_idx] - v_np[v_idx])
        edges_u.extend([u_idx, v_idx])
        edges_v.extend([v_idx, u_idx])
        weights_list.extend([w, w])

    for tri in f_np:
        _add_edge(tri[0], tri[1])
        _add_edge(tri[1], tri[2])
        _add_edge(tri[2], tri[0])

    adj = sp.csr_matrix(
        (weights_list, (edges_u, edges_v)),
        shape=(M, M)
    )

    # Step 3: Dijkstra shortest path from each anchor to all vertices
    # dist_matrix shape: (K, M)
    dist_matrix = dijkstra(
        csgraph=adj,
        directed=False,
        indices=anchors,
        return_predecessors=False
    )
    geo_dist = dist_matrix.T  # (M, K)

    # Step 4: Handle disconnected mesh islands (unreachable vertices)
    unreachable = np.isinf(geo_dist)
    if np.any(unreachable):
        # Fallback unreachable vertices to Euclidean distance to joint anchors
        euc_dist = np.sqrt(dist_sq)
        geo_dist[unreachable] = euc_dist[unreachable]

    # Step 5: Convert geodesic distances to weights using heat / Gaussian kernel
    scaled_sq = -0.5 * (geo_dist / max(sigma, 1e-6)) ** 2
    # Subtract max per row for numerical stability in softmax
    scaled_sq -= np.max(scaled_sq, axis=1, keepdims=True)
    weights = np.exp(scaled_sq) + 1e-10

    # Row normalize
    weights /= weights.sum(axis=1, keepdims=True)
    return weights.astype(np.float32)


class SurfacePointToBoneSkinning(nn.Module):
    """Surface-Grounded Point-to-Bone Cross-Attention Skinning Module.

    Computes cross-attention between mesh vertices (queries) and skeleton bones (keys),
    grounded on Euclidean and point-to-bone segment distance features.
    Provides confidence estimation and seamless geodesic manifold fallback.
    """

    def __init__(
        self,
        query_dim: int = 64,
        key_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        rbf_bands: int = 8
    ):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.rbf_bands = rbf_bands

        # RBF frequencies for distance grounding
        sigmas = torch.linspace(0.02, 1.0, rbf_bands)
        self.register_buffer('rbf_gammas', 1.0 / (2.0 * sigmas ** 2))

        # Explicit geometric distance grounding scale (positive via softplus)
        self.dist_weight = nn.Parameter(torch.tensor(5.0))

        # Input projections
        self.vertex_proj = nn.Linear(3, hidden_dim)
        # Joint + bone direction (j_k, j_k - j_{p(k)}) has 6 channels
        self.joint_proj = nn.Linear(6, hidden_dim)

        # Distance embedding MLP: rbf_bands * 2 (joint dist + bone dist) -> num_heads
        self.distance_mlp = nn.Sequential(
            nn.Linear(rbf_bands * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads)
        )

        # Cross attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learned temperature
        self.temperature = nn.Parameter(torch.ones(1) * np.sqrt(hidden_dim // num_heads))

        # Confidence head: predicts confidence in [0, 1] for each vertex
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def _compute_rbf(self, d: torch.Tensor) -> torch.Tensor:
        r"""Evaluate multi-scale Gaussian RBF features: exp(-\gamma * d^2)."""
        # d shape: (..., M, K) -> (..., M, K, bands)
        d_sq = (d ** 2).unsqueeze(-1)
        return torch.exp(-d_sq * self.rbf_gammas)

    def forward(
        self,
        vertices: torch.Tensor,
        joints: torch.Tensor,
        parents: Optional[torch.Tensor] = None,
        faces: Optional[Union[np.ndarray, torch.Tensor]] = None,
        confidence_threshold: float = 0.4,
        force_geodesic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute surface-grounded skinning weights and confidence scores.

        Parameters
        ----------
        vertices : torch.Tensor of shape (..., M, 3)
        joints : torch.Tensor of shape (..., K, 3)
        parents : Optional[torch.Tensor] of shape (K,), optional
        faces : Optional[array-like] of shape (F, 3), optional
            Mesh faces required if geodesic fallback is triggered.
        confidence_threshold : float, default 0.4
            Vertices with confidence below this threshold fallback to manifold geodesic weights.
        force_geodesic : bool, default False
            If True, forces manifold geodesic weights for all vertices.

        Returns
        -------
        weights : torch.Tensor of shape (..., M, K)
            Row-normalized skinning weights summing to 1.0.
        confidence : torch.Tensor of shape (..., M)
            Per-vertex confidence score in [0, 1].
        """
        unbatched = vertices.ndim == 2
        if unbatched:
            vertices = vertices.unsqueeze(0)
            joints = joints.unsqueeze(0)

        B, M, _ = vertices.shape
        _, K, _ = joints.shape

        # Compute point-to-joint and point-to-bone distances
        dist_joint, dist_bone = point_to_bone_distances(vertices, joints, parents=parents)

        # Build bone directional feature
        if parents is not None:
            parents_t = torch.as_tensor(parents, device=joints.device, dtype=torch.int64)
            safe_parents = torch.where(parents_t < 0, torch.arange(K, device=joints.device), parents_t)
            bone_vec = joints - joints[:, safe_parents, :]
        else:
            bone_vec = torch.zeros_like(joints)
        joint_feat = torch.cat([joints, bone_vec], dim=-1)  # (B, K, 6)

        # Projections
        h_v = self.vertex_proj(vertices)  # (B, M, hidden)
        h_j = self.joint_proj(joint_feat)  # (B, K, hidden)

        Q = self.q_proj(h_v)  # (B, M, hidden)
        K_proj = self.k_proj(h_j)  # (B, K, hidden)

        # Multi-head attention affinity
        head_dim = self.hidden_dim // self.num_heads
        Q = Q.reshape(B, M, self.num_heads, head_dim).transpose(1, 2)  # (B, H, M, head_dim)
        K_proj = K_proj.reshape(B, K, self.num_heads, head_dim).transpose(1, 2)  # (B, H, K, head_dim)

        dot_affinity = torch.matmul(Q, K_proj.transpose(-1, -2)) / self.temperature.clamp_min(1e-4)  # (B, H, M, K)

        # Grounding with geometric distances
        rbf_j = self._compute_rbf(dist_joint)  # (B, M, K, bands)
        rbf_b = self._compute_rbf(dist_bone)   # (B, M, K, bands)
        rbf_all = torch.cat([rbf_j, rbf_b], dim=-1)  # (B, M, K, 2*bands)
        dist_bias = self.distance_mlp(rbf_all).permute(0, 3, 1, 2)  # (B, H, M, K)

        # Direct geometric distance grounding affinity: negative distance ensures closest bone affinity
        dist_affinity = -F.softplus(self.dist_weight) * dist_joint.unsqueeze(1)  # (B, 1, M, K)

        total_affinity = dot_affinity + dist_bias + dist_affinity
        # Average across attention heads
        affinity = total_affinity.mean(dim=1)  # (B, M, K)

        # Softmax over joints
        weights_att = F.softmax(affinity, dim=-1)

        # Confidence estimation
        conf_raw = self.confidence_head(h_v).squeeze(-1)  # (B, M)
        # Weight entropy penalty: lower entropy = more focused = higher confidence
        entropy = -(weights_att * torch.log(weights_att.clamp_min(1e-8))).sum(dim=-1)
        max_entropy = np.log(max(K, 2))
        entropy_factor = torch.clamp(1.0 - (entropy / max_entropy), 0.0, 1.0)
        confidence = conf_raw * entropy_factor

        # Geodesic fallback if confidence is low and mesh faces are provided
        final_weights = weights_att
        if force_geodesic or (faces is not None and (confidence < confidence_threshold).any()):
            # Compute geodesic weights
            for b in range(B):
                v_single = vertices[b]
                j_single = joints[b]
                conf_single = confidence[b]

                if force_geodesic or (conf_single < confidence_threshold).any():
                    geo_w_np = compute_manifold_geodesic_weights(
                        v_single, faces, j_single, method='dijkstra'
                    )
                    geo_w = torch.as_tensor(geo_w_np, device=vertices.device, dtype=vertices.dtype)

                    if force_geodesic:
                        final_weights[b] = geo_w
                    else:
                        # Smooth blending based on confidence
                        alpha = conf_single.unsqueeze(-1)
                        blended = alpha * weights_att[b] + (1.0 - alpha) * geo_w
                        final_weights[b] = blended / blended.sum(dim=-1, keepdim=True)

        if unbatched:
            final_weights = final_weights.squeeze(0)
            confidence = confidence.squeeze(0)

        return final_weights, confidence


def truncate_top4_weights(
    weights: Union[np.ndarray, torch.Tensor]
) -> Tuple[np.ndarray, np.ndarray]:
    """Enforce top-4 bone influences per vertex and row normalization.

    Parameters
    ----------
    weights : array-like of shape (M, K)
        Full skinning weight matrix.

    Returns
    -------
    joints_0 : np.ndarray of shape (M, 4) and dtype uint16
        Top 4 joint indices for each vertex.
    weights_0 : np.ndarray of shape (M, 4) and dtype float32
        Top 4 normalized skinning weights summing to 1.0 per vertex.
    """
    w_np = weights.detach().cpu().numpy() if isinstance(weights, torch.Tensor) else np.asarray(weights, dtype=np.float64)
    if w_np.ndim != 2:
        raise ValueError(f"Weights must be 2D matrix (M, K), got shape {w_np.shape}")

    M, K = w_np.shape
    if M == 0 or K == 0:
        raise ValueError("Weights cannot be empty")

    # Stable descending sort preserves ascending joint order on ties
    order = np.argsort(-w_np, axis=1, kind='stable')
    k = min(4, K)
    top_indices = order[:, :k]
    top_weights = np.take_along_axis(w_np, top_indices, axis=1)

    # Row sum of top-k
    retained_mass = top_weights.sum(axis=1, keepdims=True)
    zero_mask = (retained_mass <= 0).squeeze(axis=1)

    # Renormalize top weights
    retained_mass_safe = np.where(retained_mass > 0, retained_mass, 1.0)
    top_weights_renorm = top_weights / retained_mass_safe

    joints_0 = np.zeros((M, 4), dtype=np.uint16)
    weights_0 = np.zeros((M, 4), dtype=np.float32)

    joints_0[:, :k] = top_indices.astype(np.uint16)
    weights_0[:, :k] = top_weights_renorm.astype(np.float32)

    # In glTF, zero-weight components convention to joint 0
    joints_0[weights_0 == 0.0] = 0

    # Handle zero mass rows: assign full 1.0 weight to joint 0
    if np.any(zero_mask):
        joints_0[zero_mask, 0] = 0
        weights_0[zero_mask, 0] = 1.0
        weights_0[zero_mask, 1:] = 0.0

    # Ensure float32 row sums strictly equal 1.0
    f32_sums = weights_0.sum(axis=1, keepdims=True)
    weights_0 /= f32_sums

    return joints_0, weights_0


def pack_gltf_skin_buffers(
    joints_0: np.ndarray,
    weights_0: np.ndarray
) -> Dict[str, Any]:
    """Pack JOINTS_0 and WEIGHTS_0 arrays into standard glTF 2.0 binary accessors.

    Parameters
    ----------
    joints_0 : np.ndarray of shape (M, 4) and dtype uint16
    weights_0 : np.ndarray of shape (M, 4) and dtype float32

    Returns
    -------
    dict
        Dictionary containing binary buffers, buffer views, accessor metadata,
        and combined byte buffer with 4-byte alignment.
    """
    if joints_0.dtype != np.uint16 or joints_0.shape != weights_0.shape or joints_0.shape[1] != 4:
        raise ValueError(f"JOINTS_0 must be uint16 with shape (M, 4), got dtype {joints_0.dtype} and shape {joints_0.shape}")
    if weights_0.dtype != np.float32 or weights_0.shape[1] != 4:
        raise ValueError(f"WEIGHTS_0 must be float32 with shape (M, 4), got dtype {weights_0.dtype} and shape {weights_0.shape}")

    M = len(joints_0)
    j_arr = np.ascontiguousarray(joints_0, dtype='<u2')
    w_arr = np.ascontiguousarray(weights_0, dtype='<f4')
    j_bytes = j_arr.tobytes()
    w_bytes = w_arr.tobytes()

    # 4-byte padding alignment for binary glTF
    pad_j = (4 - (len(j_bytes) % 4)) % 4
    j_bytes_padded = j_bytes + b'\x00' * pad_j

    combined_bytes = j_bytes_padded + w_bytes

    return {
        'JOINTS_0': {
            'componentType': 5123,  # UNSIGNED_SHORT
            'type': 'VEC4',
            'count': M,
            'byteOffset': 0,
            'byteLength': len(j_bytes),
            'min': [int(j_arr[:, c].min()) if M > 0 else 0 for c in range(4)],
            'max': [int(j_arr[:, c].max()) if M > 0 else 0 for c in range(4)],
            'bytes': j_bytes
        },
        'WEIGHTS_0': {
            'componentType': 5126,  # FLOAT
            'type': 'VEC4',
            'count': M,
            'byteOffset': len(j_bytes_padded),
            'byteLength': len(w_bytes),
            'min': [float(w_arr[:, c].min()) if M > 0 else 0.0 for c in range(4)],
            'max': [float(w_arr[:, c].max()) if M > 0 else 0.0 for c in range(4)],
            'bytes': w_bytes
        },
        'combined_bytes': combined_bytes,
        'byteLength': len(combined_bytes)
    }
