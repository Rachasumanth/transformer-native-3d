"""SC-VAE latent & Hermite target pre-encoder for refinement models.

Encodes clean/corrupted mesh pairs into continuous shape latents, coarse occupancy,
surface Hermite normals, and surface ratio loss weights.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn

from ...schema import VoxelAsset
from ...voxelize import mesh_to_voxels
from ...dataset import batch_arrays
from ...sc_vae import SparseCompressionVAE
from ...proximity import TriangleBVH
from ...sampling import _mesh


_DEFAULT_VAE = None


def get_default_vae(device='cpu', latent_channels=16):
    """Retrieve or lazily initialize the default reference SC-VAE model."""
    global _DEFAULT_VAE
    if _DEFAULT_VAE is None:
        model = SparseCompressionVAE(
            mode='shape',
            channels=(8, 16, 32, 64, 128),
            latent_channels=latent_channels,
            convolution_backend='reference',
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        _DEFAULT_VAE = model
    return _DEFAULT_VAE.to(device)


def load_mesh_or_asset(mesh_input):
    """Normalize mesh input into (vertices, faces) or a VoxelAsset.

    Supports:
    - (vertices, faces) tuple/list
    - trimesh.Trimesh object
    - dict with 'vertices' and 'faces'
    - .t3d, .glb, .obj, .ply file paths
    - VoxelAsset instance
    """
    if isinstance(mesh_input, VoxelAsset):
        return mesh_input

    if isinstance(mesh_input, (str, Path)):
        p = Path(mesh_input)
        if not p.exists():
            raise FileNotFoundError(f"Mesh file not found: {p}")
        if p.suffix.lower() == '.t3d':
            from ...container import loads
            container = loads(p.read_bytes())
            return container.asset
        import trimesh
        scene_or_mesh = trimesh.load(str(p), force='mesh')
        return np.asarray(scene_or_mesh.vertices, dtype=np.float64), np.asarray(scene_or_mesh.faces, dtype=np.int64)

    if hasattr(mesh_input, 'vertices') and hasattr(mesh_input, 'faces'):
        return np.asarray(mesh_input.vertices, dtype=np.float64), np.asarray(mesh_input.faces, dtype=np.int64)

    if isinstance(mesh_input, dict) and 'vertices' in mesh_input and 'faces' in mesh_input:
        return np.asarray(mesh_input['vertices'], dtype=np.float64), np.asarray(mesh_input['faces'], dtype=np.int64)

    if isinstance(mesh_input, (tuple, list)) and len(mesh_input) == 2:
        return np.asarray(mesh_input[0], dtype=np.float64), np.asarray(mesh_input[1], dtype=np.int64)

    raise TypeError(f"Unsupported mesh input type: {type(mesh_input)}")


def synthesize_corrupted_mesh(vertices, faces, noise_std=0.03, dropout_ratio=0.1, rng=None):
    """Synthesize a corrupted version of a mesh by adding vertex jitter and face dropout."""
    if rng is None:
        rng = np.random.default_rng(42)
    v = np.asarray(vertices, dtype=np.float64).copy()
    f = np.asarray(faces, dtype=np.int64).copy()

    extent = np.maximum(v.max(axis=0) - v.min(axis=0), 1e-4)
    scale = np.max(extent)
    jitter = rng.normal(0, noise_std * scale, size=v.shape)
    v = v + jitter

    if dropout_ratio > 0 and len(f) > 10:
        keep = rng.uniform(size=len(f)) > dropout_ratio
        if np.any(keep):
            f = f[keep]

    return v, f


def determine_aabb(clean_v, raw_v=None, margin=0.02):
    """Determine a safe enclosing AABB for voxelization."""
    clean_v = np.asarray(clean_v, dtype=np.float64)
    if raw_v is not None:
        raw_v = np.asarray(raw_v, dtype=np.float64)
        all_v = np.vstack([clean_v, raw_v])
    else:
        all_v = clean_v

    # Common canonical bounds
    if (all_v >= 0.001).all() and (all_v <= 0.999).all():
        return np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    if (all_v >= -0.499).all() and (all_v <= 0.499).all():
        return np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=np.float64)

    lo = all_v.min(axis=0)
    hi = all_v.max(axis=0)
    extent = np.maximum(hi - lo, 1e-4)
    pad = extent * margin + 1e-4
    return np.array([lo - pad, hi + pad], dtype=np.float64)


def extract_voxel_hermite_normals(asset, vertices=None, faces=None, coarse_res=4, fine_res=32):
    """Extract Hermite surface normals at voxel dual intersections and aggregate to coarse cells."""
    N = coarse_res ** 3
    if len(asset.coords) == 0:
        return np.zeros((N, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    if vertices is not None and faces is not None and len(faces) > 0:
        v, f, face_normals, _ = _mesh(vertices, faces)
        bvh = TriangleBVH(v, f)
        closest_face_ids, _ = bvh.query(asset.world_vertices)
        fine_normals = face_normals[closest_face_ids].astype(np.float32)
    else:
        from ...extraction import voxels_to_mesh
        try:
            v_ext, f_ext = voxels_to_mesh(asset)
            if len(f_ext) > 0:
                v_ext, f_ext, face_normals, _ = _mesh(v_ext, f_ext)
                bvh = TriangleBVH(v_ext, f_ext)
                closest_face_ids, _ = bvh.query(asset.world_vertices)
                fine_normals = face_normals[closest_face_ids].astype(np.float32)
            else:
                fine_normals = np.zeros((len(asset.coords), 3), dtype=np.float32)
        except Exception:
            fine_normals = np.zeros((len(asset.coords), 3), dtype=np.float32)

    f_norms = np.linalg.norm(fine_normals, axis=-1, keepdims=True)
    fine_normals = np.where(f_norms > 1e-6, fine_normals / np.maximum(f_norms, 1e-6), 0.0).astype(np.float32)

    scale = fine_res // coarse_res
    coarse_coords = asset.coords // scale
    coarse_indices = (coarse_coords[:, 0] * coarse_res + coarse_coords[:, 1]) * coarse_res + coarse_coords[:, 2]

    hermite_normals = np.zeros((N, 3), dtype=np.float32)
    counts = np.bincount(coarse_indices, minlength=N)

    np.add.at(hermite_normals, coarse_indices, fine_normals)

    active_mask = (counts > 0)
    norms = np.linalg.norm(hermite_normals, axis=-1, keepdims=True)

    # For cells where opposing normals cancelled out to zero, take the first incident fine normal
    cancelled = np.where(active_mask & (norms.squeeze(-1) <= 1e-6))[0]
    for cell_idx in cancelled:
        first_fine_idx = np.where(coarse_indices == cell_idx)[0][0]
        hermite_normals[cell_idx] = fine_normals[first_fine_idx]

    norms = np.linalg.norm(hermite_normals, axis=-1, keepdims=True)
    hermite_normals = np.where(
        active_mask[:, None] & (norms > 1e-6),
        hermite_normals / np.maximum(norms, 1e-6),
        0.0
    ).astype(np.float32)

    return hermite_normals, fine_normals


def map_vae_latents_to_coarse_cells(asset, vae_coords, vae_latents, coarse_res, fine_res, N, C):
    """Map SC-VAE bottleneck latents onto the coarse grid cells."""
    x_0_lat = np.zeros((N, C), dtype=np.float32)
    if len(asset.coords) == 0 or len(vae_coords) == 0:
        return x_0_lat

    scale = fine_res // coarse_res
    coarse_coords = asset.coords // scale
    coarse_indices = (coarse_coords[:, 0] * coarse_res + coarse_coords[:, 1]) * coarse_res + coarse_coords[:, 2]

    vae_dict = {tuple(int(x) for x in c): vae_latents[k] for k, c in enumerate(vae_coords)}
    fine_vae_coords = asset.coords // 16

    active_cells = np.unique(coarse_indices)
    for c_idx in active_cells:
        mask = (coarse_indices == c_idx)
        cell_v_coords = fine_vae_coords[mask]
        u_v_coords = np.unique(cell_v_coords, axis=0)
        lats = [vae_dict[tuple(int(x) for x in c)] for c in u_v_coords if tuple(int(x) for x in c) in vae_dict]
        if lats:
            x_0_lat[c_idx] = np.mean(lats, axis=0)

    return x_0_lat


def encode_paired_asset(
    clean_mesh,
    raw_mesh=None,
    sc_vae_model=None,
    coarse_res=4,
    fine_res=32,
    aabb=None,
    device=None,
    return_torch=True,
    output_npz_path=None,
) -> dict:
    """Pre-encode a clean and raw/corrupted asset pair into refinement training targets.

    Computes:
      - x_0_occ: Ground-truth coarse occupancy ({-1, +1}) on the full coarse grid (shape N).
      - x_0_lat: Ground-truth continuous SC-VAE latents on active cells (0 on empty cells, shape N x C).
      - c_raw: Coarse occupancy + continuous latents of corrupted input M_raw (shape N x (C + 1)).
      - hermite_normals: Ground-truth voxel surface normals for Dual Contouring (shape N x 3).
      - surface_ratio: Fraction of surface-intersecting voxels per coarse cell for loss weighting (shape N).

    Args:
      clean_mesh: Clean mesh input (tuple (V, F), Trimesh, file path, dict, or VoxelAsset).
      raw_mesh: Corrupted/raw mesh input; if None, synthesized from clean_mesh.
      sc_vae_model: Frozen SparseCompressionVAE instance; defaults to reference shape VAE.
      coarse_res: Coarse grid resolution (e.g. 4 or 8), yielding N = coarse_res^3 cells.
      fine_res: Fine voxelization grid resolution (e.g. 32 or 64).
      aabb: Explicit bounding box [[min_x, min_y, min_z], [max_x, max_y, max_z]].
      device: PyTorch device to execute the encoder on.
      return_torch: If True, return torch.Tensors; if False, return numpy.ndarrays.
      output_npz_path: Optional path to save compressed .npz archive.

    Returns:
      Dictionary containing x_0_occ, x_0_lat, c_raw, hermite_normals, surface_ratio.
    """
    if fine_res % coarse_res != 0:
        raise ValueError(f"fine_res ({fine_res}) must be an integer multiple of coarse_res ({coarse_res})")

    N = coarse_res ** 3
    scale = fine_res // coarse_res
    voxels_per_coarse = scale ** 3

    # Resolve device and VAE model
    if sc_vae_model is None:
        target_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        sc_vae_model = get_default_vae(device=target_device)
    else:
        target_device = device or next(sc_vae_model.parameters()).device
        sc_vae_model = sc_vae_model.to(target_device).eval()

    latent_channels = sc_vae_model.moments.out_features // 2

    # Load clean input
    clean_obj = load_mesh_or_asset(clean_mesh)
    clean_vertices = clean_faces = None
    if isinstance(clean_obj, tuple):
        clean_vertices, clean_faces = clean_obj

    # Load or synthesize raw input
    if raw_mesh is None:
        if clean_vertices is not None:
            raw_vertices, raw_faces = synthesize_corrupted_mesh(clean_vertices, clean_faces)
            raw_obj = (raw_vertices, raw_faces)
        else:
            raw_obj = clean_obj
    else:
        raw_obj = load_mesh_or_asset(raw_mesh)

    raw_vertices = raw_faces = None
    if isinstance(raw_obj, tuple):
        raw_vertices, raw_faces = raw_obj

    # Determine AABB
    if aabb is None:
        if clean_vertices is not None:
            aabb = determine_aabb(clean_vertices, raw_vertices)
        elif isinstance(clean_obj, VoxelAsset):
            aabb = clean_obj.aabb
        else:
            aabb = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=np.float64)
    else:
        aabb = np.asarray(aabb, dtype=np.float64)

    # Voxelize clean asset
    if isinstance(clean_obj, VoxelAsset):
        clean_asset = clean_obj
    else:
        clean_asset = mesh_to_voxels(clean_vertices, clean_faces, grid_size=(fine_res, fine_res, fine_res), aabb=aabb)

    # Voxelize raw asset
    if isinstance(raw_obj, VoxelAsset):
        raw_asset = raw_obj
    else:
        raw_asset = mesh_to_voxels(raw_vertices, raw_faces, grid_size=(fine_res, fine_res, fine_res), aabb=aabb)

    # 1. Surface ratio and x_0_occ
    if len(clean_asset.coords) > 0:
        c_coords = clean_asset.coords // scale
        c_indices = (c_coords[:, 0] * coarse_res + c_coords[:, 1]) * coarse_res + c_coords[:, 2]
        clean_counts = np.bincount(c_indices, minlength=N)
    else:
        clean_counts = np.zeros(N, dtype=np.int64)

    surface_ratio = (clean_counts.astype(np.float32) / float(voxels_per_coarse)).clip(0.0, 1.0)
    x_0_occ = np.where(clean_counts > 0, 1.0, -1.0).astype(np.float32)

    # 2. Extract Hermite surface normals
    hermite_normals, _ = extract_voxel_hermite_normals(
        clean_asset,
        vertices=clean_vertices,
        faces=clean_faces,
        coarse_res=coarse_res,
        fine_res=fine_res,
    )

    # 3. Ground-truth continuous latents x_0_lat
    if len(clean_asset.coords) > 0:
        c_arr, f_arr = batch_arrays([clean_asset], 'shape')
        c_t = torch.from_numpy(c_arr).to(target_device)
        f_t = torch.from_numpy(f_arr).to(target_device)
        with torch.no_grad():
            post = sc_vae_model.encode(c_t, f_t)
            vae_coords = post.coords[:, 1:].detach().cpu().numpy()
            vae_latents = post.mean.detach().cpu().numpy()
        x_0_lat = map_vae_latents_to_coarse_cells(
            clean_asset, vae_coords, vae_latents, coarse_res, fine_res, N, latent_channels
        )
    else:
        x_0_lat = np.zeros((N, latent_channels), dtype=np.float32)

    # 4. Corrupted conditioning latents c_raw
    if len(raw_asset.coords) > 0:
        raw_c_coords = raw_asset.coords // scale
        raw_c_indices = (raw_c_coords[:, 0] * coarse_res + raw_c_coords[:, 1]) * coarse_res + raw_c_coords[:, 2]
        raw_counts = np.bincount(raw_c_indices, minlength=N)
        raw_occ = np.where(raw_counts > 0, 1.0, -1.0).astype(np.float32)

        rc_arr, rf_arr = batch_arrays([raw_asset], 'shape')
        rc_t = torch.from_numpy(rc_arr).to(target_device)
        rf_t = torch.from_numpy(rf_arr).to(target_device)
        with torch.no_grad():
            raw_post = sc_vae_model.encode(rc_t, rf_t)
            raw_vae_coords = raw_post.coords[:, 1:].detach().cpu().numpy()
            raw_vae_latents = raw_post.mean.detach().cpu().numpy()
        raw_lat = map_vae_latents_to_coarse_cells(
            raw_asset, raw_vae_coords, raw_vae_latents, coarse_res, fine_res, N, latent_channels
        )
    else:
        raw_occ = np.full((N,), -1.0, dtype=np.float32)
        raw_lat = np.zeros((N, latent_channels), dtype=np.float32)

    c_raw = np.concatenate([raw_occ[:, None], raw_lat], axis=-1).astype(np.float32)

    # Save to NPZ if requested
    if output_npz_path is not None:
        out_p = Path(output_npz_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_p,
            x_0_occ=x_0_occ,
            x_0_lat=x_0_lat,
            c_raw=c_raw,
            hermite_normals=hermite_normals,
            surface_ratio=surface_ratio,
        )

    if return_torch:
        return {
            'x_0_occ': torch.from_numpy(x_0_occ).to(target_device),
            'x_0_lat': torch.from_numpy(x_0_lat).to(target_device),
            'c_raw': torch.from_numpy(c_raw).to(target_device),
            'hermite_normals': torch.from_numpy(hermite_normals).to(target_device),
            'surface_ratio': torch.from_numpy(surface_ratio).to(target_device),
        }

    return {
        'x_0_occ': x_0_occ,
        'x_0_lat': x_0_lat,
        'c_raw': c_raw,
        'hermite_normals': hermite_normals,
        'surface_ratio': surface_ratio,
    }


def encode_micro_batch(
    asset_pairs,
    output_npz_path,
    sc_vae_model=None,
    coarse_res=4,
    fine_res=32,
    device=None,
    return_torch=False,
) -> dict:
    """Encode a micro-batch of asset pairs and archive the stacked tensors into an .npz file.

    Args:
      asset_pairs: List of (clean_mesh, raw_mesh) pairs.
      output_npz_path: Destination path for the compressed .npz archive.
      sc_vae_model: Frozen SC-VAE encoder instance.
      coarse_res: Coarse resolution (e.g. 4).
      fine_res: Fine resolution (e.g. 32).
      device: PyTorch device.
      return_torch: If True, return torch.Tensors; else return numpy.ndarrays.

    Returns:
      Batched dictionary with shapes (B, N), (B, N, C), (B, N, C + 1), etc.
    """
    if not asset_pairs:
        raise ValueError("asset_pairs must not be empty")

    encoded_list = []
    for clean_mesh, raw_mesh in asset_pairs:
        enc = encode_paired_asset(
            clean_mesh=clean_mesh,
            raw_mesh=raw_mesh,
            sc_vae_model=sc_vae_model,
            coarse_res=coarse_res,
            fine_res=fine_res,
            device=device,
            return_torch=False,
        )
        encoded_list.append(enc)

    batch_dict = {
        'x_0_occ': np.stack([item['x_0_occ'] for item in encoded_list], axis=0),
        'x_0_lat': np.stack([item['x_0_lat'] for item in encoded_list], axis=0),
        'c_raw': np.stack([item['c_raw'] for item in encoded_list], axis=0),
        'hermite_normals': np.stack([item['hermite_normals'] for item in encoded_list], axis=0),
        'surface_ratio': np.stack([item['surface_ratio'] for item in encoded_list], axis=0),
    }

    out_p = Path(output_npz_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_p, **batch_dict)

    if return_torch:
        target_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        return {k: torch.from_numpy(v).to(target_device) for k, v in batch_dict.items()}

    return batch_dict
