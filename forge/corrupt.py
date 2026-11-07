"""Self-supervised corruption of clean meshes into raw inputs.

Severity levels emulate real raw assets: decimated game meshes, scans with
holes, badly exported faces, noisy vertices, and dirty topology.
"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .geometry import decimate_mesh


def jitter(mesh, rng, sigma):
    mesh = mesh.copy()
    mesh.vertices = mesh.vertices + rng.normal(0.0, sigma, size=mesh.vertices.shape)
    return mesh


def drop_faces(mesh, rng, fraction):
    count = int(round(len(mesh.faces) * fraction))
    if count <= 0:
        return mesh
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[rng.choice(len(mesh.faces), size=count, replace=False)] = False
    mesh = mesh.copy()
    mesh.update_faces(keep)
    return mesh


def punch_holes(mesh, rng, n_holes, radius):
    if n_holes <= 0 or len(mesh.faces) < 32:
        return mesh
    centroids = mesh.triangles_center
    seeds = centroids[rng.choice(len(centroids), size=min(n_holes, len(centroids)), replace=False)]
    tree = cKDTree(centroids)
    remove = np.zeros(len(mesh.faces), dtype=bool)
    for seed in seeds:
        remove[tree.query_ball_point(seed, radius)] = True
    if remove.mean() > 0.75:
        return mesh
    mesh = mesh.copy()
    mesh.update_faces(~remove)
    return mesh


def flip_faces(mesh, rng, fraction):
    count = int(round(len(mesh.faces) * fraction))
    if count <= 0:
        return mesh
    mesh = mesh.copy()
    selected = rng.choice(len(mesh.faces), size=count, replace=False)
    mesh.faces[selected] = mesh.faces[selected][:, ::-1]
    return mesh


def duplicate_faces(mesh, rng, count):
    if count <= 0 or not len(mesh.faces):
        return mesh
    selected = rng.choice(len(mesh.faces), size=min(count, len(mesh.faces)), replace=False)
    mesh = mesh.copy()
    mesh.faces = np.concatenate([mesh.faces, mesh.faces[selected][:, ::-1]], axis=0)
    return mesh


def corrupt_mesh(clean, rng, severity=None):
    """Return a damaged copy of a normalized clean mesh plus a recipe record."""
    severity = severity if severity is not None else rng.choice(['mild', 'medium', 'heavy'])
    mesh = clean.copy()
    recipe = {'severity': str(severity), 'ops': []}
    if severity == 'identity':
        recipe['ops'].append('identity')
        return mesh, recipe
    if severity == 'mild':
        decimate_ratio, drop_ratio = 0.5, 0.03
        holes, hole_radius = int(rng.integers(1, 3)), 0.01
        sigma, flip_ratio, dup = 0.0015, 0.0, 0
    elif severity == 'medium':
        decimate_ratio, drop_ratio = 0.25, 0.10
        holes, hole_radius = int(rng.integers(2, 6)), 0.018
        sigma, flip_ratio, dup = 0.004, 0.01, int(rng.integers(0, 40))
    else:
        decimate_ratio, drop_ratio = 0.12, 0.22
        holes, hole_radius = int(rng.integers(4, 9)), 0.028
        sigma, flip_ratio, dup = 0.008, 0.03, int(rng.integers(20, 120))
    target = max(16, int(len(mesh.faces) * decimate_ratio))
    if target < len(mesh.faces):
        mesh = decimate_mesh(mesh, target)
        recipe['ops'].append(f'decimate:{target}')
    if sigma > 0:
        mesh = jitter(mesh, rng, sigma)
        recipe['ops'].append(f'jitter:{sigma}')
    if drop_ratio > 0:
        mesh = drop_faces(mesh, rng, drop_ratio)
        recipe['ops'].append(f'drop_faces:{drop_ratio}')
    if holes > 0:
        mesh = punch_holes(mesh, rng, holes, hole_radius)
        recipe['ops'].append(f'holes:{holes}@{hole_radius}')
    if flip_ratio > 0:
        mesh = flip_faces(mesh, rng, flip_ratio)
        recipe['ops'].append(f'flip_faces:{flip_ratio}')
    if dup > 0:
        mesh = duplicate_faces(mesh, rng, dup)
        recipe['ops'].append(f'duplicate_faces:{dup}')
    if len(mesh.faces) < 12 and len(mesh.faces):
        need = 12 - len(mesh.faces)
        picks = mesh.faces[rng.integers(0, len(mesh.faces), size=need)]
        mesh = trimesh.Trimesh(vertices=mesh.vertices,
                               faces=np.concatenate([mesh.faces, picks], axis=0),
                               process=False)
        recipe['ops'].append('face-pad')
    return mesh, recipe


def degrade_colors(colors, rng, mask=None):
    """Degrade sampled sRGB colors to emulate a poor/damaged input texture."""
    values = np.array(colors, dtype=np.float64, copy=True)
    mask = np.ones(len(values), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not mask.any():
        return values
    mode = rng.choice(['none', 'tint', 'dropout', 'blur', 'desaturate', 'heavy'])
    if mode == 'none':
        return values
    selected = values[mask]
    if mode == 'tint':
        tint = rng.uniform(0.7, 1.15, size=3)
        selected[:, :3] *= tint
    elif mode == 'dropout':
        fraction = rng.uniform(0.15, 0.45)
        chosen = rng.random(len(selected)) < fraction
        selected[chosen, :3] = rng.uniform(0.35, 0.75, size=(int(chosen.sum()), 3))
    elif mode == 'blur':
        spatial = rng.normal(0.0, 0.35, size=3)
        selected[:, :3] = selected[:, :3] * 0.7 + 0.3 * np.clip(spatial, 0, 1)
    elif mode == 'desaturate':
        gray = selected[:, :3].mean(axis=1, keepdims=True)
        selected[:, :3] = 0.35 * selected[:, :3] + 0.65 * gray
    else:
        selected[:, :3] = rng.uniform(0.25, 0.9, size=(len(selected), 3))
        selected[:, 3:5] *= rng.uniform(0.0, 0.4)
    values[mask] = selected
    return np.clip(values, 0.0, 1.0)
