import unittest
from collections import Counter
import numpy as np
from t3d.schema import VoxelAsset
from t3d.voxelize import mesh_to_voxels
from t3d.extraction import voxels_to_mesh, resolve_nonmanifold_faces
from t3d.topology import orient_faces


class ExtractionTests(unittest.TestCase):
    def quad(self, axis=0, weights=None, missing=False):
        coords = np.zeros((4, 3), dtype=int)
        a, b = [i for i in range(3) if i != axis]
        coords[:, a] = [0, 0, 1, 1]
        coords[:, b] = [0, 1, 1, 0]
        flags = np.zeros((4, 3), bool)
        flags[0, axis] = True
        if missing:
            coords, flags = coords[:3], flags[:3]
        return VoxelAsset(coords, np.full(coords.shape, .5), flags,
                          (2,)*3, [[0]*3, [2]*3], split_weight=weights)

    def test_each_axis_and_missing_neighbor(self):
        for axis in range(3):
            v, f = voxels_to_mesh(self.quad(axis))
            self.assertEqual(f.shape, (2, 3))
            np.testing.assert_allclose(v[:, axis], .5)
            self.assertEqual(set(f.ravel()), set(range(4)))
        self.assertEqual(voxels_to_mesh(self.quad(missing=True))[1].shape, (0, 3))

    def test_weighted_diagonal(self):
        for weights, diagonal in (([2, 1, 2, 1], {0, 2}), ([1, 2, 1, 2], {1, 3})):
            _, faces = voxels_to_mesh(self.quad(weights=np.array(weights)[:, None]))
            self.assertEqual(set(faces[0]) & set(faces[1]), diagonal)

    def test_open_sheet_roundtrip(self):
        v = [[.4, 0, 0], [.4, 1, 0], [.4, 1, 1], [.4, 0, 1]]
        a = mesh_to_voxels(v, [[0, 1, 2], [0, 2, 3]], grid_size=(4,)*3,
                           aabb=[[0]*3, [1]*3], regularization_weight=0)
        vertices, faces = voxels_to_mesh(a)
        self.assertEqual(faces.shape, (18, 3))
        np.testing.assert_allclose(vertices[:, 0], .4, atol=1e-12)
        cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                         vertices[faces[:, 2]] - vertices[faces[:, 0]])
        self.assertTrue(np.all(np.linalg.norm(cross, axis=1) > 0))
        # Boundary QEF constraints retain the full extent of this aligned sheet.
        self.assertAlmostEqual(np.linalg.norm(cross, axis=1).sum() / 2, 1)

    def test_closed_cube_roundtrip_topology(self):
        v = np.array([[x, y, z] for x in (.2, .8) for y in (.2, .8) for z in (.2, .8)])
        faces = []
        for axis in range(3):
            for side in (.2, .8):
                ids = np.flatnonzero(v[:, axis] == side)
                a, b, c, d = ids
                faces.extend([[a, b, d], [a, d, c]])
        asset = mesh_to_voxels(v, faces, grid_size=(4,)*3, aabb=[[0]*3, [1]*3], regularization_weight=0)
        vertices, triangles = voxels_to_mesh(asset)
        triangles, report = orient_faces(vertices, triangles, outward=True)
        self.assertEqual(len(report['components']), 1)
        self.assertTrue(report['components'][0]['closed'])
        self.assertAlmostEqual(report['components'][0]['signed_volume'], .6**3)
        self.assertGreater(len(triangles), 0)
        edges = Counter(tuple(sorted((int(a), int(b)))) for tri in triangles
                        for a, b in zip(tri, np.roll(tri, -1)))
        self.assertTrue(all(n == 2 for n in edges.values()))
        self.assertEqual(len(np.unique(triangles)) - len(edges) + len(triangles), 2)
        np.testing.assert_allclose(vertices.min(axis=0), [.2]*3, atol=1e-12)
        np.testing.assert_allclose(vertices.max(axis=0), [.8]*3, atol=1e-12)

    def test_empty_grid(self):
        a = VoxelAsset(np.empty((0, 3), int), np.empty((0, 3)), np.empty((0, 3), bool),
                       (4,)*3, [[0]*3, [1]*3])
        v, f = voxels_to_mesh(a)
        self.assertEqual(v.shape, (0, 3))
        self.assertEqual(f.shape, (0, 3))

    def test_resolve_nonmanifold_faces_prunes_spurious_fin(self):
        # 3 triangles sharing edge (0, 1): 2 form a quad, 1 is a tiny fin sticking out
        v = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.1]   # fin vertex
        ])
        # Two main quad faces: [0, 1, 2] and [0, 2, 3] (edge (0, 2) is internal, edges on boundary)
        # Add fin sharing edge (0, 1): [0, 1, 4]
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 1, 4]
        ])
        # Before: edge (0, 1) has 2 incident faces, but wait: edge (0, 1) is in [0, 1, 2] and [0, 1, 4] -> 2 faces.
        # Let's add another face sharing edge (0, 1): e.g. [1, 0, 4] or another face to create >2 faces on edge (0, 1)
        faces_nm = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 1, 4],
            [1, 0, 2]   # duplicate/triple on (0, 1) and (1, 2)
        ])
        cleaned = resolve_nonmanifold_faces(v, faces_nm)
        # Check all edges in cleaned have at most 2 incident faces
        edges = Counter(tuple(sorted((int(a), int(b)))) for tri in cleaned
                        for a, b in zip(tri, np.roll(tri, -1)))
        self.assertTrue(all(count <= 2 for count in edges.values()))
        self.assertLess(len(cleaned), len(faces_nm))

    def test_resolve_nonmanifold_faces_large_vertex_indices(self):
        # Regression: packed edge keys with large vertex ids must not be used as
        # array indices inside the cleanup loop.
        rng = np.random.default_rng(0)
        v = rng.random((200_000, 3))
        faces = np.array([[16384, 71837, 5], *[[16384, 71837, i] for i in range(6, 14)]])
        cleaned = resolve_nonmanifold_faces(v, faces)
        edges = Counter(tuple(sorted((int(a), int(b)))) for tri in cleaned
                        for a, b in zip(tri, np.roll(tri, -1)))
        self.assertTrue(all(count <= 2 for count in edges.values()))
        self.assertEqual(len(cleaned), 2)

    def test_resolve_nonmanifold_faces_preserves_clean_manifold(self):
        # Cube mesh is already 2-manifold
        v = np.array([[x, y, z] for x in (.2, .8) for y in (.2, .8) for z in (.2, .8)])
        faces = []
        for axis in range(3):
            for side in (.2, .8):
                ids = np.flatnonzero(v[:, axis] == side)
                a, b, c, d = ids
                faces.extend([[a, b, d], [a, d, c]])
        faces = np.array(faces)
        cleaned = resolve_nonmanifold_faces(v, faces)
        self.assertEqual(len(cleaned), len(faces))
