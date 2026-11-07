"""Tests for watertight dual contouring extraction and hole closure."""
import unittest
from pathlib import Path
from collections import Counter
import numpy as np
import trimesh

from t3d.schema import VoxelAsset
from t3d.voxelize import mesh_to_voxels
from t3d.extraction import voxels_to_mesh, resolve_nonmanifold_faces
from t3d.container import load


class WatertightExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus_dir = Path(__file__).resolve().parent.parent / "artifacts" / "kenney-corpus-64-expanded-merged"

    def test_synthetic_cube_watertight(self):
        """A voxelized cube extracted with resolve_nonmanifold=True is 100% watertight."""
        v = np.array([[x, y, z] for x in (.2, .8) for y in (.2, .8) for z in (.2, .8)])
        faces = []
        for axis in range(3):
            for side in (.2, .8):
                ids = np.flatnonzero(v[:, axis] == side)
                a, b, c, d = ids
                faces.extend([[a, b, d], [a, d, c]])
        asset = mesh_to_voxels(v, faces, grid_size=(4,)*3, aabb=[[0]*3, [1]*3], regularization_weight=0)
        vertices, triangles = voxels_to_mesh(asset, resolve_nonmanifold=True)

        m = trimesh.Trimesh(vertices, triangles, process=False)
        self.assertTrue(m.is_watertight)
        b_rows = trimesh.grouping.group_rows(m.edges_sorted, require_count=1)
        self.assertEqual(len(b_rows), 0)
        # Verify 0 non-manifold edges (>2 incident faces)
        counts = Counter(map(tuple, m.edges_sorted))
        self.assertTrue(all(count <= 2 for count in counts.values()))

    def test_synthetic_missing_neighbor_halo_closure(self):
        """Missing transverse neighbors on marked primal edges are filled by halo shell."""
        # Quad on axis 0 with 1 cell omitted from asset.coords
        coords = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]], dtype=int)
        flags = np.zeros((3, 3), bool)
        flags[0, 0] = True
        asset = VoxelAsset(
            coords,
            np.full(coords.shape, .5),
            flags,
            (2,)*3,
            [[0]*3, [2]*3]
        )
        # Without halo: quad is dropped
        v_no_halo, f_no_halo = voxels_to_mesh(asset, halo=False)
        self.assertEqual(len(f_no_halo), 0)

        # With halo: dual quad is preserved via exterior halo vertex
        v_halo, f_halo = voxels_to_mesh(asset, halo=True)
        self.assertEqual(len(f_halo), 2)
        self.assertEqual(len(v_halo), 4)

    def test_halo_closure_at_extreme_grid_boundary(self):
        """Halo vertices must not be clipped at grid boundaries [-1, resolution]."""
        # Occupied cell at the extreme edge of a 4x4x4 grid (coord = [3, 3, 3])
        # Transverse neighbors will require coordinates at [3, 3, 4] or [4, 4, 4]
        coords = np.array([[3, 3, 3]], dtype=int)
        flags = np.zeros((1, 3), bool)
        flags[0, 0] = True  # primal edge on axis 0
        asset = VoxelAsset(
            coords,
            np.full(coords.shape, .5),
            flags,
            (4, 4, 4),
            [[0]*3, [4]*3]
        )
        # Verify halo allocates vertices at or beyond resolution (4.0) without crashing or dropping
        v_halo, f_halo = voxels_to_mesh(asset, halo=True)
        self.assertEqual(len(f_halo), 2)
        self.assertEqual(len(v_halo), 4)
        # At least one halo vertex is at or beyond the grid boundary (>= 3.5)
        self.assertTrue(np.any(v_halo >= 3.5))

    def test_resolve_nonmanifold_faces_closes_boundary_holes(self):
        """Pruning non-manifold edges leaves holes that are closed by ear clipping."""
        # Sphere-like closed octahedron with an extra spurious fin face
        v = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.5, 0.5, 0.5],  # Fin vertex
        ])
        # 8 faces of octahedron (watertight) + 1 fin face sharing edge (0, 2)
        faces = np.array([
            [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
            [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
            [0, 2, 6],  # Spurious fin on edge (0, 2)
        ])
        cleaned = resolve_nonmanifold_faces(v, faces, close_holes=True)
        m = trimesh.Trimesh(v, cleaned, process=False)
        self.assertTrue(m.is_watertight)
        b_rows = trimesh.grouping.group_rows(m.edges_sorted, require_count=1)
        self.assertEqual(len(b_rows), 0)

    def test_kenney_blaster_watertight(self):
        """Real asset blaster-a is 100% watertight with 0 boundary edges."""
        blaster_path = self.corpus_dir / "kenney-blaster-kit-blaster-a.t3d"
        if not blaster_path.exists():
            self.skipTest(f"Corpus asset not found: {blaster_path}")
        asset = load(str(blaster_path)).asset
        v, f = voxels_to_mesh(asset, resolve_nonmanifold=True)
        m = trimesh.Trimesh(v, f, process=False)

        self.assertTrue(m.is_watertight, "Blaster mesh must be watertight")
        b_rows = trimesh.grouping.group_rows(m.edges_sorted, require_count=1)
        self.assertEqual(
            len(m.edges_unique[b_rows]) if len(b_rows) > 0 else 0,
            0,
            "Blaster must have exactly 0 boundary edges"
        )
        # Verify 0 non-manifold edges
        counts = Counter(map(tuple, m.edges_sorted))
        self.assertTrue(all(count <= 2 for count in counts.values()))

    def test_kenney_suburban_fence_watertight(self):
        """Real asset suburban-fence is 100% watertight with 0 boundary edges."""
        fence_path = self.corpus_dir / "kenney-city-kit-suburban-fence.t3d"
        if not fence_path.exists():
            self.skipTest(f"Corpus asset not found: {fence_path}")
        asset = load(str(fence_path)).asset
        v, f = voxels_to_mesh(asset, resolve_nonmanifold=True)
        m = trimesh.Trimesh(v, f, process=False)

        self.assertTrue(m.is_watertight, "Suburban fence mesh must be watertight")
        b_rows = trimesh.grouping.group_rows(m.edges_sorted, require_count=1)
        self.assertEqual(
            len(m.edges_unique[b_rows]) if len(b_rows) > 0 else 0,
            0,
            "Fence must have exactly 0 boundary edges"
        )
        counts = Counter(map(tuple, m.edges_sorted))
        self.assertTrue(all(count <= 2 for count in counts.values()))

    def test_kenney_construction_cone_watertight(self):
        """Real asset construction-cone is 100% watertight with 0 boundary edges."""
        cone_path = self.corpus_dir / "kenney-city-kit-roads-construction-cone.t3d"
        if not cone_path.exists():
            self.skipTest(f"Corpus asset not found: {cone_path}")
        asset = load(str(cone_path)).asset
        v, f = voxels_to_mesh(asset, resolve_nonmanifold=True)
        m = trimesh.Trimesh(v, f, process=False)

        self.assertTrue(m.is_watertight, "Construction cone must be watertight")
        b_rows = trimesh.grouping.group_rows(m.edges_sorted, require_count=1)
        self.assertEqual(
            len(m.edges_unique[b_rows]) if len(b_rows) > 0 else 0,
            0,
            "Cone must have exactly 0 boundary edges"
        )


if __name__ == "__main__":
    unittest.main()
