"""Unit tests for geometry canonicalization and synthetic mesh corruption."""
from collections import Counter
import unittest
import numpy as np
import trimesh

from t3d.data.refinement.canonicalize_geom import (
    canonicalize_geometry,
    normalize_coordinates,
    compute_udf_and_flood_fill,
    extract_manifold_isosurface,
    decimate_crease_preserving,
)
from t3d.data.refinement.corrupt_mesh import synthesize_corrupted_mesh


class TestCanonicalizeGeom(unittest.TestCase):
    def setUp(self):
        # Create test primitives with non-zero centers and arbitrary scales
        self.sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.7)
        self.sphere.vertices += [1.2, -0.8, 3.4]

        self.box = trimesh.creation.box(extents=(2.0, 4.0, 1.0))
        self.box.vertices += [-2.0, 5.0, 0.5]

        self.cylinder = trimesh.creation.cylinder(radius=0.8, height=2.4, sections=32)
        self.cylinder.vertices += [0.5, 0.5, -1.0]

    def test_coordinate_and_margin_normalization(self):
        """Validates that primitives are normalized inside [-0.45, 0.45]^3 with 10% margin."""
        for prim in (self.sphere, self.box, self.cylinder):
            norm = normalize_coordinates(prim, target_half_extent=0.45, margin_domain=0.5)

            bounds = norm.bounds
            center = (bounds[0] + bounds[1]) / 2.0
            extents = bounds[1] - bounds[0]
            max_extent = float(np.max(extents))

            # 1. Bounding box is centered at (0, 0, 0)
            np.testing.assert_allclose(center, [0.0, 0.0, 0.0], atol=1e-5)

            # 2. Longest axis extent is strictly 0.90 (within [-0.45, 0.45])
            self.assertAlmostEqual(max_extent, 0.90, places=5)

            # 3. All vertex coordinates are strictly within [-0.45, 0.45]
            self.assertTrue(np.all(norm.vertices >= -0.45 - 1e-6))
            self.assertTrue(np.all(norm.vertices <= 0.45 + 1e-6))

            # 4. Mandatory 10% empty boundary margin within [-0.5, 0.5]^3 is strictly preserved
            self.assertGreaterEqual(bounds[0].min(), -0.45 - 1e-6)
            self.assertLessEqual(bounds[1].max(), 0.45 + 1e-6)
            self.assertGreaterEqual(bounds[0].min() - (-0.5), 0.05 - 1e-6)
            self.assertGreaterEqual(0.5 - bounds[1].max(), 0.05 - 1e-6)

    def test_remeshing_watertight_and_nonmanifold_edges(self):
        """Validates UDF + Outside-In Flood Fill remeshing produces 0 non-manifold edges,

        watertight topology, and face count within [1500, 5000].
        """
        # Test cylinder (which has sharp 90-degree creases and curved sides)
        cyl = trimesh.creation.cylinder(radius=0.35, height=0.75, sections=32)
        clean = canonicalize_geometry(cyl, target_faces=(1500, 5000), resolution=128)

        # 1. Output is trimesh.Trimesh
        self.assertIsInstance(clean, trimesh.Trimesh)

        # 2. Coordinates strictly inside [-0.45, 0.45]^3
        self.assertTrue(np.all(clean.vertices >= -0.45 - 1e-5))
        self.assertTrue(np.all(clean.vertices <= 0.45 + 1e-5))

        # 3. Face count is within [1500, 5000]
        self.assertGreaterEqual(len(clean.faces), 1500)
        self.assertLessEqual(len(clean.faces), 5000)

        # 4. Zero non-manifold edges
        edges = [
            tuple(sorted((int(a), int(b))))
            for tri in clean.faces
            for a, b in zip(tri, np.roll(tri, -1))
        ]
        counts = Counter(edges)
        non_manifold = [e for e, c in counts.items() if c > 2]
        self.assertEqual(len(non_manifold), 0, f"Found non-manifold edges: {len(non_manifold)}")

        # 5. Watertight topology (0 boundary edges, every edge shared by exactly 2 faces)
        boundary = [e for e, c in counts.items() if c == 1]
        self.assertEqual(len(boundary), 0, f"Found boundary edges: {len(boundary)}")
        self.assertTrue(all(c == 2 for c in counts.values()))
        self.assertTrue(clean.is_watertight)

    def test_open_single_sided_sheet_remeshing(self):
        """Validates that an open single-sided sheet is thickened and remeshed into a 2-manifold."""
        # 2-triangle planar quad
        v = np.array([[-0.3, -0.3, 0.0], [0.3, -0.3, 0.0], [0.3, 0.3, 0.0], [-0.3, 0.3, 0.0]])
        f = np.array([[0, 1, 2], [0, 2, 3]])
        open_mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)

        clean = canonicalize_geometry(open_mesh, target_faces=(1500, 5000), resolution=128)
        self.assertTrue(clean.is_watertight)
        self.assertGreater(clean.volume, 0.0)

        # Check edge manifoldness
        edges = [
            tuple(sorted((int(a), int(b))))
            for tri in clean.faces
            for a, b in zip(tri, np.roll(tri, -1))
        ]
        counts = Counter(edges)
        self.assertTrue(all(c == 2 for c in counts.values()))

    def test_synthetic_corruption_generator(self):
        """Validates synthetic corruption produces degraded mesh with holes, jitter, and distractors."""
        clean = trimesh.creation.icosphere(subdivisions=4, radius=0.4)
        corrupted = synthesize_corrupted_mesh(clean, seed=42)

        self.assertIsInstance(corrupted, trimesh.Trimesh)

        # 1. Mesh contains holes (not watertight, has open boundary edges)
        self.assertFalse(corrupted.is_watertight)
        edges = [
            tuple(sorted((int(a), int(b))))
            for tri in corrupted.faces
            for a, b in zip(tri, np.roll(tri, -1))
        ]
        counts = Counter(edges)
        boundary_edges = [e for e, c in counts.items() if c == 1]
        self.assertGreater(len(boundary_edges), 0)

        # 2. Aggressive unconstrained decimation produces roughly 300-500 faces
        # (plus 3-8 distractor faces)
        self.assertLessEqual(len(corrupted.faces), 600)
        self.assertGreaterEqual(len(corrupted.faces), 200)

        # 3. 32^3 coordinate quantization
        # Coordinates must be close to grid multiples of 1/32 = 0.03125
        quantized_diff = np.abs(corrupted.vertices * 32.0 - np.round(corrupted.vertices * 32.0))
        # Note: distractor triangles add small offsets, so check primary mesh vertices
        n_quantized = np.sum(quantized_diff < 1e-4)
        self.assertGreater(n_quantized, len(corrupted.vertices) * 0.7)

        # 4. Floating distractors: multiple disconnected components
        components = corrupted.split(only_watertight=False)
        self.assertGreater(len(components), 1)

        # 5. Deterministic reproducibility with seed
        repeat = synthesize_corrupted_mesh(clean, seed=42)
        np.testing.assert_allclose(corrupted.vertices, repeat.vertices)
        np.testing.assert_array_equal(corrupted.faces, repeat.faces)

        # 6. Unpacking support (v, f = corrupted)
        v, f = corrupted
        self.assertEqual(len(v), len(corrupted.vertices))
        self.assertEqual(len(f), len(corrupted.faces))


if __name__ == "__main__":
    unittest.main()
