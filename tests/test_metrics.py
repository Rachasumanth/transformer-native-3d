import unittest
import numpy as np
from t3d.metrics import surface_comparison


class MetricTests(unittest.TestCase):
    def test_identical_and_translated_plane(self):
        v=np.array([[0.,0,0],[1,0,0],[0,1,0]]);f=np.array([[0,1,2]])
        identical=surface_comparison(v,f,v,f,samples=32)
        self.assertLess(identical['mean_distance'],1e-12)
        self.assertEqual(identical['fscore'],1)
        shifted=surface_comparison(v,f,v+[0,0,.1],f,samples=32)
        self.assertAlmostEqual(shifted['mean_distance'],.1)
        self.assertEqual(shifted['fscore'],0)
        self.assertEqual(shifted['boundary_edges'],3)
        self.assertAlmostEqual(shifted['boundary_edge_length'],2+2**.5)

    def test_topology_welds_duplicate_vertices_at_uv_seam(self):
        vertices=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0],
                           [0,0,0],[1,1,0]])
        faces=np.array([[0,1,2],[4,5,3]])
        result=surface_comparison(vertices,faces,vertices,faces,samples=32)
        self.assertEqual(result['boundary_edges'],4)
        self.assertEqual(result['nonmanifold_edges'],0)
        self.assertTrue(result['valid_topology'])
        self.assertAlmostEqual(result['boundary_edge_length'],4.)

    def test_empty_prediction_is_failure(self):
        result = surface_comparison([],[],[],[])
        self.assertFalse(result['valid'])
        self.assertFalse(result['valid_topology'])

    def test_nonmanifold_mesh_fails_topology_validation(self):
        # Three triangles sharing the same edge (0, 1)
        ref_v = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        ref_f = np.array([[0, 1, 2]])
        v = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]])
        f = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
        result = surface_comparison(ref_v, ref_f, v, f, samples=32)
        self.assertTrue(result['valid'])
        self.assertEqual(result['nonmanifold_edges'], 1)
        self.assertFalse(result['valid_topology'])
