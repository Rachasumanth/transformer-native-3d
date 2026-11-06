import unittest
import numpy as np
from t3d.materials import closest_surface
from t3d.proximity import TriangleBVH


class ProximityTests(unittest.TestCase):
    def test_spatial_pruning_matches_exhaustive_projection(self):
        rng=np.random.default_rng(704)
        centers=np.array([[x*3,y*3,z*3] for x in range(4) for y in range(4) for z in range(4)])
        triangles=centers[:,None,:]+rng.uniform(-.3,.3,(len(centers),3,3))
        vertices=triangles.reshape(-1,3)
        faces=np.arange(len(vertices)).reshape(-1,3)
        points=centers+rng.normal(0,.2,centers.shape)
        expected_ids,expected_bc=closest_surface(points,vertices,faces,method='brute')
        stats={}
        ids,bc=TriangleBVH(vertices,faces,leaf_size=4).query(points,diagnostics=stats)
        np.testing.assert_array_equal(ids,expected_ids)
        np.testing.assert_allclose(bc,expected_bc,atol=1e-12)
        self.assertLess(stats['triangle_query_pairs'],stats['exhaustive_pairs']//2)
        auto_ids,auto_bc=closest_surface(points,vertices,faces)
        np.testing.assert_array_equal(auto_ids,expected_ids)
        np.testing.assert_allclose(auto_bc,expected_bc,atol=1e-12)

    def test_tie_uses_source_order_and_empty_queries(self):
        vertices=np.array([[x,y,z] for x in (1,-1) for y,z in ((0,0),(1,0),(0,1))])
        faces=np.arange(6).reshape(2,3)
        tree=TriangleBVH(vertices,faces,leaf_size=1)
        ids,bc=tree.query([[0,.25,.25]])
        np.testing.assert_array_equal(ids,[0])
        np.testing.assert_allclose(bc,[[.5,.25,.25]])
        ids,bc=tree.query(np.empty((0,3)))
        self.assertEqual(ids.shape,(0,));self.assertEqual(bc.shape,(0,3))
        with self.assertRaises(ValueError):tree.query([[np.nan,0,0]])
