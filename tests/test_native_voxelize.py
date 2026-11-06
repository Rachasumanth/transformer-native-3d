import unittest
import json
import numpy as np
from t3d.native_voxelize import from_native_grid


class NativeGridAdapterTests(unittest.TestCase):
    def test_physical_relative_positions_anisotropic_nonzero_origin(self):
        coords=np.array([[1,2,3],[0,0,0]])
        local=np.array([[.25,.5,.75],[.1,.2,.3]])
        bounds=[[-2,-3,-4],[2,7,20]];grid=(4,5,6)
        size=(np.array(bounds[1])-bounds[0])/grid
        positions=(coords+local)*size
        report={}
        asset=from_native_grid(coords,positions,np.zeros((2,3),bool),grid_size=grid,aabb=bounds,diagnostics=report)
        np.testing.assert_allclose(asset.world_vertices,positions[::-1]+bounds[0])
        np.testing.assert_allclose(asset.dual_vertices,local[::-1])
        self.assertEqual(report['clamped_components'],0)
        json.dumps(report)  # Regression: NumPy roundoff tolerance broke the real Kaggle report.

    def test_wrong_units_and_float_roundoff(self):
        coords=np.array([[2,2,2]])
        with self.assertRaisesRegex(ValueError,'position units'):
            from_native_grid(coords,[[.5,.5,.5]],np.zeros((1,3),bool),grid_size=(8,)*3,aabb=[[0]*3,[1]*3])
        report={}
        asset=from_native_grid(coords,[[.24999999,.3,.3]],np.zeros((1,3),bool),
                              grid_size=(8,)*3,aabb=[[0]*3,[1]*3],diagnostics=report)
        self.assertEqual(asset.dual_vertices[0,0],0)
        self.assertEqual(report['clamped_components'],1)
