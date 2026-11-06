import importlib.util
import unittest
import numpy as np
from t3d.schema import VoxelAsset
from t3d.extraction import voxels_to_mesh as cpu_extract
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from t3d.extraction_torch import voxels_to_mesh


@unittest.skipUnless(HAS_TORCH, 'Torch extraction is checked on Kaggle')
class TensorExtractionTests(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def extract(self, asset):
        tensor = lambda x: torch.as_tensor(x, device=self.device)
        return voxels_to_mesh(tensor(asset.coords.astype(np.int64)), tensor(asset.dual_vertices),
                              tensor(asset.intersected_flag), tuple(int(x) for x in asset.grid_size),
                              asset.aabb, None if asset.split_weight is None else tensor(asset.split_weight))

    def test_random_sparse_parity_and_input_order(self):
        rng = np.random.default_rng(129)
        all_coords = np.indices((4,5,6)).reshape(3,-1).T
        for weighted in (False, True):
            for sparse in (False, True):
                coords = all_coords[rng.permutation(len(all_coords))]
                if sparse:
                    coords = coords[:90]
                asset = VoxelAsset(coords, rng.random(coords.shape), rng.random(coords.shape) > .4,
                                   (4,5,6), [[-1,-2,-3],[2,4,6]],
                                   split_weight=rng.normal(size=(len(coords),1))*1e100 if weighted else None)
                expected_v, expected_f = cpu_extract(asset)
                v, f = self.extract(asset)
                self.assertEqual(v.device.type, self.device)
                np.testing.assert_allclose(v.cpu().numpy(), expected_v)
                np.testing.assert_array_equal(f.cpu().numpy(), expected_f)

    def test_boundary_aliases_degenerate_and_empty(self):
        # A y overflow could alias a valid x neighbor if key bounds were ignored.
        for coords in (np.array([[0,1,0],[1,0,0],[1,0,1]]), np.empty((0,3),int)):
            asset = VoxelAsset(coords, np.zeros(coords.shape), np.ones(coords.shape,bool),
                               (2,2,2), [[0]*3,[2]*3])
            v,f = self.extract(asset)
            np.testing.assert_array_equal(f.cpu().numpy(), cpu_extract(asset)[1])
        coords = np.array([[0,0,0],[0,0,1],[0,1,1],[0,1,0]])
        offsets = np.array([[.5,1,1],[.5,1,0],[.5,0,0],[.5,0,1]])
        asset = VoxelAsset(coords,offsets,np.array([[1,0,0],[0,0,0],[0,0,0],[0,0,0]],bool),
                           (2,2,2),[[0]*3,[2]*3])
        np.testing.assert_array_equal(self.extract(asset)[1].cpu().numpy(), cpu_extract(asset)[1])

    def test_duplicate_and_overflow_rejection(self):
        device=self.device
        c=torch.zeros((2,3),device=device,dtype=torch.int64)
        d=torch.zeros((2,3),device=device)
        f=torch.zeros((2,3),device=device,dtype=torch.bool)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            voxels_to_mesh(c,d,f,(2,2,2),[[0]*3,[1]*3])
        with self.assertRaisesRegex(ValueError,'int64'):
            voxels_to_mesh(c,d,f,(2**30,)*3,[[0]*3,[1]*3])
