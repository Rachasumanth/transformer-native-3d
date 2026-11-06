import itertools
import unittest
import torch
from t3d.triton_sparse import neighbor_map, triton_submanifold_conv
from t3d.sc_vae import SparseCompressionVAE
from t3d.sparse_blocks import SubmanifoldConv3d


class TritonSparseMapTests(unittest.TestCase):
    def test_neighbor_map_matches_coordinate_lookup(self):
        generator = torch.Generator().manual_seed(31)
        grid = torch.cartesian_prod(torch.arange(2), torch.arange(5), torch.arange(5), torch.arange(5))
        coords = grid[torch.randperm(len(grid), generator=generator)[:90]]
        mapping = neighbor_map(coords)
        lookup = {tuple(c.tolist()): i for i,c in enumerate(coords)}
        for row, coordinate in enumerate(coords.tolist()):
            for column, offset in enumerate(itertools.product((-1,0,1), repeat=3)):
                key=(coordinate[0], *(coordinate[i+1]+offset[i] for i in range(3)))
                self.assertEqual(int(mapping[row,column]), lookup.get(key,-1))

    def test_cpu_does_not_silently_use_experimental_kernel(self):
        coords = torch.zeros((1,4), dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError,'Triton and CUDA'):
            triton_submanifold_conv(coords,torch.ones((1,2)),torch.ones((27,2,2)),torch.ones(2))

    def test_backend_selection_preserves_checkpoint_keys(self):
        reference=SparseCompressionVAE('shape',channels=(2,4,8,16,32),latent_channels=4)
        accelerated=SparseCompressionVAE('shape',channels=(2,4,8,16,32),latent_channels=4,convolution_backend='triton')
        self.assertEqual(reference.state_dict().keys(),accelerated.state_dict().keys())
        accelerated.load_state_dict(reference.state_dict())
        self.assertEqual([m.backend for m in accelerated.modules() if isinstance(m,SubmanifoldConv3d)],['triton']*10)
        with self.assertRaisesRegex(ValueError,'Unknown sparse convolution backend'):
            SparseCompressionVAE(convolution_backend='bad')


if __name__ == '__main__':
    unittest.main()
