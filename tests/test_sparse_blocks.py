import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from t3d.sparse_blocks import SubmanifoldConv3d, SparseConvNeXtBlock


@unittest.skipUnless(HAS_TORCH, 'PyTorch tests run in Kaggle')
class SparseBlockTests(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_neighbors_missing_cells_and_batches(self):
        coords = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], device=self.device)
        features = torch.tensor([[2.], [100.], [1.]], device=self.device, requires_grad=True)
        conv = SubmanifoldConv3d(1).to(self.device)
        with torch.no_grad():
            conv.weight.fill_(1)
            conv.bias.zero_()
        values = conv(coords, features)
        torch.testing.assert_close(values, torch.tensor([[3.], [100.], [3.]], device=self.device))
        values.sum().backward()
        torch.testing.assert_close(features.grad, torch.tensor([[2.], [1.], [2.]], device=self.device))

    def test_center_identity_and_coordinate_permutation(self):
        coords = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]], device=self.device)
        f = torch.randn(2, 3, device=self.device)
        conv = SubmanifoldConv3d(3).to(self.device)
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[13].copy_(torch.eye(3, device=self.device))
            conv.bias.zero_()
        torch.testing.assert_close(conv(coords, f), f)
        torch.testing.assert_close(conv(coords.flip(0), f.flip(0)), f.flip(0))

    def test_residual_block_gradients(self):
        coords = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]], device=self.device)
        f = torch.randn(2, 4, device=self.device, requires_grad=True)
        block = SparseConvNeXtBlock(4, layer_scale=1.).to(self.device)
        output = block(coords, f)
        self.assertEqual(output.shape, f.shape)
        output.square().mean().backward()
        for p in block.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
