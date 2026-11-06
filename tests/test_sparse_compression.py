import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from t3d.sparse_compression import downsample_shortcut, upsample_shortcut, SparseDownsample, SparseUpsample


@unittest.skipUnless(HAS_TORCH, 'PyTorch tests run in the Kaggle GPU notebook')
class CompressionTests(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_missing_children_and_batch_separation(self):
        c = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1], [1, 0, 0, 0]], device=self.device)
        f = torch.tensor([[1.], [8.], [4.]], device=self.device, requires_grad=True)
        parents, values, mask = downsample_shortcut(c, f, 2)
        torch.testing.assert_close(values, torch.tensor([[.25, 2], [1., 0]], device=self.device))
        self.assertEqual(mask.sum().item(), 3)
        values.sum().backward()
        torch.testing.assert_close(f.grad, torch.full_like(f, .25))

    def test_unaveraged_shortcut_is_exact(self):
        c = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 1]], device=self.device)
        f = torch.tensor([[1., 2.], [3., 4.]], device=self.device)
        p, g, m = downsample_shortcut(c, f, 16)
        restored, values = upsample_shortcut(p, g, 2, m)
        torch.testing.assert_close(restored, c)
        torch.testing.assert_close(values, f)

    def test_four_stage_support_and_gradients(self):
        torch.manual_seed(3)
        c = torch.tensor([[0, 0, 0, 0], [0, 15, 15, 15], [0, 7, 3, 9], [1, 2, 4, 8]], device=self.device)
        original = torch.unique(c, dim=0, sorted=True)
        f = torch.randn(4, 2, device=self.device, requires_grad=True)
        masks, levels = [], []
        for i in range(4):
            block = SparseDownsample(2*2**i, 4*2**i).to(self.device)
            levels.append(block)
            c, f, mask = block(c, f)
            masks.append(mask)
        self.assertEqual(len(c), 2)
        torch.testing.assert_close(c[:, 1:], torch.zeros_like(c[:, 1:]))
        losses = []
        for i in reversed(range(4)):
            block = SparseUpsample(4*2**i, 2*2**i).to(self.device)
            levels.append(block)
            c, f, logits = block(c, f, masks[i])
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logits, masks[i].float()))
        self.assertEqual(len(c), 4)
        torch.testing.assert_close(c, original)
        (f.square().mean()+sum(losses)).backward()
        for block in levels:
            for parameter in block.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_predicted_pruning_and_invalid_coordinates(self):
        c = torch.tensor([[0, 0, 0, 0]], device=self.device)
        f = torch.ones(1, 2, device=self.device)
        block = SparseUpsample(2, 1).to(self.device)
        with torch.no_grad():
            block.occupancy_head.weight.zero_()
            block.occupancy_head.bias.fill_(-100)
        children, values, logits = block(c, f)
        self.assertEqual(children.shape, (0, 4))
        self.assertEqual(values.shape, (0, 1))
        with self.assertRaises(ValueError):
            downsample_shortcut(c.repeat(2, 1), f.repeat(2, 1), 2)
