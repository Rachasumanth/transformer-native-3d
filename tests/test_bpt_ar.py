import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from t3d.bpt_ar import BPTAutoregressiveTransformer, teacher_forcing_batch, BOS, EOS, PAD


@unittest.skipUnless(HAS_TORCH, 'PyTorch execution provided by Kaggle')
class BPTARTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BPTAutoregressiveTransformer(width=24, heads=2, depth=2,
            condition_tokens=4, context_length=16).to(self.device).eval()
        self.points = torch.randn(2, 32, 3, device=self.device)

    def test_causality_and_point_permutation(self):
        tokens = torch.randint(0, 5120, (2, 9), device=self.device)
        with torch.no_grad():
            original = self.model(tokens, self.points)
            changed = tokens.clone()
            changed[:, 5:] = (changed[:, 5:] + 113) % 5120
            modified = self.model(changed, self.points)
            permuted = self.model(tokens, self.points.flip(1))
        torch.testing.assert_close(original[:, :5], modified[:, :5], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(original, permuted, atol=2e-5, rtol=2e-5)
        self.assertGreater((original[:, 5:] - modified[:, 5:]).abs().max().item(), .01)

    def test_teacher_forcing_and_padding_loss(self):
        inputs, targets = teacher_forcing_batch([[2, 1024, 1025], [3]], self.device)
        self.assertEqual(inputs.tolist(), [[BOS, 2, 1024, 1025], [BOS, 3, PAD, PAD]])
        self.assertEqual(targets.tolist(), [[2, 1024, 1025, EOS], [3, EOS, PAD, PAD]])
        loss = self.model.loss(inputs, targets, self.points)
        individual = (self.model.loss(inputs[:1], targets[:1], self.points[:1]) * 4 +
            self.model.loss(inputs[1:2, :2], targets[1:2, :2], self.points[1:2]) * 2) / 6
        torch.testing.assert_close(loss, individual)
        loss.backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(self.model.point[0].weight.grad.abs().sum().item(), 0)

    def test_point_condition_changes_predictions(self):
        tokens = torch.full((2, 5), BOS, device=self.device)
        with torch.no_grad():
            a = self.model(tokens, self.points)
            b = self.model(tokens, self.points + 1)
        self.assertGreater((a - b).abs().max().item(), .01)

    def test_cached_logits_match_full_causal_forward(self):
        tokens = torch.randint(0, 5120, (2, 9), device=self.device)
        with torch.no_grad():
            expected = self.model(tokens, self.points)
            condition = self.model.encode_points(self.points)
            caches = [None] * len(self.model.blocks)
            actual = []
            for position in range(tokens.shape[1]):
                x = self.model.token(tokens[:, position:position + 1]) + self.model.position.weight[position]
                for i, block in enumerate(self.model.blocks):
                    x, caches[i] = block.step(x, condition, caches[i])
                actual.append(self.model.head(self.model.norm(x)))
            torch.testing.assert_close(torch.cat(actual, dim=1), expected, atol=2e-5, rtol=2e-5)

    def test_generation_stops_and_limits(self):
        with torch.no_grad():
            self.model.head.weight.zero_()
            self.model.head.weight[EOS] = self.model.norm.bias + 1
            self.model.norm.bias.fill_(10)
        result = self.model.generate(self.points[:1])
        self.assertEqual(result, dict(tokens=[], stop_reason='eos'))
        with torch.no_grad():
            self.model.head.weight[EOS].zero_()
            self.model.head.weight[4].fill_(1)
        result = self.model.generate(self.points[:1], max_tokens=3)
        self.assertEqual(result, dict(tokens=[4, 4, 4], stop_reason='token_limit'))
        with self.assertRaises(ValueError):
            self.model.generate(self.points[:1], max_tokens=17)

    def test_crop_position_offset(self):
        tokens = torch.randint(0, 5120, (2, 4), device=self.device)
        with torch.no_grad():
            expected = self.model(tokens, self.points, position_offset=3)
            condition = self.model.encode_points(self.points)
            caches = [None] * len(self.model.blocks)
            values = []
            for position in range(4):
                x = self.model.token(tokens[:, position:position + 1]) + self.model.position.weight[position + 3]
                for i, block in enumerate(self.model.blocks):
                    x, caches[i] = block.step(x, condition, caches[i])
                values.append(self.model.head(self.model.norm(x)))
            torch.testing.assert_close(expected, torch.cat(values, 1), atol=2e-5, rtol=2e-5)
        with self.assertRaises(ValueError):
            self.model(tokens, self.points, position_offset=13)

    def test_constrained_sampling_defers_eos_until_valid_patch(self):
        from t3d.bpt import decode
        import numpy as np
        with torch.no_grad():
            self.model.head.weight.zero_()
            self.model.norm.bias.fill_(10)
            self.model.head.weight[EOS].fill_(2)
            self.model.head.weight[1024:].fill_(1)
            self.model.head.weight[BOS].zero_()
            self.model.head.weight[PAD].zero_()
            self.model.head.weight[EOS].fill_(2)
        result = self.model.generate(self.points[:1], grammar_constrained=True)
        self.assertEqual(result['stop_reason'], 'eos')
        vertices, faces = decode(np.asarray(result['tokens']))
        self.assertEqual(len(faces), 1)
        self.assertGreater(np.linalg.norm(np.cross(vertices[faces[0, 1]] - vertices[faces[0, 0]],
                                                 vertices[faces[0, 2]] - vertices[faces[0, 0]])), 0)


if __name__ == '__main__':
    unittest.main()
