import importlib.util
import tempfile
from pathlib import Path
import unittest
HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from t3d.training import VAETrainer
    from test_dataset import fixture


@unittest.skipUnless(HAS_TORCH,'Training resume verified in Kaggle')
class TrainingTests(unittest.TestCase):
    def test_stochastic_resume_matches_next_update(self):
        device='cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(18)
        trainer=VAETrainer(mode='material',channels=(2,4,8,16,32),latent_channels=4,
                           device=device,manifest_sha256='fixture fingerprint')
        trainer.update([fixture()])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'resume.pt'; trainer.save(path)
            expected=trainer.update([fixture()])
            restored=VAETrainer.resume(path,device=device,manifest_sha256='fixture fingerprint')
            actual=restored.update([fixture()])
            for name in expected: self.assertAlmostEqual(actual[name],expected[name],places=6)
            for a,b in zip(trainer.model.parameters(),restored.model.parameters()):
                torch.testing.assert_close(a,b)
            self.assertEqual(restored.steps,2)
            self.assertEqual(restored.batch_cursor,2)
            with self.assertRaises(ValueError):
                VAETrainer.resume(path,device=device,manifest_sha256='wrong dataset')

    def test_weighted_shape_resume_keeps_objective(self):
        device='cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(71)
        weights=[2.,3.,4.]
        trainer=VAETrainer(mode='shape',channels=(2,4,8,16,32),latent_channels=4,
                           device=device,manifest_sha256='weighted',edge_positive_weight=weights)
        trainer.update([fixture()])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'weighted.pt';trainer.save(path)
            expected=trainer.update([fixture()])
            restored=VAETrainer.resume(path,device=device,manifest_sha256='weighted')
            self.assertEqual(restored.edge_positive_weight,weights)
            actual=restored.update([fixture()])
            for name in expected:self.assertAlmostEqual(actual[name],expected[name],places=6)
            for a,b in zip(trainer.model.parameters(),restored.model.parameters()):
                torch.testing.assert_close(a,b)

    def test_occupancy_weights_survive_resume(self):
        device='cuda' if torch.cuda.is_available() else 'cpu'
        trainer=VAETrainer(mode='shape',channels=(2,4,8,16,32),latent_channels=4,device=device,manifest_sha256='weighted-levels',occupancy_level_weights=[4,2,1,1])
        trainer.update([fixture()])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'weights.pt';trainer.save(path)
            expected=trainer.update([fixture()])
            resumed=VAETrainer.resume(path,device=device,manifest_sha256='weighted-levels')
            self.assertEqual(resumed.occupancy_level_weights,(4.,2.,1.,1.))
            actual=resumed.update([fixture()])
            for name in expected:self.assertAlmostEqual(expected[name],actual[name],places=6)
        with self.assertRaises(ValueError):
            VAETrainer(mode='shape',device=device,occupancy_level_weights=[0,0,0,0])

    def test_shuffled_order_and_batch_size_resume(self):
        trainer=VAETrainer(mode='shape',channels=(2,4,8,16,32),latent_channels=4,device='cpu',manifest_sha256='shuffle',shuffle_seed=42,batch_size=3)
        rng=torch.get_rng_state().clone();first=trainer.next_indices(7,3)
        torch.testing.assert_close(torch.get_rng_state(),rng)
        self.assertEqual(len(first),3);trainer.batch_cursor+=len(first)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'shuffle.pt';trainer.save(path)
            resumed=VAETrainer.resume(path,device='cpu',manifest_sha256='shuffle')
            self.assertEqual(resumed.shuffle_seed,42);self.assertEqual(resumed.batch_size,3)
            seen=list(first)
            while len(seen)<7:
                expected=trainer.next_indices(7,3);actual=resumed.next_indices(7,3)
                self.assertEqual(expected,actual);seen.extend(expected)
                trainer.batch_cursor+=len(expected);resumed.batch_cursor+=len(actual)
            self.assertEqual(sorted(seen),list(range(7)))
            self.assertEqual(trainer.next_indices(7,3),resumed.next_indices(7,3))
            self.assertEqual(trainer.epoch,1)
            with self.assertRaisesRegex(ValueError,'Batch size differs'):resumed.next_indices(7,2)
