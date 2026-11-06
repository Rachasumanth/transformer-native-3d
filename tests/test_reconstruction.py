import importlib.util
import unittest
HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from t3d.sc_vae import SparseCompressionVAE
    from t3d.reconstruction import reconstruct_asset
    from test_dataset import fixture


@unittest.skipUnless(HAS_TORCH,'Decode on Kaggle')
class ReconstructionTests(unittest.TestCase):
    def test_edge_threshold_does_not_change_support(self):
        device='cuda' if torch.cuda.is_available() else 'cpu'
        shape=SparseCompressionVAE('shape',channels=(2,4,8,16,32),latent_channels=4).to(device)
        with torch.no_grad():
            shape.output.weight.zero_();shape.output.bias.zero_()
        a=fixture()
        low=reconstruct_asset(a,shape,support='teacher',threshold=.5,edge_threshold=.1)
        high=reconstruct_asset(a,shape,support='teacher',threshold=.5,edge_threshold=.9)
        self.assertEqual(low.coords.tolist(),high.coords.tolist())
        self.assertTrue(low.intersected_flag.all())
        self.assertFalse(high.intersected_flag.any())
        with self.assertRaises(ValueError):reconstruct_asset(a,shape,edge_threshold=float('nan'))

    def test_teacher_and_empty_predicted_support(self):
        device='cuda' if torch.cuda.is_available() else 'cpu'
        shape=SparseCompressionVAE('shape',channels=(2,4,8,16,32),latent_channels=4).to(device)
        material=SparseCompressionVAE('material',channels=(2,4,8,16,32),latent_channels=4).to(device)
        a=fixture()
        restored=reconstruct_asset(a,shape,material,support='teacher')
        self.assertEqual(restored.coords.tolist(),a.coords.tolist())
        self.assertEqual(restored.pbr.shape,(1,6))
        with torch.no_grad():
            for up in shape.up:
                up.occupancy_head.weight.zero_();up.occupancy_head.bias.fill_(-100)
        empty=reconstruct_asset(a,shape,material,support='predicted')
        self.assertEqual(len(empty.coords),0)
        self.assertEqual(empty.pbr.shape,(0,6))
