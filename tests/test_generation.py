import importlib.util
import unittest
HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from t3d.generation import ThreeStageFlow


@unittest.skipUnless(HAS_TORCH,'Three-stage CUDA tests run on Kaggle')
class GenerationTests(unittest.TestCase):
    def setUp(self):self.device='cuda' if torch.cuda.is_available() else 'cpu'
    def pipeline(self):
        return ThreeStageFlow(2,1,3,grid_size=(2,2,2),width=24,heads=2,depth=1,patch_size=4).to(self.device)

    def test_stages_receive_correct_support_and_clean_shape(self):
        class Oracle(nn.Module):
            def __init__(self,mode,empty=False):super().__init__();self.mode=mode;self.empty=empty
            def forward(self,x,coords,time,condition,*,shape=None,**kwargs):
                if self.mode=='occupancy':
                    target=((coords.sum(dim=1)==0)&(not self.empty)).to(x.dtype)[:,None]
                elif self.mode=='geometry':target=x.new_tensor([2.,3.]).expand(len(x),2)
                else:
                    assert shape is not None
                    torch.testing.assert_close(shape,x.new_tensor([2.,3.]).expand(len(x),2))
                    target=shape[:,:1]*4
                return (target-x)/(1-time)
        m=self.pipeline();m.occupancy=Oracle('occupancy');m.geometry=Oracle('geometry');m.material=Oracle('material')
        result=m.generate(torch.ones(1,3,device=self.device),steps=3)
        torch.testing.assert_close(result.coords,torch.tensor([[0,0,0]],device=self.device))
        torch.testing.assert_close(result.shape,torch.tensor([[2.,3.]],device=self.device))
        torch.testing.assert_close(result.material,torch.tensor([[8.]],device=self.device))
        m.occupancy.empty=True
        empty=m.generate(None,steps=2)
        self.assertEqual(empty.shape.shape,(0,2));self.assertEqual(empty.material.shape,(0,1))

    def test_three_losses_backward_and_invalid_grid(self):
        m=self.pipeline();coords=torch.tensor([[0,0,0],[1,1,1]],device=self.device)
        losses=m.losses(coords,torch.randn(2,2,device=self.device),torch.randn(2,1,device=self.device),
                         torch.randn(2,3,device=self.device))
        self.assertEqual(set(losses),{'occupancy','geometry','material'})
        sum(losses.values()).backward()
        for model in (m.occupancy,m.geometry,m.material):
            self.assertIsNotNone(model.output.weight.grad)
            self.assertTrue(torch.isfinite(model.output.weight.grad).all())
        with self.assertRaises(ValueError):m.losses(coords*3,torch.ones(2,2,device=self.device),torch.ones(2,1,device=self.device),None)
        with self.assertRaises(ValueError):ThreeStageFlow(2,1,3,grid_size=(100,)*3)
