import importlib.util
import unittest
HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from t3d.flow import flow_pair,GramSchmidtRoPE,SparseFlowDiT,flow_loss,sample_flow


@unittest.skipUnless(HAS_TORCH,'Flow CUDA verification in Kaggle')
class FlowTests(unittest.TestCase):
    def setUp(self):self.device='cuda' if torch.cuda.is_available() else 'cpu'
    def test_trajectory_and_euler_solution(self):
        clean=torch.tensor([[3.,4]],device=self.device);noise=torch.tensor([[1.,0]],device=self.device)
        x,v=flow_pair(clean,noise,clean.new_tensor(.25))
        torch.testing.assert_close(x,torch.tensor([[1.5,1]],device=self.device))
        class Oracle:
            def __call__(self,*args,**kwargs):return v
        torch.testing.assert_close(sample_flow(Oracle(),noise,None,None,steps=8,method='euler'),clean)
        torch.testing.assert_close(sample_flow(Oracle(),noise,None,None,steps=8,method='midpoint'),clean)
        with self.assertRaises(ValueError):
            sample_flow(Oracle(),noise,None,None,steps=8,method='rk4')

    def test_rope_rotation_and_norm(self):
        rope=GramSchmidtRoPE(2,12).to(self.device)
        with torch.no_grad():rope.axes.zero_()
        rotation=rope.rotation()
        torch.testing.assert_close(rotation@rotation.transpose(-1,-2),torch.eye(3,device=self.device).expand(2,3,3))
        q=torch.randn(2,7,12,device=self.device)
        out=rope(q,torch.randn(7,3,device=self.device))
        torch.testing.assert_close(out.norm(dim=-1),q.norm(dim=-1))

    def test_conditional_model_optimization_and_shapes(self):
        torch.manual_seed(7)
        m=SparseFlowDiT(8,6,width=24,heads=2,depth=2,patch_size=3,shape_channels=4).to(self.device)
        x=torch.randn(7,8,device=self.device);coords=torch.randn(7,3,device=self.device)
        prompt=torch.randn(3,6,device=self.device);shape=torch.randn(7,4,device=self.device)
        order=torch.arange(7,device=self.device).flip(0)
        optimizer=torch.optim.AdamW(m.parameters(),lr=.001)
        for _ in range(3):
            optimizer.zero_grad();loss=flow_loss(m,x,coords,prompt,orders=[order],shape=shape)
            self.assertTrue(torch.isfinite(loss));loss.backward();optimizer.step()
        for p in m.parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
        result=sample_flow(m,x,coords,prompt,steps=3,shape=shape,orders=[order])
        self.assertEqual(result.shape,x.shape);self.assertTrue(torch.isfinite(result).all())
        result_mid=sample_flow(m,x,coords,prompt,steps=3,shape=shape,orders=[order],method='midpoint')
        self.assertEqual(result_mid.shape,x.shape);self.assertTrue(torch.isfinite(result_mid).all())
        with self.assertRaises(ValueError):m(x,coords,.5,prompt,shape=shape,orders=[order*0])
