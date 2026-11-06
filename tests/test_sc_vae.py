import importlib.util
import unittest
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from t3d.sc_vae import SparseCompressionVAE, reconstruction_loss


@unittest.skipUnless(HAS_TORCH, 'Run PyTorch tests in Kaggle')
class VaeTests(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.coords = torch.tensor([[0, 15, 15, 15], [0, 0, 0, 0], [0, 7, 3, 9], [1, 1, 2, 3]], device=self.device)

    def model(self, mode):
        return SparseCompressionVAE(mode, channels=(2, 4, 8, 16, 32), latent_channels=4).to(self.device)

    def test_shape_loss_backward_and_support(self):
        m = self.model('shape')
        f = torch.rand(4, 7, device=self.device)
        f[:, 3:6] = f[:, 3:6] > .5
        loss, terms = reconstruction_loss(m, self.coords, f)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for p in m.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreaterEqual(terms['kl'].item(), 0)

    def test_material_posterior_and_decoupling(self):
        m = self.model('material')
        f = torch.rand(4, 6, device=self.device)
        p = m.encode(self.coords, f)
        self.assertEqual(p.mean.shape, (2, 4))
        torch.testing.assert_close(p.sample(False), p.mean)
        self.assertFalse(torch.equal(p.sample(), p.sample()))
        loss, _ = reconstruction_loss(m, self.coords, f, stochastic=False)
        self.assertTrue(torch.isfinite(loss))
        other = self.model('shape')
        self.assertFalse(set(map(id, m.parameters())) & set(map(id, other.parameters())))
        with self.assertRaises(ValueError):
            m.decode(p.coords, p.mean, max_active=1)

    def test_decoder_trace_records_pruned_parents(self):
        m=self.model('shape');features=torch.rand(4,7,device=self.device)
        posterior=m.encode(self.coords,features)
        teacher=[]
        out,_,_=m.decode(posterior.coords,posterior.mean,teacher_occupancy=posterior.occupancy,trace=teacher)
        self.assertEqual(len(teacher),4)
        for step,expected in zip(teacher,reversed(posterior.occupancy)):
            self.assertTrue(step['teacher_forced'])
            torch.testing.assert_close(step['selected'],expected)
        torch.testing.assert_close(teacher[-1]['children'],out)
        with torch.no_grad():
            for up in m.up:
                up.occupancy_head.weight.zero_();up.occupancy_head.bias.fill_(-100)
        trace=[];out,_,_=m.decode(posterior.coords,posterior.mean,trace=trace)
        self.assertEqual(len(trace),4)
        self.assertEqual(len(trace[0]['parents']),len(posterior.coords))
        self.assertEqual(len(trace[0]['children']),0)
        self.assertEqual(len(out),0)
        self.assertFalse(trace[0]['selected'].any())
        self.assertFalse(trace[0]['teacher_forced'])

    def test_coarse_weight_matches_first_decoder_level(self):
        m=self.model('shape');features=torch.rand(4,7,device=self.device)
        posterior,(_,_,logits)=m(self.coords,features,stochastic=False)
        expected=torch.nn.functional.binary_cross_entropy_with_logits(logits[0],posterior.occupancy[-1].to(logits[0].dtype))
        loss,parts=reconstruction_loss(m,self.coords,features,stochastic=False,occupancy_level_weights=[1,0,0,0])
        torch.testing.assert_close(parts['occupancy'],expected)
        _,default=reconstruction_loss(m,self.coords,features,stochastic=False)
        _,equal=reconstruction_loss(m,self.coords,features,stochastic=False,occupancy_level_weights=[2,2,2,2])
        torch.testing.assert_close(default['occupancy'],equal['occupancy'])
        loss.backward()
        self.assertTrue(torch.isfinite(m.up[0].occupancy_head.weight.grad).all())
        for weights in ([0,0,0,0],[1,2,3],[-1,1,1,1],[float('nan'),1,1,1]):
            with self.assertRaises(ValueError):reconstruction_loss(m,self.coords,features,stochastic=False,occupancy_level_weights=weights)
