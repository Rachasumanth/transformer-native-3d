import unittest
try:
    import torch
    from t3d.skin_ae import SkinAE, skin_reconstruction_loss
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch required')
class SkinAETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SkinAE(width=16, heads=2, layers=1).to(self.device)
        self.joints = torch.tensor([[0.,0,0],[0,1,0],[1,1,0]], device=self.device)
        self.parents = torch.tensor([-1,0,1], device=self.device)
        self.weights = torch.tensor([[1.,0,0],[.2,.3,.5]], device=self.device)

    def test_weights_gradients_and_freeze(self):
        joint, vertex, logits = self.model(self.joints,self.parents,self.weights)
        self.assertEqual(tuple(joint.shape),(3,4));self.assertEqual(tuple(vertex.shape),(2,4))
        torch.testing.assert_close(vertex,self.weights@joint)
        torch.testing.assert_close(logits.softmax(-1).sum(-1),torch.ones(2,device=self.device))
        loss = skin_reconstruction_loss(logits,self.weights);loss.backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.model.freeze()
        self.assertFalse(self.model.training)
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))

    def test_joint_permutation_and_variable_count(self):
        self.model.eval()
        joint, vertex, logits = self.model(self.joints,self.parents,self.weights)
        order=torch.tensor([2,0,1],device=self.device);inverse=torch.argsort(order)
        parent=self.parents[order];parent=torch.where(parent<0,parent,inverse[parent.clamp_min(0)])
        other,v2,l2=self.model(self.joints[order],parent,self.weights[:,order])
        torch.testing.assert_close(other,joint[order],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(v2,vertex,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(l2,logits[:,order],atol=1e-6,rtol=1e-5)
        _,v,l=self.model(self.joints[:1],self.parents[:1],torch.ones(2,1,device=self.device))
        self.assertEqual(tuple(v.shape),(2,4));torch.testing.assert_close(l.softmax(-1),torch.ones_like(l))

    def test_invalid_rig_and_weights(self):
        with self.assertRaises(ValueError):
            self.model.encode_joints(self.joints,torch.tensor([1,0,-1],device=self.device))
        with self.assertRaises(ValueError):
            self.model(self.joints,self.parents,self.weights*2)
        with self.assertRaises(ValueError):
            skin_reconstruction_loss(torch.zeros_like(self.weights),self.weights*0)
