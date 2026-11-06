import unittest
import numpy as np
from t3d.schema import VoxelAsset
from tools.audit_support_pyramid import compare


class SupportAuditTests(unittest.TestCase):
    def test_ancestor_support_and_empty_predictions(self):
        def asset(coords):
            c=np.asarray(coords,dtype=int).reshape(-1,3)
            return VoxelAsset(c,np.full(c.shape,.5),np.zeros(c.shape,bool),(16,)*3,[[0]*3,[1]*3])
        target=asset([[0,0,0],[1,0,0],[8,0,0]])
        pred=asset([[1,0,0],[2,0,0]])
        result=compare(target,pred)
        self.assertEqual((result[0]['true_positive'],result[0]['false_positive'],result[0]['false_negative']),(1,1,2))
        self.assertEqual((result[1]['true_positive'],result[1]['false_positive'],result[1]['false_negative']),(1,1,1))
        self.assertEqual((result[-1]['true_positive'],result[-1]['false_positive'],result[-1]['false_negative']),(1,0,0))
        self.assertIsNone(compare(target,asset([]))[0]['precision'])
