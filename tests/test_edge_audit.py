import unittest
import numpy as np
from t3d.schema import VoxelAsset
from tools.audit_edge_predictions import counts


class EdgeAuditTests(unittest.TestCase):
    def test_union_of_coordinates_counts_missing_and_extra_edges(self):
        def asset(coords,flags):
            return VoxelAsset(coords,np.full((len(coords),3),.5),np.array(flags,bool),(4,4,4),[[0]*3,[1]*3])
        target=asset([[0,0,0],[1,0,0]],[[1,0,0],[1,1,0]])
        prediction=asset([[1,0,0],[2,0,0]],[[1,0,0],[1,0,1]])
        result=counts(target,prediction)
        self.assertEqual([(r['true_positive'],r['false_positive'],r['false_negative']) for r in result],
                         [(1,1,1),(0,0,1),(0,1,0)])
        self.assertIsNone(result[1]['precision'])
        self.assertIsNone(result[2]['recall'])
        self.assertEqual(result[0]['false_positive_outside_reference_support'],1)
        self.assertEqual(result[0]['false_negative_missing_support'],1)
        self.assertEqual(result[1]['false_negative_on_predicted_support'],1)
