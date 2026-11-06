import unittest
import numpy as np
from tools.audit_decoder_trace import stage_counts


class DecoderTraceAuditTests(unittest.TestCase):
    def test_distinguishes_pruned_parent_from_local_miss(self):
        parents=np.array([[0,0,0]])
        children=np.array([[0,0,0],[1,1,1]])
        source=np.array([[0,0,0],[1,0,0],[6,0,0]])
        r=stage_counts(parents,children,source,1)
        self.assertEqual((r['true_positive'],r['false_positive'],r['false_negative']),(1,1,2))
        self.assertEqual((r['missed_unreachable'],r['missed_reachable']),(1,1))
