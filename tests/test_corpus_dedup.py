import unittest
from tools.merge_kenney_shards import exact_deduplicate


class CorpusDedupTests(unittest.TestCase):
    def test_transitive_cross_split_groups_are_all_excluded(self):
        def row(name,split,source,prepared):return dict(asset_id=name,split=split,source_sha256=source,sha256=prepared)
        rows=[row('a','train','s1','p1'),row('b','train','s1','p2'),row('c','validation','s2','p2'),
              row('d','train','s3','p3'),row('e','train','s3','p4'),row('f','validation','s4','p5')]
        kept,rejected=exact_deduplicate(rows[::-1])
        self.assertEqual([r['asset_id'] for r in kept],['d','f'])
        self.assertEqual({r['asset_id'] for r in rejected},{'a','b','c','e'})
        self.assertTrue(all(r['reason']=='cross-split exact duplicate' for r in rejected if r['asset_id'] in ('a','b','c')))

    def test_empty(self):
        self.assertEqual(exact_deduplicate([]),([],[]))
