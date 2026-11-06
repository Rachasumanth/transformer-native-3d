import unittest
from types import SimpleNamespace
from tools.train_kenney_corpus_32 import select_eval_subset


class KenneyCorpusEvalTests(unittest.TestCase):
    def test_select_eval_subset_balance_and_determinism(self):
        records = [
            SimpleNamespace(source_group='group-a', asset_id=f'a-{i}') for i in range(20)
        ] + [
            SimpleNamespace(source_group='group-b', asset_id=f'b-{i}') for i in range(30)
        ]
        dataset = SimpleNamespace(records=records)

        sel1 = select_eval_subset(dataset, per_group=5, seed=42)
        sel2 = select_eval_subset(dataset, per_group=5, seed=42)

        self.assertEqual(sel1, sel2)
        self.assertEqual(len(sel1), 10)

        # 5 from group-a (indices 0..19), 5 from group-b (indices 20..49)
        group_a_selected = [i for i in sel1 if i < 20]
        group_b_selected = [i for i in sel1 if i >= 20]
        self.assertEqual(len(group_a_selected), 5)
        self.assertEqual(len(group_b_selected), 5)

    def test_select_eval_subset_capped_by_available(self):
        records = [
            SimpleNamespace(source_group='small-group', asset_id=f's-{i}') for i in range(3)
        ]
        dataset = SimpleNamespace(records=records)
        sel = select_eval_subset(dataset, per_group=10, seed=42)
        self.assertEqual(len(sel), 3)
