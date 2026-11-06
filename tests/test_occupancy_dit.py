import unittest
import numpy as np
import torch
from types import SimpleNamespace
from tools.generate_corpus_prompts import id_to_prompt
from tools.train_kenney_occupancy_dit import build_coarse_occupancy


class OccupancyDiTTests(unittest.TestCase):
    def test_id_to_prompt_varieties(self):
        self.assertEqual(id_to_prompt('kenney-car-kit-ambulance'), 'an ambulance vehicle asset')
        self.assertEqual(id_to_prompt('kenney-food-kit-advocado-half'), 'an advocado half food item')
        self.assertEqual(id_to_prompt('kenney-nature-kit-tree-thin-fall'), 'a tree thin fall nature environment asset')
        self.assertEqual(id_to_prompt('kenney-pirate-kit-ship-medium'), 'a ship medium pirate theme asset')
        self.assertEqual(id_to_prompt('kenney-space-kit-alien'), 'an alien sci-fi space asset')

    def test_build_coarse_occupancy_downsampling(self):
        # 32³ grid points downsampled to 16³
        # Point (0, 0, 0) and (1, 1, 1) both map to coarse (0, 0, 0)
        coords = np.array([[0, 0, 0], [1, 1, 1], [30, 31, 31]], dtype=int)
        asset = SimpleNamespace(coords=coords)
        occ = build_coarse_occupancy(asset, coarse_grid=16, input_grid=32)

        self.assertEqual(occ.shape, (16 ** 3, 1))
        # (0, 0, 0) should be occupied
        self.assertEqual(occ[0, 0].item(), 1.0)
        # (15, 15, 15) should be occupied (30//2=15, 31//2=15)
        last_idx = (15 * 16 + 15) * 16 + 15
        self.assertEqual(occ[last_idx, 0].item(), 1.0)
        # Total occupied cells should be exactly 2
        self.assertEqual(occ.sum().item(), 2.0)
