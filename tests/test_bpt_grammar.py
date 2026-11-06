import unittest
import numpy as np
from t3d.bpt import decode, encode, quantize
from t3d.bpt_grammar import BPTGrammar


class BPTGrammarTests(unittest.TestCase):
    def test_encoded_fan_is_accepted(self):
        vertices = np.array([[0, 0, 0], [.4, 0, 0], [0, .4, 0], [0, 0, .4]])
        tokens = encode(vertices, [[0, 1, 2], [0, 2, 3]])
        grammar = BPTGrammar()
        for token in tokens:
            self.assertTrue(grammar.allowed_tokens()[token])
            grammar.advance(token)
        self.assertTrue(grammar.can_end)

    def test_pending_blocks_and_triangle_area(self):
        g = BPTGrammar()
        self.assertFalse(g.can_end)
        g.advance(0)
        with self.assertRaises(ValueError):
            g.advance(1)
        g.advance(1024)  # center (0,0,0)
        with self.assertRaises(ValueError):
            g.advance(1024)
        g.advance(1025)  # first peripheral (0,0,1)
        self.assertFalse(g.can_end)
        with self.assertRaises(ValueError):
            g.advance(1026)  # collinear
        g.advance(1040)  # (0,1,0)
        self.assertTrue(g.can_end)
        g.advance(512)
        self.assertFalse(g.can_end)
        with self.assertRaises(ValueError):
            g.advance(0)

    def test_random_accepted_streams_decode_with_nonzero_area(self):
        rng = np.random.default_rng(9)
        for _ in range(20):
            grammar = BPTGrammar()
            tokens = []
            for step in range(200):
                token = int(rng.choice(np.flatnonzero(grammar.allowed_tokens())))
                grammar.advance(token)
                tokens.append(token)
                if grammar.can_end and step > 20:
                    break
            self.assertTrue(grammar.can_end)
            v, f = decode(np.asarray(tokens))
            q = quantize(v)
            area = np.cross(q[f[:, 1]] - q[f[:, 0]], q[f[:, 2]] - q[f[:, 0]])
            self.assertTrue(np.any(area != 0, axis=1).all())


if __name__ == '__main__':
    unittest.main()
