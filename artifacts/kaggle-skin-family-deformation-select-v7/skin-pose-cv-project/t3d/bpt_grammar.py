"""Incremental validity masks for the local 5120-token BPT fan grammar.

Masks enforce complete patches and nonzero quantized triangle area. They do
not ensure manifold topology, unique faces, geometric fidelity or coverage.
"""
import operator
import numpy as np
from .bpt import _coordinate

_OFFSETS = np.column_stack((np.arange(4096) // 256,
                            np.arange(4096) // 16 % 16, np.arange(4096) % 16))


class BPTGrammar:
    def __init__(self):
        self.block = None
        self.pending = False
        self.patch = []

    @property
    def can_end(self):
        return len(self.patch) >= 3 and not self.pending

    def allowed_tokens(self):
        allowed = np.zeros(5120, dtype=bool)
        if self.block is None:
            allowed[:512] = True
            return allowed
        if self.can_end:
            allowed[:512] = True
        if self.patch and not self.pending:
            allowed[512:1024] = True
        base = np.array([self.block // 64, self.block // 8 % 8, self.block % 8]) * 16
        candidates = base + _OFFSETS
        valid = np.ones(4096, dtype=bool)
        if self.patch:
            center = np.asarray(self.patch[0])
            valid &= np.any(candidates != center, axis=1)
        if len(self.patch) >= 2:
            previous = np.asarray(self.patch[-1])
            valid &= np.any(np.cross(previous - center, candidates - center) != 0, axis=1)
        allowed[1024:] = valid
        return allowed

    def advance(self, token):
        token = operator.index(token)
        if not 0 <= token < 5120 or not self.allowed_tokens()[token]:
            raise ValueError('Token violates incremental BPT grammar')
        if token < 512:
            self.patch = []
            self.block = token
            self.pending = True
        elif token < 1024:
            self.block = token - 512
            self.pending = True
        else:
            self.patch.append(_coordinate(self.block, token - 1024))
            self.pending = False
