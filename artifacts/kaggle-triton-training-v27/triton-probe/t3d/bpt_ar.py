"""Point-cloud-conditioned causal Transformer for the local BPT vocabulary.

The point encoder pools unordered XYZ samples into learned query tokens.
Training uses BOS + geometry as inputs and geometry + EOS as targets. PAD is
ignored by the loss. This baseline uses full causal attention within a bounded
context; Hourglass attention, sliding KV caches and DPO remain separate work.
"""
import torch
from torch import nn
from torch.nn import functional as F

VOCAB_SIZE = 5120
BOS, EOS, PAD = 5120, 5121, 5122


class CausalBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(),
                                 nn.Linear(width * 4, width))

    def forward(self, x, condition):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(
            batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.output(attended.transpose(1, 2).reshape(batch, length, width))
        cross, _ = self.cross(self.norm2(x), condition, condition, need_weights=False)
        x = x + cross
        return x + self.ffn(self.norm3(x))

    def step(self, x, condition, cache):
        """One new token attends to every cached token and itself."""
        batch, _, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(
            batch, 1, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        attended = F.scaled_dot_product_attention(q, k, v)
        x = x + self.output(attended.transpose(1, 2).reshape(batch, 1, width))
        cross, _ = self.cross(self.norm2(x), condition, condition, need_weights=False)
        x = x + cross
        return x + self.ffn(self.norm3(x)), (k, v)


class BPTAutoregressiveTransformer(nn.Module):
    def __init__(self, width=256, heads=8, depth=6, condition_tokens=32,
                 context_length=1024):
        super().__init__()
        if min(width, heads, depth, condition_tokens, context_length) <= 0 or width % heads:
            raise ValueError('Positive dimensions and width divisible by heads required')
        self.context_length = context_length
        self.token = nn.Embedding(VOCAB_SIZE + 3, width, padding_idx=PAD)
        self.position = nn.Embedding(context_length, width)
        self.point = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.queries = nn.Parameter(torch.randn(condition_tokens, width) * .02)
        self.point_pool = nn.MultiheadAttention(width, heads, batch_first=True)
        self.blocks = nn.ModuleList([CausalBlock(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, VOCAB_SIZE + 3, bias=False)

    def encode_points(self, points):
        if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] == 0:
            raise ValueError('Conditioning points must have shape (B, N, 3), N > 0')
        if not torch.isfinite(points).all():
            raise ValueError('Conditioning points must be finite')
        features = self.point(points)
        queries = self.queries[None].expand(len(points), -1, -1)
        return self.point_pool(queries, features, features, need_weights=False)[0]

    def forward(self, tokens, points, *, position_offset=0):
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError('Tokens must be a (B, T) torch.long tensor')
        if (not isinstance(position_offset, int) or position_offset < 0 or
                not 0 < tokens.shape[1] <= self.context_length - position_offset or len(tokens) != len(points)):
            raise ValueError('Batch mismatch or sequence exceeds model context')
        if torch.any(tokens < 0) or torch.any(tokens > PAD):
            raise ValueError('Token outside local vocabulary')
        condition = self.encode_points(points)
        x = self.token(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device) + position_offset)
        for block in self.blocks:
            x = block(x, condition)
        return self.head(self.norm(x))

    def loss(self, inputs, targets, points, *, position_offset=0):
        if targets.shape != inputs.shape or targets.dtype != torch.long:
            raise ValueError('Targets must match input shape and have integer token IDs')
        if not torch.any(targets != PAD):
            raise ValueError('Loss requires at least one non-padding target')
        logits = self(inputs, points, position_offset=position_offset)
        return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE + 3),
                               targets.reshape(-1), ignore_index=PAD)

    @torch.no_grad()
    def generate(self, points, *, max_tokens=None, temperature=0., generator=None,
                 grammar_constrained=False):
        """Generate one stream with per-layer KV caches; return explicit stop reason.

        Temperature zero is greedy. BOS/PAD cannot be sampled. Geometry tokens
        can optionally be constrained to complete, nondegenerate BPT patches.
        The caller must still validate the result using BPT decode.
        EOS is omitted from the returned stream. Context exhaustion is not EOS.
        """
        if self.training:
            raise ValueError('Call eval() before generation')
        if len(points) != 1 or temperature < 0:
            raise ValueError('Generation requires one cloud and nonnegative temperature')
        limit = self.context_length if max_tokens is None else max_tokens
        if not isinstance(limit, int) or not 0 < limit <= self.context_length:
            raise ValueError('Generation limit must fit the model context')
        condition = self.encode_points(points)
        from .bpt_grammar import BPTGrammar
        grammar = BPTGrammar() if grammar_constrained else None
        caches = [None] * len(self.blocks)
        token = torch.tensor([[BOS]], device=points.device)
        result = []
        for position in range(limit):
            x = self.token(token) + self.position.weight[position]
            for layer, block in enumerate(self.blocks):
                x, caches[layer] = block.step(x, condition, caches[layer])
            logits = self.head(self.norm(x))[0, 0].clone()
            logits[BOS] = logits[PAD] = -float('inf')
            if grammar is not None:
                allowed = torch.as_tensor(grammar.allowed_tokens(), device=logits.device)
                logits[:VOCAB_SIZE].masked_fill_(~allowed, -float('inf'))
                if not grammar.can_end:
                    logits[EOS] = -float('inf')
            if temperature == 0:
                next_token = logits.argmax()
            else:
                next_token = torch.multinomial((logits / temperature).softmax(-1),
                                               1, generator=generator)[0]
            value = int(next_token)
            if value == EOS:
                return dict(tokens=result, stop_reason='eos')
            result.append(value)
            if grammar is not None:
                grammar.advance(value)
            token = next_token.reshape(1, 1)
        return dict(tokens=result, stop_reason='context_limit' if limit == self.context_length
                    else 'token_limit')


def teacher_forcing_batch(sequences, device=None):
    """Right-pad complete streams; every active target predicts the next token."""
    if not sequences:
        raise ValueError('At least one geometry sequence required')
    streams = []
    for sequence in sequences:
        stream = torch.as_tensor(sequence, dtype=torch.long, device=device)
        if stream.ndim != 1 or stream.numel() == 0:
            raise ValueError('Geometry streams must be nonempty vectors')
        if torch.any(stream < 0) or torch.any(stream >= VOCAB_SIZE):
            raise ValueError('Geometry stream contains a reserved or invalid token')
        streams.append(stream)
    length = max(len(stream) for stream in streams) + 1
    inputs = torch.full((len(streams), length), PAD, dtype=torch.long, device=device)
    targets = inputs.clone()
    for row, stream in enumerate(streams):
        inputs[row, 0] = BOS
        inputs[row, 1:len(stream) + 1] = stream
        targets[row, :len(stream)] = stream
        targets[row, len(stream)] = EOS
    return inputs, targets
