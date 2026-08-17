from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MHA4Attention(nn.Module):
    """Four-head baseline with separately addressable Q/K/V projections."""

    def __init__(self, model_dim: int = 512, heads: int = 4) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.model_dim = model_dim
        self.heads = heads
        self.head_dim = model_dim // heads
        self.query = nn.Linear(model_dim, model_dim, bias=False)
        self.key = nn.Linear(model_dim, model_dim, bias=False)
        self.value = nn.Linear(model_dim, model_dim, bias=True)
        self.output = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape

        def split(projection: nn.Linear) -> Tensor:
            return projection(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)

        query = split(self.query)
        key = split(self.key)
        value = split(self.value)
        output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(batch, length, self.model_dim)
        return self.output(output)


class MHA4Block(nn.Module):
    def __init__(self, model_dim: int = 512, feedforward_dim: int = 1536) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(model_dim)
        self.attention = MHA4Attention(model_dim, 4)
        self.feedforward_norm = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, model_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feedforward(self.feedforward_norm(x))


class MHA4LM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 2048,
        model_dim: int = 512,
        num_layers: int = 6,
        feedforward_dim: int = 1536,
        max_sequence_length: int = 64,
    ) -> None:
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, model_dim)
        self.stack = nn.Module()
        self.stack.blocks = nn.ModuleList(
            MHA4Block(model_dim, feedforward_dim) for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        length = token_ids.shape[1]
        positions = torch.arange(length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)[None]
        for block in self.stack.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        return logits, loss
