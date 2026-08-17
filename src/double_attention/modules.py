from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import DoubleAttentionConfig
from .ops import (
    assemble_score_maps,
    dictionary_route_reference,
    resolve_backend,
    routed_attention_reference,
    score_scale_from_log,
)
from .triton_kernels import dictionary_route_triton, routed_attention_triton


class SharedDictionaryBank(nn.Module):
    """Globally or stage-wise shared routing dictionary.

    The attached d=512 experiment used separate assignment and reconstruction
    atoms.  Set ``untied=False`` for the simpler single-D formulation from the
    project report.
    """

    def __init__(self, routing_dim: int, dictionary_size: int, *, untied: bool = True) -> None:
        super().__init__()
        self.routing_dim = routing_dim
        self.dictionary_size = dictionary_size
        self.untied = untied
        std = routing_dim**-0.5
        self.raw_key = nn.Parameter(torch.randn(routing_dim, dictionary_size) * std)
        if untied:
            self.raw_value = nn.Parameter(torch.randn(routing_dim, dictionary_size) * std)
        else:
            self.register_parameter("raw_value", None)

    def normalized(self, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
        key = F.normalize(self.raw_key, dim=0, eps=eps)
        value = key if self.raw_value is None else F.normalize(self.raw_value, dim=0, eps=eps)
        return key, value

    def extra_repr(self) -> str:
        return (
            f"routing_dim={self.routing_dim}, dictionary_size={self.dictionary_size}, "
            f"untied={self.untied}"
        )


@dataclass
class DoubleAttentionAux:
    routing_query: Tensor
    routing_key: Tensor
    score_query: Tensor
    score_key: Tensor
    score_scales: Tensor
    map_weights: Tensor
    backend: str


class SharedDictionaryAttention(nn.Module):
    """Content/routing-decoupled attention with experiment-factorized maps."""

    def __init__(
        self,
        config: DoubleAttentionConfig,
        *,
        bank: SharedDictionaryBank | None = None,
        create_bank: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        width = config.qk_branches * config.routing_dim
        self.query = nn.Linear(config.model_dim, width, bias=False)
        self.key = nn.Linear(config.model_dim, width, bias=False)
        self.value = nn.Linear(config.model_dim, config.model_dim, bias=config.value_bias)
        self.bank = (
            bank
            if bank is not None
            else (
                SharedDictionaryBank(
                    config.routing_dim,
                    config.dictionary_size,
                    untied=config.untied_dictionary,
                )
                if create_bank
                else None
            )
        )

        initial_log_scale = math.log(config.initial_score_scale)
        if config.outer_maps == 1:
            initial_scales = torch.tensor([initial_log_scale])
        else:
            # QK1-S2 would remain exactly symmetric if both maps started with
            # the same score temperature.  A tiny deterministic offset gives
            # the two outer softmaxes a degree of freedom without adding Q/K.
            initial_scales = initial_log_scale + torch.linspace(-0.05, 0.05, config.outer_maps)
        self.log_score_scales = nn.Parameter(initial_scales)
        if config.learnable_beta:
            self.log_beta = nn.Parameter(torch.tensor(math.log(config.beta)))
        else:
            self.register_buffer("log_beta", torch.tensor(math.log(config.beta)), persistent=True)

        if config.map_combine == "weighted_sum":
            if config.outer_maps > 1:
                self.map_mix_logits = nn.Parameter(torch.zeros(config.outer_maps))
            else:
                self.register_parameter("map_mix_logits", None)
            self.output = (
                nn.Linear(config.model_dim, config.model_dim, bias=False)
                if config.output_projection
                else nn.Identity()
            )
        else:
            self.register_parameter("map_mix_logits", None)
            self.output = nn.Linear(
                config.outer_maps * config.model_dim,
                config.model_dim,
                bias=False,
            )

    @property
    def beta(self) -> Tensor:
        return self.log_beta.clamp(math.log(0.05), math.log(64.0)).exp()

    def _project_branches(self, projection: nn.Linear, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        result = projection(x)
        result = result.view(batch, length, self.config.qk_branches, self.config.routing_dim)
        return result.permute(0, 2, 1, 3).contiguous()

    def _route(self, x: Tensor, dictionary_key: Tensor, dictionary_value: Tensor, backend: str) -> Tensor:
        if backend == "triton":
            return dictionary_route_triton(
                x,
                dictionary_key,
                dictionary_value,
                self.beta,
                self.config.eps,
            )
        return dictionary_route_reference(
            x,
            dictionary_key,
            dictionary_value,
            self.beta,
            self.config.eps,
        )

    def forward(
        self,
        x: Tensor,
        *,
        bank: SharedDictionaryBank | None = None,
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, DoubleAttentionAux]:
        if x.ndim != 3 or x.shape[-1] != self.config.model_dim:
            raise ValueError(
                f"expected x with shape [B, T, {self.config.model_dim}], got {tuple(x.shape)}"
            )
        active_bank = bank if bank is not None else self.bank
        if active_bank is None:
            raise ValueError("no dictionary bank supplied")
        if (
            active_bank.routing_dim != self.config.routing_dim
            or active_bank.dictionary_size != self.config.dictionary_size
        ):
            raise ValueError("dictionary bank dimensions do not match the attention config")

        projected_query = self._project_branches(self.query, x)
        projected_key = self._project_branches(self.key, x)
        dense_value = self.value(x)
        dictionary_key, dictionary_value = active_bank.normalized(self.config.eps)
        # AMP keeps the master dictionary parameters in fp32 while linear
        # projections become fp16/bf16.  Cast the normalized compute views so
        # Triton/cuBLAS receive one dtype; gradients still flow to fp32 masters.
        if dictionary_key.dtype != projected_query.dtype:
            dictionary_key = dictionary_key.to(projected_query.dtype)
            dictionary_value = dictionary_value.to(projected_query.dtype)
        backend = resolve_backend(
            self.config.backend,
            (projected_query, projected_key, dense_value, dictionary_key, dictionary_value),
        )
        # The project experiment concatenated Q and K tokens before the two
        # dictionary GEMMs.  This is mathematically identical to two calls and
        # gives both cuBLAS and the Triton row kernels a larger launch.
        projected_pair = torch.cat((projected_query, projected_key), dim=2)
        routed_pair = self._route(projected_pair, dictionary_key, dictionary_value, backend)
        routed_query, routed_key = routed_pair.split(x.shape[1], dim=2)
        score_query, score_key = assemble_score_maps(
            routed_query,
            routed_key,
            self.config.outer_maps,
        )
        score_scales = score_scale_from_log(self.log_score_scales)
        if backend == "triton":
            map_output = routed_attention_triton(
                score_query,
                score_key,
                dense_value,
                score_scales,
                self.config.causal,
            )
        else:
            map_output = routed_attention_reference(
                score_query,
                score_key,
                dense_value,
                score_scales,
                self.config.causal,
            )

        if self.config.map_combine == "weighted_sum":
            if self.map_mix_logits is None:
                map_weights = x.new_ones(1)
                combined = map_output[:, 0]
            else:
                map_weights = torch.softmax(self.map_mix_logits, dim=0).to(map_output.dtype)
                combined = torch.einsum("s,bstd->btd", map_weights, map_output)
            output = self.output(combined)
        else:
            map_weights = x.new_full(
                (self.config.outer_maps,),
                1.0 / self.config.outer_maps,
            )
            concatenated = map_output.permute(0, 2, 1, 3).flatten(2)
            output = self.output(concatenated)

        if not return_aux:
            return output
        return output, DoubleAttentionAux(
            routing_query=routed_query,
            routing_key=routed_key,
            score_query=score_query,
            score_key=score_key,
            score_scales=score_scales,
            map_weights=map_weights,
            backend=backend,
        )


class DoubleAttentionBlock(nn.Module):
    def __init__(
        self,
        config: DoubleAttentionConfig,
        *,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        feedforward_dim = feedforward_dim or 3 * config.model_dim
        self.attention_norm = nn.LayerNorm(config.model_dim)
        self.attention = SharedDictionaryAttention(config, create_bank=False)
        self.feedforward_norm = nn.LayerNorm(config.model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(config.model_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, config.model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, bank: SharedDictionaryBank) -> Tensor:
        x = x + self.attention(self.attention_norm(x), bank=bank)
        return x + self.feedforward(self.feedforward_norm(x))


class DoubleAttentionStack(nn.Module):
    """Transformer blocks with configurable global or stage-wise dictionaries."""

    def __init__(
        self,
        config: DoubleAttentionConfig,
        num_layers: int,
        *,
        dictionary_group_size: int | None = None,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        group_size = dictionary_group_size or num_layers
        if group_size <= 0:
            raise ValueError("dictionary_group_size must be positive")
        self.config = config
        self.num_layers = num_layers
        self.dictionary_group_size = group_size
        bank_count = math.ceil(num_layers / group_size)
        self.banks = nn.ModuleList(
            SharedDictionaryBank(
                config.routing_dim,
                config.dictionary_size,
                untied=config.untied_dictionary,
            )
            for _ in range(bank_count)
        )
        self.blocks = nn.ModuleList(
            DoubleAttentionBlock(
                config,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        for index, block in enumerate(self.blocks):
            bank_index = min(index // self.dictionary_group_size, len(self.banks) - 1)
            x = block(x, self.banks[bank_index])
        return x


class DoubleAttentionLM(nn.Module):
    """Small causal LM wrapper for the 8k/12k screening experiments."""

    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        config: DoubleAttentionConfig,
        num_layers: int,
        *,
        dictionary_group_size: int | None = None,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.token_embedding = nn.Embedding(vocab_size, config.model_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, config.model_dim)
        self.stack = DoubleAttentionStack(
            config,
            num_layers,
            dictionary_group_size=dictionary_group_size,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.final_norm = nn.LayerNorm(config.model_dim)
        self.lm_head = nn.Linear(config.model_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [B, T]")
        length = token_ids.shape[1]
        if length > self.max_sequence_length:
            raise ValueError(
                f"sequence length {length} exceeds configured maximum {self.max_sequence_length}"
            )
        positions = torch.arange(length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)[None]
        logits = self.lm_head(self.final_norm(self.stack(hidden)))
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets must match token_ids shape")
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss
