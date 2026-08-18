from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class LLDMConfig:
    """Configuration for a layer-local dictionary mixer.

    ``dictionary_size`` is the shared feature width and ``relational_dim`` is
    the rank of each token-interaction map.
    """

    model_dim: int = 512
    dictionary_size: int = 1024
    relational_dim: int = 128
    relational_maps: int = 2
    dropout: float = 0.0
    eps: float = 1e-6
    initial_context_scale: float = 0.1
    center_relational_assignments: bool = False
    assignment_scale: float = 0.25
    separate_context_output: bool = False
    independent_relational_readouts: bool = False
    independent_query_key: bool = False

    def __post_init__(self) -> None:
        for name in (
            "model_dim",
            "dictionary_size",
            "relational_dim",
            "relational_maps",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.initial_context_scale < 0:
            raise ValueError("initial_context_scale must be non-negative")
        if self.assignment_scale <= 0:
            raise ValueError("assignment_scale must be positive")


@dataclass
class LLDMAux:
    features: Tensor
    assignments: Tensor
    map_weights: Tensor
    score_scales: Tensor


class LayerLocalDictionaryMixerBlock(nn.Module):
    """One feature decomposition followed by local and relational mixing.

    The layer-local dictionary produces a single feature state ``z``.  The
    local path consumes ``z`` directly, while the relational path derives a
    calibrated simplex assignment from the same tensor.  Every relational map
    uses a low-rank feature projector in both directions.
    """

    def __init__(self, config: LLDMConfig) -> None:
        super().__init__()
        self.config = config
        d = config.model_dim
        m = config.dictionary_size
        r = config.relational_dim
        s = config.relational_maps

        self.input_norm = nn.RMSNorm(d, eps=config.eps)
        self.dictionary = nn.Linear(d, m, bias=False)
        if config.independent_relational_readouts:
            self.relational_projections = nn.ModuleList()
            self.query_projections = nn.ModuleList(
                nn.Linear(m, r, bias=False) for _ in range(s)
            )
            self.key_projections = nn.ModuleList(
                nn.Linear(m, r, bias=False) for _ in range(s)
            )
            self.value_projections = nn.ModuleList(
                nn.Linear(m, r, bias=False) for _ in range(s)
            )
            self.context_projections = nn.ModuleList(
                nn.Linear(r, d, bias=False) for _ in range(s)
            )
            self.query_gates = nn.ParameterList()
            self.key_gates = nn.ParameterList()
            self.value_gates = nn.ParameterList()
        else:
            self.relational_projections = nn.ModuleList(
                nn.Linear(m, r, bias=False) for _ in range(s)
            )
            self.query_projections = (
                nn.ModuleList(nn.Linear(m, r, bias=False) for _ in range(s))
                if config.independent_query_key
                else nn.ModuleList()
            )
            self.key_projections = (
                nn.ModuleList(nn.Linear(m, r, bias=False) for _ in range(s))
                if config.independent_query_key
                else nn.ModuleList()
            )
            self.value_projections = nn.ModuleList()
            self.context_projections = nn.ModuleList()
            self.query_gates = (
                nn.ParameterList()
                if config.independent_query_key
                else nn.ParameterList(nn.Parameter(torch.ones(m)) for _ in range(s))
            )
            self.key_gates = (
                nn.ParameterList()
                if config.independent_query_key
                else nn.ParameterList(nn.Parameter(torch.ones(m)) for _ in range(s))
            )
            self.value_gates = nn.ParameterList(
                nn.Parameter(torch.ones(m)) for _ in range(s)
            )

        initial_score_scale = math.log(math.sqrt(r))
        self.log_score_scales = nn.Parameter(torch.full((s,), initial_score_scale))
        self.map_mix_logits = nn.Parameter(torch.zeros(s))
        self.context_scale = nn.Parameter(torch.tensor(config.initial_context_scale))
        self.output = nn.Linear(m, d, bias=False)
        self.context_output = (
            nn.Linear(m, d, bias=False)
            if config.separate_context_output
            else None
        )
        self.dropout = nn.Dropout(config.dropout)

        # The default 1/4 matches the calibrated assignment-logit standard
        # deviation used by the existing dictionary route.
        self.register_buffer(
            "assignment_scale",
            torch.tensor(config.assignment_scale),
            persistent=True,
        )

    def _feature_state(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Rows of Linear.weight are dictionary atoms (columns of D in xD).
        atoms = F.normalize(self.dictionary.weight, dim=1, eps=self.config.eps)
        coefficients = F.linear(self.input_norm(x), atoms)
        features = 2.0 * F.silu(coefficients)

        centered = features - features.mean(dim=-1, keepdim=True)
        standardized = centered * torch.rsqrt(
            centered.square().mean(dim=-1, keepdim=True) + self.config.eps
        )
        assignments = torch.softmax(
            standardized * self.assignment_scale.to(standardized.dtype),
            dim=-1,
        )
        return features, assignments

    def forward(
        self,
        x: Tensor,
        *,
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, LLDMAux]:
        if x.ndim != 3 or x.shape[-1] != self.config.model_dim:
            raise ValueError(
                f"expected x with shape [B, T, {self.config.model_dim}], "
                f"got {tuple(x.shape)}"
            )

        features, assignments = self._feature_state(x)
        relational_assignments = assignments
        if self.config.center_relational_assignments:
            relational_assignments = assignments - assignments.mean(dim=-1, keepdim=True)
        queries: list[Tensor] = []
        keys: list[Tensor] = []
        values: list[Tensor] = []
        if self.config.independent_relational_readouts:
            # Removing the uniform simplex component exposes token-dependent
            # feature variation; independent Q/K maps avoid forcing one metric
            # to serve both sides of the attention score.
            relational_assignments = assignments - assignments.mean(dim=-1, keepdim=True)
            for query_projection, key_projection, value_projection in zip(
                self.query_projections,
                self.key_projections,
                self.value_projections,
                strict=True,
            ):
                queries.append(
                    F.normalize(
                        query_projection(relational_assignments),
                        dim=-1,
                        eps=self.config.eps,
                    )
                )
                keys.append(
                    F.normalize(
                        key_projection(relational_assignments),
                        dim=-1,
                        eps=self.config.eps,
                    )
                )
                values.append(value_projection(features))
        elif self.config.independent_query_key:
            relational_assignments = assignments - assignments.mean(dim=-1, keepdim=True)
            for index, (query_projection, key_projection, value_projection) in enumerate(
                zip(
                    self.query_projections,
                    self.key_projections,
                    self.relational_projections,
                    strict=True,
                )
            ):
                queries.append(
                    F.normalize(
                        query_projection(relational_assignments),
                        dim=-1,
                        eps=self.config.eps,
                    )
                )
                keys.append(
                    F.normalize(
                        key_projection(relational_assignments),
                        dim=-1,
                        eps=self.config.eps,
                    )
                )
                values.append(value_projection(features * self.value_gates[index]))
        else:
            for index, projection in enumerate(self.relational_projections):
                queries.append(
                    F.normalize(
                        projection(relational_assignments * self.query_gates[index]),
                        dim=-1,
                        eps=self.config.eps,
                    )
                )
                keys.append(
                    F.normalize(
                        projection(relational_assignments * self.key_gates[index]),
                        dim=-1,
                        eps=self.config.eps,
                    )
                )
                values.append(projection(features * self.value_gates[index]))

        query = torch.stack(queries, dim=1)
        key = torch.stack(keys, dim=1)
        value = torch.stack(values, dim=1)
        score_scales = self.log_score_scales.clamp(math.log(0.05), math.log(64.0)).exp()
        # SDPA applies 1/sqrt(r). Multiplying Q by scale*sqrt(r) yields the
        # desired learned scale on the dot product of unit-length Q and K.
        query = query * (
            score_scales.to(query.dtype)[None, :, None, None]
            * math.sqrt(self.config.relational_dim)
        )
        mixed = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=True,
        )

        map_weights = torch.softmax(self.map_mix_logits, dim=0).to(mixed.dtype)
        if self.config.independent_relational_readouts:
            contextual_output = torch.zeros_like(x)
            for index, projection in enumerate(self.context_projections):
                contextual_output = (
                    contextual_output
                    + map_weights[index] * projection(mixed[:, index])
                )
            output = (
                x
                + self.dropout(self.output(features))
                + self.dropout(
                    self.context_scale.to(features.dtype) * contextual_output
                )
            )
        else:
            context = torch.zeros_like(features)
            for index, projection in enumerate(self.relational_projections):
                # The same low-rank projection lifts relational content back
                # into the layer's feature coordinates.
                lifted = F.linear(mixed[:, index], projection.weight.transpose(0, 1))
                context = context + map_weights[index] * lifted

            scaled_context = self.context_scale.to(features.dtype) * context
            if self.context_output is None:
                output = x + self.dropout(self.output(features + scaled_context))
            else:
                output = (
                    x
                    + self.dropout(self.output(features))
                    + self.dropout(self.context_output(scaled_context))
                )
        if not return_aux:
            return output
        return output, LLDMAux(
            features=features,
            assignments=assignments,
            map_weights=map_weights,
            score_scales=score_scales,
        )


class LayerLocalDictionaryMixerStack(nn.Module):
    def __init__(self, config: LLDMConfig, num_layers: int) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.config = config
        self.blocks = nn.ModuleList(
            LayerLocalDictionaryMixerBlock(config) for _ in range(num_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class LayerLocalDictionaryMixerLM(nn.Module):
    """Causal language model using only LLDM blocks."""

    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        config: LLDMConfig | None = None,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        config = config or LLDMConfig()
        self.config = config
        self.max_sequence_length = max_sequence_length
        self.token_embedding = nn.Embedding(vocab_size, config.model_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, config.model_dim)
        self.stack = LayerLocalDictionaryMixerStack(config, num_layers)
        self.final_norm = nn.RMSNorm(config.model_dim, eps=config.eps)
        self.lm_head = nn.Linear(config.model_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [B, T]")
        length = token_ids.shape[1]
        if length > self.max_sequence_length:
            raise ValueError(
                f"sequence length {length} exceeds configured maximum "
                f"{self.max_sequence_length}"
            )
        positions = torch.arange(length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)[None]
        logits = self.lm_head(self.final_norm(self.stack(hidden)))
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets must match token_ids shape")
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        return logits, loss
