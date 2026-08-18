from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import Backend, DictionaryActivation, DictionaryAssignment

if TYPE_CHECKING:
    from collections.abc import Sequence


def dictionary_route_reference(
    x: Tensor,
    dictionary_key: Tensor,
    dictionary_value: Tensor,
    beta: Tensor | float,
    eps: float = 1e-6,
    activation: DictionaryActivation = "identity",
    assignment: DictionaryAssignment = "softmax",
    silu_gain: float = 1.0,
    standardize_logits: bool = False,
    standardized_logit_scale: float | None = None,
    normalize_input: bool = True,
    normalize_output: bool = True,
) -> Tensor:
    """Reference implementation of the learned dictionary feature map.

    Args:
        x: ``[..., routing_dim]`` projected query or key vectors.
        dictionary_key: Normalized assignment dictionary ``[routing_dim, M]``.
        dictionary_value: Normalized reconstruction dictionary ``[routing_dim, M]``.
        beta: Positive soft-assignment inverse temperature.
    """

    routed_input = F.normalize(x, dim=-1, eps=eps) if normalize_input else x
    logits = routed_input @ dictionary_key
    if activation == "silu":
        # The factor two preserves unit slope around zero: 2 SiLU(x) ~= x.
        logits = 2.0 * F.silu(logits)
    elif activation != "identity":
        raise ValueError(f"unsupported dictionary activation: {activation}")
    if standardize_logits:
        centered = logits - logits.mean(dim=-1, keepdim=True)
        inverse_std = torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + eps)
        logit_scale = (
            x.shape[-1] ** -0.5
            if standardized_logit_scale is None
            else standardized_logit_scale
        )
        logits = centered * inverse_std * logit_scale
    if assignment == "softmax":
        weights = torch.softmax(logits * beta, dim=-1)
    elif assignment == "silu":
        weights = (2.0 / silu_gain) * F.silu(silu_gain * logits)
    else:
        raise ValueError(f"unsupported dictionary assignment: {assignment}")
    reconstructed = weights @ dictionary_value.transpose(0, 1)
    return F.normalize(reconstructed, dim=-1, eps=eps) if normalize_output else reconstructed


def routed_attention_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scales: Tensor,
    causal: bool = True,
) -> Tensor:
    """Attention with low-dimensional routing and a shared dense value.

    ``query`` and ``key`` are ``[B, S, T, R]``.  ``value`` is ``[B, T, D]``
    and is deliberately not split or compressed per score map.
    """

    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("query and key must have matching [B, S, T, R] shapes")
    if value.ndim != 3 or value.shape[:2] != (query.shape[0], query.shape[2]):
        raise ValueError("value must have shape [B, T, D] matching query")
    if scales.shape != (query.shape[1],):
        raise ValueError(f"scales must have shape [{query.shape[1]}]")

    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores.float() * scales.float().view(1, -1, 1, 1)
    if causal:
        length = query.shape[-2]
        mask = torch.ones(length, length, dtype=torch.bool, device=query.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1).to(value.dtype)
    expanded_value = value[:, None].expand(-1, query.shape[1], -1, -1)
    return torch.matmul(probabilities, expanded_value)


def assemble_score_maps(
    query: Tensor,
    key: Tensor,
    outer_maps: int,
) -> tuple[Tensor, Tensor]:
    """Map Q/K branches to independently normalized outer attention maps.

    * More Q/K branches than maps: concatenate equal branch groups.  The
      vectors are scaled so the resulting dot product is the RMS-normalized
      sum of branch scores (QK2-S1).
    * More maps than Q/K branches: repeat each branch.  Independent learned
      score temperatures then isolate the effect of multiple softmaxes
      (QK1-S2).
    """

    if query.ndim != 4 or query.shape != key.shape:
        raise ValueError("query and key must have matching [B, H, T, R] shapes")
    branches = query.shape[1]
    if branches == outer_maps:
        return query, key

    if branches > outer_maps:
        if branches % outer_maps:
            raise ValueError("qk_branches must be divisible by outer_maps")
        group = branches // outer_maps
        # Scaling both operands by group**(-1/4) makes their dot product the
        # sum of branch scores divided by sqrt(group).
        operand_scale = group ** -0.25
        q_parts: list[Tensor] = []
        k_parts: list[Tensor] = []
        for index in range(outer_maps):
            start = index * group
            stop = start + group
            q_parts.append(query[:, start:stop].movedim(1, 2).flatten(-2) * operand_scale)
            k_parts.append(key[:, start:stop].movedim(1, 2).flatten(-2) * operand_scale)
        return torch.stack(q_parts, dim=1), torch.stack(k_parts, dim=1)

    if outer_maps % branches:
        raise ValueError("outer_maps must be divisible by qk_branches")
    repeats = outer_maps // branches
    return (
        query.repeat_interleave(repeats, dim=1),
        key.repeat_interleave(repeats, dim=1),
    )


def resolve_backend(backend: Backend, tensors: "Sequence[Tensor]") -> Backend:
    if backend == "torch":
        return "torch"

    from .triton_kernels import TRITON_AVAILABLE

    supported = (
        TRITON_AVAILABLE
        and bool(tensors)
        and all(t.is_cuda for t in tensors)
        and all(t.dtype in {torch.float16, torch.bfloat16} for t in tensors)
    )
    if backend == "triton" and not supported:
        reason = "Triton is not installed" if not TRITON_AVAILABLE else "CUDA fp16/bf16 tensors are required"
        raise RuntimeError(f"Triton backend requested but unavailable: {reason}")
    return "triton" if supported else "torch"


def score_scale_from_log(log_scale: Tensor) -> Tensor:
    return log_scale.clamp(math.log(0.25), math.log(1024.0)).exp()
