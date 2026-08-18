from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


Backend = Literal["auto", "torch", "triton"]
MapCombine = Literal["weighted_sum", "concat"]
DictionaryActivation = Literal["identity", "silu"]
DictionaryAssignment = Literal["softmax", "silu"]


@dataclass(frozen=True)
class DoubleAttentionConfig:
    """Configuration for one shared-dictionary attention experiment.

    ``routing_dim`` is the width of *each* Q/K branch.  The total Q/K
    projection width is therefore ``routing_dim * qk_branches``.
    """

    model_dim: int = 512
    routing_dim: int = 256
    dictionary_size: int = 512
    qk_branches: int = 1
    outer_maps: int = 1
    beta: float = 4.0
    initial_score_scale: float = 16.0
    learnable_beta: bool = False
    dictionary_activation: DictionaryActivation = "identity"
    dictionary_assignment: DictionaryAssignment = "softmax"
    dictionary_silu_gain: float = 1.0
    standardize_dictionary_logits: bool = False
    standardized_logit_scale: float | None = None
    normalize_routing_input: bool = True
    normalize_routing_output: bool = True
    q_dictionary_feedforward: bool = False
    untied_dictionary: bool = True
    output_projection: bool = True
    value_bias: bool = True
    map_combine: MapCombine = "weighted_sum"
    causal: bool = True
    backend: Backend = "auto"
    eps: float = 1e-6

    def __post_init__(self) -> None:
        positive = {
            "model_dim": self.model_dim,
            "routing_dim": self.routing_dim,
            "dictionary_size": self.dictionary_size,
            "qk_branches": self.qk_branches,
            "outer_maps": self.outer_maps,
            "beta": self.beta,
            "initial_score_scale": self.initial_score_scale,
            "dictionary_silu_gain": self.dictionary_silu_gain,
            "eps": self.eps,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        if self.standardized_logit_scale is not None and self.standardized_logit_scale <= 0:
            raise ValueError("standardized_logit_scale must be positive when provided")
        larger = max(self.qk_branches, self.outer_maps)
        smaller = min(self.qk_branches, self.outer_maps)
        if larger % smaller:
            raise ValueError(
                "qk_branches and outer_maps must divide one another so score "
                "maps can be grouped or repeated without an ambiguous mapping"
            )
        if self.backend not in {"auto", "torch", "triton"}:
            raise ValueError(f"unsupported backend: {self.backend}")
        if self.map_combine not in {"weighted_sum", "concat"}:
            raise ValueError(f"unsupported map_combine: {self.map_combine}")
        if self.dictionary_activation not in {"identity", "silu"}:
            raise ValueError(
                f"unsupported dictionary activation: {self.dictionary_activation}"
            )
        if self.dictionary_assignment not in {"softmax", "silu"}:
            raise ValueError(
                f"unsupported dictionary assignment: {self.dictionary_assignment}"
            )
        if self.map_combine == "concat" and not self.output_projection:
            raise ValueError("concat map combination requires output_projection=True")

    @property
    def total_qk_width(self) -> int:
        return self.routing_dim * self.qk_branches

    def with_updates(self, **updates: object) -> "DoubleAttentionConfig":
        return replace(self, **updates)


_PRESETS: dict[
    str, tuple[int, int, int, DictionaryActivation, bool, float | None, bool, bool]
] = {
    # name: (Q/K projections, outer maps, routing width, activation,
    #        standardize logits, explicit pre-beta scale, input norm, output norm)
    "a1": (1, 1, 256, "identity", False, None, True, True),
    "a1-silu": (1, 1, 256, "silu", False, None, True, True),
    "a1-silu-logitnorm": (1, 1, 256, "silu", True, None, False, True),
    # beta=4 remains fixed, so a pre-beta scale of 0.25 gives softmax(z).
    "a1-silu-logitnorm-t1": (1, 1, 256, "silu", True, 0.25, False, True),
    "a1-r512-d512": (1, 1, 512, "identity", False, None, True, True),
    "a1-r512-d1024": (1, 1, 512, "identity", False, None, True, True),
    "a1-r512-d1536": (1, 1, 512, "identity", False, None, True, True),
    "a1-r512-d1536-qffn": (1, 1, 512, "identity", False, None, True, True),
    "a1-r512-d2855-qffn": (1, 1, 512, "identity", False, None, True, True),
    "a1-r512-d1764-qffn-l8": (1, 1, 512, "identity", False, None, True, True),
    "a1-d1536": (1, 1, 256, "identity", False, None, True, True),
    "a1-d1536-qffn": (1, 1, 256, "identity", False, None, True, True),
    "a1-no-softmax": (1, 1, 256, "identity", False, None, True, True),
    "a1-no-softmax-g4": (1, 1, 256, "identity", False, None, True, True),
    "a1-no-qnorm": (1, 1, 256, "identity", False, None, False, True),
    "a1-no-dpnorm": (1, 1, 256, "identity", False, None, True, False),
    "a1-no-norm": (1, 1, 256, "identity", False, None, False, False),
    "qk2-s1": (2, 1, 256, "identity", False, None, True, True),
    "qk1-s2": (1, 2, 256, "identity", False, None, True, True),
    "qk2-s2": (2, 2, 256, "identity", False, None, True, True),
    "qk4-s4": (4, 4, 128, "identity", False, None, True, True),
}

_PRESET_OVERRIDES: dict[str, dict[str, object]] = {
    # Preserve the baseline dictionary-logit and outer-attention variances
    # when routing width grows from 256 to 512.
    "a1-r512-d512": {
        "dictionary_size": 512,
        "beta": 4.0 * (2.0**0.5),
        "initial_score_scale": 512.0**0.5,
    },
    "a1-r512-d1024": {
        "dictionary_size": 1024,
        "beta": 4.0 * (2.0**0.5),
        "initial_score_scale": 512.0**0.5,
    },
    "a1-r512-d1536": {
        "dictionary_size": 1536,
        "beta": 4.0 * (2.0**0.5),
        "initial_score_scale": 512.0**0.5,
    },
    "a1-r512-d1536-qffn": {
        "dictionary_size": 1536,
        "beta": 4.0 * (2.0**0.5),
        "initial_score_scale": 512.0**0.5,
        "q_dictionary_feedforward": True,
    },
    "a1-r512-d2855-qffn": {
        "dictionary_size": 2855,
        "beta": 4.0 * (2.0**0.5),
        "initial_score_scale": 512.0**0.5,
        "q_dictionary_feedforward": True,
    },
    "a1-r512-d1764-qffn-l8": {
        "dictionary_size": 1764,
        "beta": 4.0 * (2.0**0.5),
        "initial_score_scale": 512.0**0.5,
        "q_dictionary_feedforward": True,
    },
    "a1-d1536": {"dictionary_size": 1536},
    "a1-d1536-qffn": {
        "dictionary_size": 1536,
        "q_dictionary_feedforward": True,
    },
    "a1-no-softmax": {
        "dictionary_assignment": "silu",
        "dictionary_silu_gain": 1.0,
    },
    "a1-no-softmax-g4": {
        "dictionary_assignment": "silu",
        "dictionary_silu_gain": 4.0,
    },
}


def experiment_config(name: str, **overrides: object) -> DoubleAttentionConfig:
    """Build one of the experiment variants recorded in the project.

    QK2 variants use two 256-wide branches (512 total Q/K width).  QK4-S4
    uses four 128-wide branches, matching the head width of MHA4 at d=512.
    """

    key = name.lower().replace("_", "-")
    try:
        (
            qk_branches,
            outer_maps,
            routing_dim,
            dictionary_activation,
            standardize_dictionary_logits,
            standardized_logit_scale,
            normalize_routing_input,
            normalize_routing_output,
        ) = _PRESETS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(_PRESETS))
        raise ValueError(f"unknown experiment {name!r}; choose one of: {choices}") from exc
    values: dict[str, object] = {
        "qk_branches": qk_branches,
        "outer_maps": outer_maps,
        "routing_dim": routing_dim,
        "dictionary_activation": dictionary_activation,
        "standardize_dictionary_logits": standardize_dictionary_logits,
        "standardized_logit_scale": standardized_logit_scale,
        "normalize_routing_input": normalize_routing_input,
        "normalize_routing_output": normalize_routing_output,
    }
    values.update(_PRESET_OVERRIDES.get(key, {}))
    values.update(overrides)
    return DoubleAttentionConfig(**values)


def experiment_names() -> tuple[str, ...]:
    return tuple(_PRESETS)
