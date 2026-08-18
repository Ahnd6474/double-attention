from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F

import double_attention.modules as attention_modules
from double_attention import (
    DoubleAttentionConfig,
    DoubleAttentionBlock,
    DoubleAttentionLM,
    DoubleAttentionStack,
    MHA4LM,
    SharedDictionaryAttention,
    experiment_config,
    experiment_names,
)
from double_attention.ops import dictionary_route_reference


def small_config(name: str, **overrides: object) -> DoubleAttentionConfig:
    values: dict[str, object] = {
        "model_dim": 32,
        "routing_dim": 8,
        "dictionary_size": 16,
        "output_projection": True,
        "backend": "torch",
    }
    values.update(overrides)
    return experiment_config(name, **values)


@pytest.mark.parametrize("name", experiment_names())
def test_all_experiment_variants_forward_and_backward(name: str) -> None:
    torch.manual_seed(7)
    module = SharedDictionaryAttention(small_config(name))
    x = torch.randn(2, 7, 32, requires_grad=True)
    output, aux = module(x, return_aux=True)

    assert output.shape == x.shape
    assert aux.routing_query.shape[:3] == (2, module.config.qk_branches, 7)
    assert aux.score_query.shape[:3] == (2, module.config.outer_maps, 7)
    assert aux.map_weights.shape == (module.config.outer_maps,)
    assert aux.backend == "torch"

    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert module.query.weight.grad is not None
    assert module.key.weight.grad is not None
    assert module.bank is not None
    assert module.bank.raw_key.grad is not None


@pytest.mark.parametrize(
    ("name", "gain"),
    [("a1-no-softmax", 1.0), ("a1-no-softmax-g4", 4.0)],
)
def test_direct_silu_assignment_matches_formula(name: str, gain: float) -> None:
    torch.manual_seed(19)
    config = small_config(name)
    x = torch.randn(2, 5, config.routing_dim)
    dictionary = F.normalize(
        torch.randn(config.routing_dim, config.dictionary_size), dim=0
    )

    actual = dictionary_route_reference(
        x,
        dictionary,
        dictionary,
        config.beta,
        assignment=config.dictionary_assignment,
        silu_gain=config.dictionary_silu_gain,
    )
    normalized_x = F.normalize(x, dim=-1)
    coefficients = (2.0 / gain) * F.silu(gain * (normalized_x @ dictionary))
    expected = F.normalize(coefficients @ dictionary.T, dim=-1)

    assert config.dictionary_assignment == "silu"
    assert config.dictionary_silu_gain == gain
    assert (coefficients < 0).any()
    torch.testing.assert_close(actual, expected)


def test_qk2_s1_concatenates_two_score_geometries() -> None:
    module = SharedDictionaryAttention(small_config("qk2-s1"))
    _, aux = module(torch.randn(1, 5, 32), return_aux=True)
    assert aux.routing_query.shape == (1, 2, 5, 8)
    assert aux.score_query.shape == (1, 1, 5, 16)


def test_fused_query_key_projection_preserves_outputs_gradients_and_state_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(8)
    module = SharedDictionaryAttention(small_config("qk2-s2"))
    assert "query.weight" in module.state_dict()
    assert "key.weight" in module.state_dict()
    assert not any("fused" in name for name in module.state_dict())
    x = torch.randn(2, 7, 32, requires_grad=True)

    monkeypatch.setattr(attention_modules, "FUSED_QUERY_KEY_PROJECTION", False)
    separate = module(x)
    separate.square().mean().backward()
    separate_gradients = (
        x.grad.clone(),
        module.query.weight.grad.clone(),
        module.key.weight.grad.clone(),
    )

    module.zero_grad(set_to_none=True)
    x.grad = None
    monkeypatch.setattr(attention_modules, "FUSED_QUERY_KEY_PROJECTION", True)
    fused = module(x)
    fused.square().mean().backward()

    torch.testing.assert_close(fused, separate, atol=1e-6, rtol=1e-5)
    for expected, actual in zip(
        separate_gradients,
        (x.grad, module.query.weight.grad, module.key.weight.grad),
        strict=True,
    ):
        assert actual is not None
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_qk1_s2_repeats_geometry_but_breaks_temperature_symmetry() -> None:
    module = SharedDictionaryAttention(small_config("qk1-s2"))
    _, aux = module(torch.randn(1, 5, 32), return_aux=True)
    torch.testing.assert_close(aux.score_query[:, 0], aux.score_query[:, 1])
    assert aux.score_scales[0] != aux.score_scales[1]


def test_a1_silu_activates_dictionary_logits_before_softmax() -> None:
    torch.manual_seed(19)
    x = torch.randn(2, 3, 8)
    dictionary = F.normalize(torch.randn(8, 16), dim=0)
    beta = torch.tensor(4.0)

    actual = dictionary_route_reference(x, dictionary, dictionary, beta, activation="silu")
    normalized = F.normalize(x, dim=-1)
    logits = 2.0 * F.silu(normalized @ dictionary)
    expected = F.normalize(torch.softmax(beta * logits, dim=-1) @ dictionary.T, dim=-1)

    torch.testing.assert_close(actual, expected)
    assert experiment_config("a1-silu").dictionary_activation == "silu"


def test_a1_silu_logitnorm_standardizes_after_activation() -> None:
    torch.manual_seed(23)
    x = torch.randn(2, 3, 8)
    dictionary = F.normalize(torch.randn(8, 16), dim=0)
    beta = torch.tensor(4.0)

    actual = dictionary_route_reference(
        x,
        dictionary,
        dictionary,
        beta,
        activation="silu",
        standardize_logits=True,
        normalize_input=False,
    )
    activated = 2.0 * F.silu(x @ dictionary)
    centered = activated - activated.mean(dim=-1, keepdim=True)
    calibrated = centered * torch.rsqrt(
        centered.square().mean(dim=-1, keepdim=True) + 1e-6
    ) * (x.shape[-1] ** -0.5)
    expected = F.normalize(
        torch.softmax(beta * calibrated, dim=-1) @ dictionary.T,
        dim=-1,
    )

    torch.testing.assert_close(actual, expected)
    config = experiment_config("a1-silu-logitnorm")
    assert config.dictionary_activation == "silu"
    assert config.standardize_dictionary_logits
    assert not config.normalize_routing_input
    assert config.normalize_routing_output


def test_a1_silu_logitnorm_t1_uses_unit_softmax_scale() -> None:
    torch.manual_seed(29)
    x = torch.randn(2, 3, 8)
    dictionary = F.normalize(torch.randn(8, 16), dim=0)
    beta = torch.tensor(4.0)

    actual = dictionary_route_reference(
        x,
        dictionary,
        dictionary,
        beta,
        activation="silu",
        standardize_logits=True,
        standardized_logit_scale=0.25,
        normalize_input=False,
    )
    activated = 2.0 * F.silu(x @ dictionary)
    centered = activated - activated.mean(dim=-1, keepdim=True)
    standardized = centered * torch.rsqrt(
        centered.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    expected = F.normalize(
        torch.softmax(standardized, dim=-1) @ dictionary.T,
        dim=-1,
    )

    torch.testing.assert_close(actual, expected)
    config = experiment_config("a1-silu-logitnorm-t1")
    assert config.standardized_logit_scale == 0.25


@pytest.mark.parametrize(
    ("name", "dictionary_size"),
    [
        ("a1-r512-d512", 512),
        ("a1-r512-d1024", 1024),
        ("a1-r512-d1536", 1536),
        ("a1-r512-d1536-qffn", 1536),
        ("a1-r512-d2855-qffn", 2855),
        ("a1-r512-d1764-qffn-l8", 1764),
    ],
)
def test_a1_r512_presets_calibrate_temperature(
    name: str, dictionary_size: int
) -> None:
    config = experiment_config(name)
    assert config.routing_dim == 512
    assert config.dictionary_size == dictionary_size
    assert config.beta == pytest.approx(4.0 * (2.0**0.5))
    assert config.initial_score_scale == pytest.approx(512.0**0.5)


def test_d1536_presets_keep_routing_width_and_select_q_ffn() -> None:
    control = experiment_config("a1-d1536")
    q_ffn = experiment_config("a1-d1536-qffn")
    full_rank_control = experiment_config("a1-r512-d1536")
    full_rank_q_ffn = experiment_config("a1-r512-d1536-qffn")
    matched_q_ffn = experiment_config("a1-r512-d2855-qffn")
    deep_matched_q_ffn = experiment_config("a1-r512-d1764-qffn-l8")
    assert control.routing_dim == q_ffn.routing_dim == 256
    assert control.dictionary_size == q_ffn.dictionary_size == 1536
    assert not control.q_dictionary_feedforward
    assert q_ffn.q_dictionary_feedforward
    assert full_rank_control.routing_dim == full_rank_q_ffn.routing_dim == 512
    assert full_rank_control.dictionary_size == full_rank_q_ffn.dictionary_size == 1536
    assert not full_rank_control.q_dictionary_feedforward
    assert full_rank_q_ffn.q_dictionary_feedforward
    assert matched_q_ffn.routing_dim == 512
    assert matched_q_ffn.dictionary_size == 2855
    assert matched_q_ffn.q_dictionary_feedforward
    assert deep_matched_q_ffn.routing_dim == 512
    assert deep_matched_q_ffn.dictionary_size == 1764
    assert deep_matched_q_ffn.q_dictionary_feedforward


def test_q_dictionary_ffn_matches_w_gelu_dt_qx_formula() -> None:
    torch.manual_seed(37)
    config = experiment_config(
        "a1-d1536-qffn",
        model_dim=32,
        routing_dim=8,
        dictionary_size=48,
        backend="torch",
        untied_dictionary=False,
    )
    block = DoubleAttentionBlock(config, feedforward_dim=48)
    bank = attention_modules.SharedDictionaryBank(8, 48, untied=False)
    normalized_dictionary = bank.normalized()
    x = torch.randn(2, 5, 32)

    attention_residual = x + block.attention(
        block.attention_norm(x),
        bank=bank,
        normalized_dictionary=normalized_dictionary,
    )
    normalized_x = block.feedforward_norm(attention_residual)
    query = block.attention.query(normalized_x)
    hidden = F.linear(query, normalized_dictionary[0].T.contiguous())
    expected = attention_residual + block.feedforward(hidden)
    actual = block(x, bank, normalized_dictionary=normalized_dictionary)

    assert isinstance(block.feedforward[0], torch.nn.Identity)
    assert hidden.shape == (2, 5, 48)
    torch.testing.assert_close(actual, expected)


def test_q_dictionary_ffn_rejects_mismatched_hidden_width() -> None:
    config = experiment_config(
        "a1-d1536-qffn",
        model_dim=32,
        routing_dim=8,
        dictionary_size=48,
    )
    with pytest.raises(ValueError, match="feedforward_dim == dictionary_size"):
        DoubleAttentionBlock(config, feedforward_dim=64)


@pytest.mark.parametrize(
    ("name", "normalize_input", "normalize_output"),
    [
        ("a1", True, True),
        ("a1-no-qnorm", False, True),
        ("a1-no-dpnorm", True, False),
        ("a1-no-norm", False, False),
    ],
)
def test_a1_normalization_presets(
    name: str, normalize_input: bool, normalize_output: bool
) -> None:
    config = experiment_config(name)
    assert config.normalize_routing_input is normalize_input
    assert config.normalize_routing_output is normalize_output


def test_causal_prefix_is_independent_of_future_tokens() -> None:
    torch.manual_seed(11)
    config = DoubleAttentionConfig(
        model_dim=16,
        routing_dim=8,
        dictionary_size=16,
        output_projection=False,
        backend="torch",
        causal=True,
    )
    module = SharedDictionaryAttention(config).eval()
    original = torch.randn(1, 8, 16)
    changed = original.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 10
    with torch.no_grad():
        original_output = module(original)
        changed_output = module(changed)
    torch.testing.assert_close(original_output[:, :4], changed_output[:, :4], atol=1e-6, rtol=1e-5)


def test_stagewise_dictionary_count() -> None:
    config = small_config("a1")
    stack = DoubleAttentionStack(config, num_layers=8, dictionary_group_size=4, feedforward_dim=48)
    assert len(stack.banks) == 2
    output = stack(torch.randn(2, 6, 32))
    assert output.shape == (2, 6, 32)


def test_global_dictionary_is_default() -> None:
    stack = DoubleAttentionStack(small_config("a1"), num_layers=3, feedforward_dim=48)
    assert len(stack.banks) == 1


def test_stack_normalizes_each_shared_dictionary_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = DoubleAttentionStack(
        small_config("a1"),
        num_layers=6,
        dictionary_group_size=3,
        feedforward_dim=48,
    )
    calls = 0
    original = attention_modules.SharedDictionaryBank.normalized

    def counted_normalized(
        bank: attention_modules.SharedDictionaryBank,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal calls
        calls += 1
        return original(bank, eps)

    monkeypatch.setattr(
        attention_modules.SharedDictionaryBank,
        "normalized",
        counted_normalized,
    )
    stack(torch.randn(2, 6, 32)).square().mean().backward()
    assert calls == len(stack.banks)


def test_shared_dictionary_normalization_preserves_outputs_and_gradients() -> None:
    torch.manual_seed(13)
    stack = DoubleAttentionStack(
        small_config("qk2-s2"),
        num_layers=3,
        dictionary_group_size=3,
        feedforward_dim=48,
    )
    repeated_stack = copy.deepcopy(stack)
    x = torch.randn(2, 6, 32, requires_grad=True)
    repeated_x = x.detach().clone().requires_grad_()

    shared_output = stack(x)
    repeated_output = repeated_x
    for block in repeated_stack.blocks:
        repeated_output = block(repeated_output, repeated_stack.banks[0])
    shared_output.square().mean().backward()
    repeated_output.square().mean().backward()

    torch.testing.assert_close(shared_output, repeated_output, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(x.grad, repeated_x.grad, atol=1e-6, rtol=1e-5)
    for shared_parameter, repeated_parameter in zip(
        stack.parameters(),
        repeated_stack.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            shared_parameter.grad,
            repeated_parameter.grad,
            atol=1e-6,
            rtol=1e-5,
        )


def test_tied_dictionary_reuses_assignment_atoms_for_reconstruction() -> None:
    module = SharedDictionaryAttention(small_config("a1", untied_dictionary=False))
    assert module.bank is not None
    assert module.bank.raw_value is None
    key, value = module.bank.normalized()
    assert key.data_ptr() == value.data_ptr()

    module(torch.randn(2, 5, 32)).square().mean().backward()
    assert module.bank.raw_key.grad is not None


def test_lm_wrapper_ties_embeddings_and_computes_loss() -> None:
    model = DoubleAttentionLM(
        vocab_size=41,
        max_sequence_length=12,
        config=small_config("a1"),
        num_layers=2,
        feedforward_dim=48,
    )
    assert model.lm_head.weight is model.token_embedding.weight
    token_ids = torch.randint(0, 41, (2, 9))
    logits, loss = model(token_ids, token_ids.roll(-1, dims=1))
    assert logits.shape == (2, 9, 41)
    assert loss is not None and math.isfinite(loss.item())


def test_mha4_baseline_matches_lm_contract() -> None:
    model = MHA4LM(
        vocab_size=41,
        model_dim=32,
        num_layers=2,
        feedforward_dim=48,
        max_sequence_length=12,
    )
    token_ids = torch.randint(0, 41, (2, 9))
    logits, loss = model(token_ids, token_ids.roll(-1, dims=1))
    assert logits.shape == (2, 9, 41)
    assert loss is not None and math.isfinite(loss.item())


def test_invalid_nondivisible_factorization_is_rejected() -> None:
    with pytest.raises(ValueError, match="divide one another"):
        DoubleAttentionConfig(qk_branches=3, outer_maps=2)


def test_explicit_triton_backend_fails_cleanly_on_cpu() -> None:
    config = DoubleAttentionConfig(
        model_dim=16,
        routing_dim=8,
        dictionary_size=16,
        backend="triton",
    )
    module = SharedDictionaryAttention(config)
    with pytest.raises(RuntimeError, match="Triton backend requested"):
        module(torch.randn(1, 4, 16))
