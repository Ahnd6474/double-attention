from __future__ import annotations

import math

import pytest
import torch

from double_attention import (
    DoubleAttentionConfig,
    DoubleAttentionLM,
    DoubleAttentionStack,
    MHA4LM,
    SharedDictionaryAttention,
    experiment_config,
    experiment_names,
)


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
    assert module.bank is not None
    assert module.bank.raw_key.grad is not None


def test_qk2_s1_concatenates_two_score_geometries() -> None:
    module = SharedDictionaryAttention(small_config("qk2-s1"))
    _, aux = module(torch.randn(1, 5, 32), return_aux=True)
    assert aux.routing_query.shape == (1, 2, 5, 8)
    assert aux.score_query.shape == (1, 1, 5, 16)


def test_qk1_s2_repeats_geometry_but_breaks_temperature_symmetry() -> None:
    module = SharedDictionaryAttention(small_config("qk1-s2"))
    _, aux = module(torch.randn(1, 5, 32), return_aux=True)
    torch.testing.assert_close(aux.score_query[:, 0], aux.score_query[:, 1])
    assert aux.score_scales[0] != aux.score_scales[1]


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
