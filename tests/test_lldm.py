from __future__ import annotations

import math

import torch

from double_attention import (
    LLDMConfig,
    LayerLocalDictionaryMixerBlock,
    LayerLocalDictionaryMixerLM,
    LayerLocalDictionaryMixerStack,
)


def small_config(**overrides: object) -> LLDMConfig:
    values: dict[str, object] = {
        "model_dim": 32,
        "dictionary_size": 48,
        "relational_dim": 8,
        "relational_maps": 2,
    }
    values.update(overrides)
    return LLDMConfig(**values)


def test_lldm_block_forward_backward_and_shared_feature_state() -> None:
    torch.manual_seed(41)
    block = LayerLocalDictionaryMixerBlock(small_config())
    x = torch.randn(2, 7, 32, requires_grad=True)
    output, aux = block(x, return_aux=True)

    assert output.shape == x.shape
    assert aux.features.shape == (2, 7, 48)
    assert aux.assignments.shape == aux.features.shape
    assert aux.map_weights.shape == (2,)
    torch.testing.assert_close(aux.assignments.sum(dim=-1), torch.ones(2, 7))
    torch.testing.assert_close(aux.map_weights.sum(), torch.tensor(1.0))

    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert block.dictionary.weight.grad is not None
    assert block.output.weight.grad is not None
    for projection in block.relational_projections:
        assert projection.weight.grad is not None


def test_lldm_causal_prefix_is_independent_of_future_tokens() -> None:
    torch.manual_seed(43)
    block = LayerLocalDictionaryMixerBlock(small_config()).eval()
    original = torch.randn(1, 8, 32)
    changed = original.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 10
    with torch.no_grad():
        original_output = block(original)
        changed_output = block(changed)
    torch.testing.assert_close(
        original_output[:, :4],
        changed_output[:, :4],
        atol=1e-5,
        rtol=1e-5,
    )


def test_centered_relational_assignment_removes_uniform_component() -> None:
    torch.manual_seed(47)
    block = LayerLocalDictionaryMixerBlock(
        small_config(center_relational_assignments=True)
    )
    x = torch.randn(2, 7, 32)
    _, assignments = block._feature_state(x)
    centered = assignments - assignments.mean(dim=-1, keepdim=True)
    torch.testing.assert_close(centered.sum(dim=-1), torch.zeros(2, 7), atol=1e-6, rtol=0)


def test_assignment_scale_controls_simplex_entropy() -> None:
    torch.manual_seed(53)
    diffuse = LayerLocalDictionaryMixerBlock(small_config(assignment_scale=0.25))
    sharp = LayerLocalDictionaryMixerBlock(small_config(assignment_scale=1.0))
    sharp.load_state_dict(diffuse.state_dict(), strict=False)
    sharp.assignment_scale.fill_(1.0)
    x = torch.randn(2, 7, 32)
    _, diffuse_assignment = diffuse._feature_state(x)
    _, sharp_assignment = sharp._feature_state(x)
    diffuse_entropy = -(diffuse_assignment * diffuse_assignment.log()).sum(dim=-1).mean()
    sharp_entropy = -(sharp_assignment * sharp_assignment.log()).sum(dim=-1).mean()
    assert sharp_entropy < diffuse_entropy


def test_lldm_stack_uses_one_dictionary_per_layer() -> None:
    stack = LayerLocalDictionaryMixerStack(small_config(), num_layers=3)
    dictionaries = [block.dictionary.weight for block in stack.blocks]
    assert len({parameter.data_ptr() for parameter in dictionaries}) == 3
    assert stack(torch.randn(2, 6, 32)).shape == (2, 6, 32)


def test_lldm_lm_contract_and_tied_embeddings() -> None:
    model = LayerLocalDictionaryMixerLM(
        vocab_size=41,
        max_sequence_length=12,
        config=small_config(),
        num_layers=2,
    )
    assert model.lm_head.weight is model.token_embedding.weight
    token_ids = torch.randint(0, 41, (2, 9))
    logits, loss = model(token_ids, token_ids.roll(-1, dims=1))
    assert logits.shape == (2, 9, 41)
    assert loss is not None and math.isfinite(loss.item())


def test_lldm2_parameter_reduction_at_experiment_shape() -> None:
    model = LayerLocalDictionaryMixerLM(
        vocab_size=2048,
        max_sequence_length=64,
        config=LLDMConfig(),
        num_layers=6,
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == 8_986_142


def test_parameter_matched_lldm_is_close_to_a1_size() -> None:
    model = LayerLocalDictionaryMixerLM(
        vocab_size=2048,
        max_sequence_length=64,
        config=LLDMConfig(dictionary_size=1536, relational_dim=256),
        num_layers=6,
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == 15_296_030
    assert abs(parameters - 15_396_870) / 15_396_870 < 0.01


def test_separate_output_lldm_is_parameter_matched_and_has_two_syntheses() -> None:
    model = LayerLocalDictionaryMixerLM(
        vocab_size=2048,
        max_sequence_length=64,
        config=LLDMConfig(
            dictionary_size=1152,
            relational_dim=256,
            separate_context_output=True,
        ),
        num_layers=6,
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == 15_282_206
    assert abs(parameters - 15_396_870) / 15_396_870 < 0.01
    assert all(block.context_output is not None for block in model.stack.blocks)


def test_independent_relational_readouts_are_parameter_matched() -> None:
    model = LayerLocalDictionaryMixerLM(
        vocab_size=2048,
        max_sequence_length=64,
        config=LLDMConfig(
            dictionary_size=1264,
            relational_dim=128,
            independent_relational_readouts=True,
        ),
        num_layers=6,
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == 15_461_918
    assert abs(parameters - 15_396_870) / 15_396_870 < 0.01
    for block in model.stack.blocks:
        assert len(block.relational_projections) == 0
        assert len(block.query_projections) == 2
        assert len(block.key_projections) == 2
        assert len(block.value_projections) == 2
        assert len(block.context_projections) == 2


def test_independent_query_key_control_keeps_shared_value_lift() -> None:
    model = LayerLocalDictionaryMixerLM(
        vocab_size=2048,
        max_sequence_length=64,
        config=LLDMConfig(
            dictionary_size=1024,
            relational_dim=128,
            separate_context_output=True,
            independent_query_key=True,
        ),
        num_layers=6,
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == 15_253_022
    assert abs(parameters - 15_396_870) / 15_396_870 < 0.01
    for block in model.stack.blocks:
        assert len(block.query_projections) == 2
        assert len(block.key_projections) == 2
        assert len(block.relational_projections) == 2
        assert len(block.value_projections) == 0
        assert len(block.context_projections) == 0
