from __future__ import annotations

import copy

import pytest
import torch

import double_attention.modules as attention_modules
import double_attention.triton_kernels as triton_kernels
from double_attention import DoubleAttentionStack, SharedDictionaryAttention, experiment_config
from double_attention.ops import dictionary_route_reference, routed_attention_reference
from double_attention.triton_kernels import (
    TRITON_AVAILABLE,
    dictionary_route_triton,
    routed_attention_triton,
)


pytestmark = pytest.mark.skipif(
    not TRITON_AVAILABLE or not torch.cuda.is_available(),
    reason="Triton CUDA runtime is unavailable",
)


def test_fused_query_key_projection_amp_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(8)
    device = torch.device("cuda")
    module = SharedDictionaryAttention(
        experiment_config(
            "qk4-s4",
            model_dim=64,
            routing_dim=16,
            dictionary_size=32,
            backend="torch",
        )
    ).to(device)
    x = torch.randn(2, 17, 64, device=device, requires_grad=True)

    monkeypatch.setattr(attention_modules, "FUSED_QUERY_KEY_PROJECTION", False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        separate_query, separate_key = module._project_query_key(x)
        separate_loss = separate_query.float().square().mean()
        separate_loss = separate_loss + separate_key.float().square().mean()
    separate_loss.backward()
    separate_gradients = (
        x.grad.clone(),
        module.query.weight.grad.clone(),
        module.key.weight.grad.clone(),
    )

    module.zero_grad(set_to_none=True)
    x.grad = None
    monkeypatch.setattr(attention_modules, "FUSED_QUERY_KEY_PROJECTION", True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fused_query, fused_key = module._project_query_key(x)
        fused_loss = fused_query.float().square().mean()
        fused_loss = fused_loss + fused_key.float().square().mean()
    fused_loss.backward()

    torch.testing.assert_close(fused_query, separate_query, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(fused_key, separate_key, atol=2e-2, rtol=2e-2)
    for expected, actual in zip(
        separate_gradients,
        (x.grad, module.query.weight.grad, module.key.weight.grad),
        strict=True,
    ):
        assert actual is not None
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_shared_dictionary_normalization_amp_parity() -> None:
    torch.manual_seed(10)
    device = torch.device("cuda")
    config = experiment_config(
        "qk4-s4",
        model_dim=64,
        routing_dim=128,
        dictionary_size=512,
        backend="triton",
    )
    stack = DoubleAttentionStack(
        config,
        num_layers=2,
        dictionary_group_size=2,
        feedforward_dim=96,
    ).to(device)
    repeated_stack = copy.deepcopy(stack)
    x = torch.randn(1, 9, 64, device=device, requires_grad=True)
    repeated_x = x.detach().clone().requires_grad_()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        shared_output = stack(x)
        repeated_output = repeated_x
        for block in repeated_stack.blocks:
            repeated_output = block(repeated_output, repeated_stack.banks[0])
        shared_loss = shared_output.float().square().mean()
        repeated_loss = repeated_output.float().square().mean()
    shared_loss.backward()
    repeated_loss.backward()

    torch.testing.assert_close(shared_output, repeated_output, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(x.grad, repeated_x.grad, atol=3e-2, rtol=3e-2)
    for shared_parameter, repeated_parameter in zip(
        stack.parameters(),
        repeated_stack.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            shared_parameter.grad,
            repeated_parameter.grad,
            atol=3e-2,
            rtol=3e-2,
        )


@pytest.mark.parametrize(
    (
        "silu_logits",
        "standardize_logits",
        "standardized_logit_scale",
        "normalize_input",
        "normalize_output",
    ),
    [
        (False, False, None, True, True),
        (True, False, None, True, True),
        (True, True, None, False, True),
        (True, True, 0.25, False, True),
        (False, False, None, False, True),
        (False, False, None, True, False),
        (False, False, None, False, False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dictionary_route_forward_and_backward(
    dtype: torch.dtype,
    silu_logits: bool,
    standardize_logits: bool,
    standardized_logit_scale: float | None,
    normalize_input: bool,
    normalize_output: bool,
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(2, 3, 17, 64, device=device, dtype=dtype, requires_grad=True)
    dk = torch.nn.functional.normalize(
        torch.randn(64, 128, device=device, dtype=dtype), dim=0
    ).requires_grad_()
    dv = torch.nn.functional.normalize(
        torch.randn(64, 128, device=device, dtype=dtype), dim=0
    ).requires_grad_()
    beta = torch.tensor(4.0, device=device, requires_grad=True)

    activation = "silu" if silu_logits else "identity"
    expected = dictionary_route_reference(
        x,
        dk,
        dv,
        beta,
        activation=activation,
        standardize_logits=standardize_logits,
        standardized_logit_scale=standardized_logit_scale,
        normalize_input=normalize_input,
        normalize_output=normalize_output,
    )
    actual = dictionary_route_triton(
        x,
        dk,
        dv,
        beta,
        silu_logits=silu_logits,
        standardize_logits=standardize_logits,
        standardized_logit_scale=standardized_logit_scale,
        normalize_input=normalize_input,
        normalize_output=normalize_output,
    )
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)

    gradient = torch.randn_like(actual)
    actual.backward(gradient, retain_graph=True)
    actual_gradients = (x.grad.clone(), dk.grad.clone(), dv.grad.clone(), beta.grad.clone())
    for tensor in (x, dk, dv, beta):
        tensor.grad = None
    expected.backward(gradient)
    for index, (actual_gradient, tensor) in enumerate(
        zip(actual_gradients, (x, dk, dv, beta), strict=True)
    ):
        reference_gradient = tensor.grad
        assert reference_gradient is not None
        atol = 3e-1 if dtype == torch.bfloat16 and not normalize_input else 3e-2
        torch.testing.assert_close(
            actual_gradient, reference_gradient, atol=atol, rtol=3e-2
        )
        if index < 3:
            error_rms = (actual_gradient.float() - reference_gradient.float()).square().mean().sqrt()
            reference_rms = reference_gradient.float().square().mean().sqrt()
            assert error_rms / reference_rms.clamp_min(1e-12) < 3e-2


@pytest.mark.parametrize("gain", [1.0, 4.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_direct_silu_dictionary_route_forward_and_backward(
    dtype: torch.dtype, gain: float
) -> None:
    torch.manual_seed(41 + int(gain))
    device = torch.device("cuda")
    x = torch.randn(1, 2, 11, 256, device=device, dtype=dtype, requires_grad=True)
    dk = torch.nn.functional.normalize(
        torch.randn(256, 512, device=device, dtype=dtype), dim=0
    ).requires_grad_()
    dv = torch.nn.functional.normalize(
        torch.randn(256, 512, device=device, dtype=dtype), dim=0
    ).requires_grad_()
    beta = torch.tensor(4.0, device=device)

    expected = dictionary_route_reference(
        x, dk, dv, beta, assignment="silu", silu_gain=gain
    )
    actual = dictionary_route_triton(
        x, dk, dv, beta, direct_silu=True, silu_gain=gain
    )
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)

    gradient = torch.randn_like(actual)
    actual.backward(gradient, retain_graph=True)
    actual_gradients = tuple(tensor.grad.clone() for tensor in (x, dk, dv))
    for tensor in (x, dk, dv):
        tensor.grad = None
    expected.backward(gradient)
    for actual_gradient, tensor in zip(
        actual_gradients, (x, dk, dv), strict=True
    ):
        assert tensor.grad is not None
        torch.testing.assert_close(actual_gradient, tensor.grad, atol=5e-2, rtol=5e-2)


def test_dictionary_route_fixed_beta_backward() -> None:
    torch.manual_seed(6)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    x = torch.randn(2, 2, 19, 64, device=device, dtype=dtype, requires_grad=True)
    dk = torch.nn.functional.normalize(
        torch.randn(64, 128, device=device, dtype=dtype), dim=0
    ).requires_grad_()
    dv = torch.nn.functional.normalize(
        torch.randn(64, 128, device=device, dtype=dtype), dim=0
    ).requires_grad_()
    beta = torch.tensor(4.0, device=device)

    expected = dictionary_route_reference(x, dk, dv, beta)
    actual = dictionary_route_triton(x, dk, dv, beta)
    gradient = torch.randn_like(actual)
    actual.backward(gradient, retain_graph=True)
    actual_gradients = (x.grad.clone(), dk.grad.clone(), dv.grad.clone())
    for tensor in (x, dk, dv):
        tensor.grad = None
    expected.backward(gradient)
    for actual_gradient, tensor in zip(actual_gradients, (x, dk, dv), strict=True):
        torch.testing.assert_close(actual_gradient, tensor.grad, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("dictionary_size", [512, 1024])
def test_wide_dictionary_route_forward_backward(dictionary_size: int) -> None:
    torch.manual_seed(31 + dictionary_size)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    x = torch.randn(1, 2, 7, 512, device=device, dtype=dtype, requires_grad=True)
    dictionary_key = torch.nn.functional.normalize(
        torch.randn(512, dictionary_size, device=device, dtype=dtype), dim=0
    ).requires_grad_()
    dictionary_value = torch.nn.functional.normalize(
        torch.randn(512, dictionary_size, device=device, dtype=dtype), dim=0
    ).requires_grad_()
    beta = torch.tensor(4.0 * (2.0**0.5), device=device, requires_grad=True)

    expected = dictionary_route_reference(
        x, dictionary_key, dictionary_value, beta
    )
    actual = dictionary_route_triton(x, dictionary_key, dictionary_value, beta)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)

    gradient = torch.randn_like(actual)
    actual.backward(gradient, retain_graph=True)
    actual_gradients = tuple(
        tensor.grad.clone() for tensor in (x, dictionary_key, dictionary_value, beta)
    )
    for tensor in (x, dictionary_key, dictionary_value, beta):
        tensor.grad = None
    expected.backward(gradient)
    for actual_gradient, tensor in zip(
        actual_gradients,
        (x, dictionary_key, dictionary_value, beta),
        strict=True,
    ):
        assert tensor.grad is not None
        torch.testing.assert_close(actual_gradient, tensor.grad, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_normalized_assignment_matches_cublas(dtype: torch.dtype) -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    x = torch.randn(37, 256, device=device, dtype=dtype)
    dictionary_key = torch.nn.functional.normalize(
        torch.randn(256, 512, device=device, dtype=dtype), dim=0
    )

    expected_normalized, expected_inverse_norm = triton_kernels._row_l2_normalize(
        x, 1e-6
    )
    expected_logits = expected_normalized @ dictionary_key
    normalized, inverse_norm, logits = triton_kernels._normalized_assignment(
        x, dictionary_key, 1e-6
    )

    torch.testing.assert_close(normalized, expected_normalized, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(inverse_norm, expected_inverse_norm, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(logits, expected_logits, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize(
    (
        "silu_logits",
        "standardize_logits",
        "standardized_logit_scale",
        "normalize_input",
        "normalize_output",
    ),
    [
        (False, False, None, True, True),
        (True, False, None, True, True),
        (True, True, None, False, True),
        (True, True, 0.25, False, True),
        (False, False, None, False, True),
        (False, False, None, True, False),
        (False, False, None, False, False),
    ],
)
@pytest.mark.parametrize("routing_dim", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fully_fused_dictionary_route_forward_backward(
    dtype: torch.dtype,
    routing_dim: int,
    silu_logits: bool,
    standardize_logits: bool,
    standardized_logit_scale: float | None,
    normalize_input: bool,
    normalize_output: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(triton_kernels, "FUSED_DICTIONARY_ROUTE", True)
    monkeypatch.setattr(triton_kernels, "FUSED_DICTIONARY_ROUTE_BACKWARD", True)
    torch.manual_seed(9 + routing_dim)
    device = torch.device("cuda")
    x = torch.randn(1, 2, 9, routing_dim, device=device, dtype=dtype, requires_grad=True)
    dictionary_key = torch.nn.functional.normalize(
        torch.randn(routing_dim, 512, device=device, dtype=dtype), dim=0
    ).requires_grad_()
    dictionary_value = torch.nn.functional.normalize(
        torch.randn(routing_dim, 512, device=device, dtype=dtype), dim=0
    ).requires_grad_()
    beta = torch.tensor(4.0, device=device, requires_grad=True)

    activation = "silu" if silu_logits else "identity"
    expected = dictionary_route_reference(
        x,
        dictionary_key,
        dictionary_value,
        beta,
        activation=activation,
        standardize_logits=standardize_logits,
        standardized_logit_scale=standardized_logit_scale,
        normalize_input=normalize_input,
        normalize_output=normalize_output,
    )
    actual = dictionary_route_triton(
        x,
        dictionary_key,
        dictionary_value,
        beta,
        silu_logits=silu_logits,
        standardize_logits=standardize_logits,
        standardized_logit_scale=standardized_logit_scale,
        normalize_input=normalize_input,
        normalize_output=normalize_output,
    )
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)

    gradient = torch.randn_like(actual)
    actual.backward(gradient, retain_graph=True)
    actual_gradients = tuple(
        tensor.grad.clone() for tensor in (x, dictionary_key, dictionary_value, beta)
    )
    for tensor in (x, dictionary_key, dictionary_value, beta):
        tensor.grad = None
    expected.backward(gradient)
    for index, (actual_gradient, tensor) in enumerate(
        zip(
            actual_gradients,
            (x, dictionary_key, dictionary_value, beta),
            strict=True,
        )
    ):
        reference_gradient = tensor.grad
        assert reference_gradient is not None
        atol = 3e-1 if dtype == torch.bfloat16 and not normalize_input else 3e-2
        torch.testing.assert_close(
            actual_gradient, reference_gradient, atol=atol, rtol=3e-2
        )
        if index < 3:
            error_rms = (actual_gradient.float() - reference_gradient.float()).square().mean().sqrt()
            reference_rms = reference_gradient.float().square().mean().sqrt()
            assert error_rms / reference_rms.clamp_min(1e-12) < 3e-2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_routed_attention_forward_and_backward(dtype: torch.dtype, causal: bool) -> None:
    torch.manual_seed(1)
    device = torch.device("cuda")
    # Routed Q/K are normalized by the dictionary feature map before they
    # reach the outer attention kernel.  Test that production contract rather
    # than an unrealistically saturated softmax over unnormalized vectors.
    q = torch.nn.functional.normalize(
        torch.randn(2, 2, 65, 64, device=device, dtype=dtype), dim=-1
    ).requires_grad_()
    k = torch.nn.functional.normalize(
        torch.randn(2, 2, 65, 64, device=device, dtype=dtype), dim=-1
    ).requires_grad_()
    v = torch.randn(2, 65, 96, device=device, dtype=dtype, requires_grad=True)
    scales = torch.tensor([12.0, 17.0], device=device, requires_grad=True)

    expected = routed_attention_reference(q, k, v, scales, causal)
    actual = routed_attention_triton(q, k, v, scales, causal)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)

    gradient = torch.randn_like(actual)
    actual.backward(gradient, retain_graph=True)
    actual_gradients = (q.grad.clone(), k.grad.clone(), v.grad.clone(), scales.grad.clone())
    for tensor in (q, k, v, scales):
        tensor.grad = None
    expected.backward(gradient)
    backward_atol = 1.6e-1 if dtype == torch.bfloat16 else 4e-2
    for actual_gradient, tensor in zip(actual_gradients, (q, k, v, scales), strict=True):
        torch.testing.assert_close(
            actual_gradient,
            tensor.grad,
            atol=backward_atol,
            rtol=4e-2,
        )


def test_routed_attention_production_shape_backward() -> None:
    torch.manual_seed(2)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q = torch.nn.functional.normalize(
        torch.randn(1, 4, 64, 128, device=device, dtype=dtype), dim=-1
    ).requires_grad_()
    k = torch.nn.functional.normalize(
        torch.randn(1, 4, 64, 128, device=device, dtype=dtype), dim=-1
    ).requires_grad_()
    v = torch.randn(1, 64, 512, device=device, dtype=dtype, requires_grad=True)
    scales = torch.full((4,), 12.0, device=device, requires_grad=True)

    expected = routed_attention_reference(q, k, v, scales, causal=True)
    actual = routed_attention_triton(q, k, v, scales, causal=True)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)

    gradient = torch.randn_like(actual)
    actual.backward(gradient, retain_graph=True)
    actual_gradients = (q.grad.clone(), k.grad.clone(), v.grad.clone(), scales.grad.clone())
    for tensor in (q, k, v, scales):
        tensor.grad = None
    expected.backward(gradient)
    for actual_gradient, tensor in zip(actual_gradients, (q, k, v, scales), strict=True):
        torch.testing.assert_close(actual_gradient, tensor.grad, atol=2e-1, rtol=5e-2)


def test_routed_attention_long_context_backward() -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")
    dtype = torch.float16
    q = torch.nn.functional.normalize(
        torch.randn(1, 2, 257, 64, device=device, dtype=dtype), dim=-1
    ).requires_grad_()
    k = torch.nn.functional.normalize(
        torch.randn(1, 2, 257, 64, device=device, dtype=dtype), dim=-1
    ).requires_grad_()
    v = torch.randn(1, 257, 80, device=device, dtype=dtype, requires_grad=True)
    scales = torch.tensor([11.0, 15.0], device=device, requires_grad=True)

    expected = routed_attention_reference(q, k, v, scales, causal=True)
    actual = routed_attention_triton(q, k, v, scales, causal=True)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)

    gradient = torch.randn_like(actual)
    actual.backward(gradient, retain_graph=True)
    actual_gradients = (q.grad.clone(), k.grad.clone(), v.grad.clone(), scales.grad.clone())
    for tensor in (q, k, v, scales):
        tensor.grad = None
    expected.backward(gradient)
    for actual_gradient, tensor in zip(actual_gradients, (q, k, v, scales), strict=True):
        torch.testing.assert_close(actual_gradient, tensor.grad, atol=5e-2, rtol=5e-2)


def test_routed_attention_wide_routing_backward() -> None:
    torch.manual_seed(4)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q = torch.nn.functional.normalize(
        torch.randn(1, 2, 64, 256, device=device, dtype=dtype), dim=-1
    ).requires_grad_()
    k = torch.nn.functional.normalize(
        torch.randn(1, 2, 64, 256, device=device, dtype=dtype), dim=-1
    ).requires_grad_()
    v = torch.randn(1, 64, 128, device=device, dtype=dtype, requires_grad=True)
    scales = torch.tensor([10.0, 14.0], device=device, requires_grad=True)

    expected = routed_attention_reference(q, k, v, scales, causal=True)
    actual = routed_attention_triton(q, k, v, scales, causal=True)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)

    gradient = torch.randn_like(actual)
    actual.backward(gradient, retain_graph=True)
    actual_gradients = (q.grad.clone(), k.grad.clone(), v.grad.clone(), scales.grad.clone())
    for tensor in (q, k, v, scales):
        tensor.grad = None
    expected.backward(gradient)
    for actual_gradient, tensor in zip(actual_gradients, (q, k, v, scales), strict=True):
        torch.testing.assert_close(actual_gradient, tensor.grad, atol=2e-1, rtol=5e-2)


def test_routed_attention_recompute_fallback_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(triton_kernels, "MAX_PRECOMPUTED_DP_BYTES", 0)
    torch.manual_seed(5)
    device = torch.device("cuda")
    dtype = torch.float16
    q = torch.nn.functional.normalize(
        torch.randn(1, 2, 33, 32, device=device, dtype=dtype), dim=-1
    ).requires_grad_()
    k = torch.nn.functional.normalize(
        torch.randn(1, 2, 33, 32, device=device, dtype=dtype), dim=-1
    ).requires_grad_()
    v = torch.randn(1, 33, 48, device=device, dtype=dtype, requires_grad=True)
    scales = torch.tensor([9.0, 13.0], device=device, requires_grad=True)

    expected = routed_attention_reference(q, k, v, scales, causal=True)
    actual = routed_attention_triton(q, k, v, scales, causal=True)
    gradient = torch.randn_like(actual)
    actual.backward(gradient, retain_graph=True)
    actual_gradients = (q.grad.clone(), k.grad.clone(), v.grad.clone(), scales.grad.clone())
    for tensor in (q, k, v, scales):
        tensor.grad = None
    expected.backward(gradient)
    for actual_gradient, tensor in zip(actual_gradients, (q, k, v, scales), strict=True):
        torch.testing.assert_close(actual_gradient, tensor.grad, atol=5e-2, rtol=5e-2)
