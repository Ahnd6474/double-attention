from __future__ import annotations

import pytest
import torch

import double_attention.triton_kernels as triton_kernels
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dictionary_route_forward_and_backward(dtype: torch.dtype) -> None:
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

    expected = dictionary_route_reference(x, dk, dv, beta)
    actual = dictionary_route_triton(x, dk, dv, beta)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)

    gradient = torch.randn_like(actual)
    actual.backward(gradient, retain_graph=True)
    actual_gradients = (x.grad.clone(), dk.grad.clone(), dv.grad.clone(), beta.grad.clone())
    for tensor in (x, dk, dv, beta):
        tensor.grad = None
    expected.backward(gradient)
    for actual_gradient, tensor in zip(actual_gradients, (x, dk, dv, beta), strict=True):
        torch.testing.assert_close(actual_gradient, tensor.grad, atol=3e-2, rtol=3e-2)


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
