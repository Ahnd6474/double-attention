from __future__ import annotations

import argparse
import statistics
import time

import torch

from double_attention import SharedDictionaryAttention, experiment_config


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_forward(
    module: SharedDictionaryAttention,
    x: torch.Tensor,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        module(x)
    synchronize(x.device)
    started = time.perf_counter()
    for _ in range(iterations):
        module(x)
    synchronize(x.device)
    return (time.perf_counter() - started) * 1000 / iterations


def training_step(module: SharedDictionaryAttention, x: torch.Tensor) -> None:
    module.zero_grad(set_to_none=True)
    x.grad = None
    output = module(x)
    output.float().square().mean().backward()


def benchmark_backward(
    module: SharedDictionaryAttention,
    x: torch.Tensor,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        training_step(module, x)
    synchronize(x.device)
    started = time.perf_counter()
    for _ in range(iterations):
        training_step(module, x)
    synchronize(x.device)
    return (time.perf_counter() - started) * 1000 / iterations


def benchmark_paired_backward(
    triton_module: SharedDictionaryAttention,
    torch_module: SharedDictionaryAttention,
    x: torch.Tensor,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    modules = {"triton": triton_module, "torch": torch_module}
    for _ in range(warmup):
        training_step(triton_module, x)
        training_step(torch_module, x)
    synchronize(x.device)

    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        "triton": [],
        "torch": [],
    }
    for iteration in range(iterations):
        order = ("triton", "torch") if iteration % 2 == 0 else ("torch", "triton")
        for name in order:
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            training_step(modules[name], x)
            finished.record()
            events[name].append((started, finished))
    synchronize(x.device)
    medians = {
        name: statistics.median(started.elapsed_time(finished) for started, finished in pairs)
        for name, pairs in events.items()
    }
    return medians["triton"], medians["torch"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="a1")
    parser.add_argument("--backend", choices=("auto", "torch", "triton"), default="auto")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--dtype", choices=("auto", "fp16", "bf16"), default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--backward",
        action="store_true",
        help="benchmark a forward+backward training step instead of inference",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="paired CUDA-event comparison of Triton and PyTorch backward",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    if args.compare and (device.type != "cuda" or not args.backward):
        parser.error("--compare requires CUDA and --backward")
    config = experiment_config(args.variant, backend=args.backend)
    module = SharedDictionaryAttention(config).to(device=device, dtype=dtype).eval()
    x = torch.randn(
        args.batch,
        args.sequence,
        config.model_dim,
        device=device,
        dtype=dtype,
        requires_grad=args.backward,
    )
    if args.compare:
        triton_config = experiment_config(args.variant, backend="triton")
        torch_config = experiment_config(args.variant, backend="torch")
        triton_module = SharedDictionaryAttention(triton_config).to(
            device=device, dtype=dtype
        ).train()
        torch_module = SharedDictionaryAttention(torch_config).to(
            device=device, dtype=dtype
        ).train()
        torch_module.load_state_dict(triton_module.state_dict())
        triton_ms, torch_ms = benchmark_paired_backward(
            triton_module,
            torch_module,
            x,
            args.warmup,
            args.iterations,
        )
        print(
            f"variant={args.variant} mode=paired-forward+backward device={device} "
            f"dtype={dtype} batch={args.batch} sequence={args.sequence} "
            f"triton_ms={triton_ms:.3f} torch_ms={torch_ms:.3f} "
            f"speedup={torch_ms / triton_ms:.3f}x"
        )
        return
    if args.backward:
        module.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        milliseconds = benchmark_backward(module, x, args.warmup, args.iterations)
    else:
        with torch.inference_mode():
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            milliseconds = benchmark_forward(module, x, args.warmup, args.iterations)
    tokens_per_second = args.batch * args.sequence / (milliseconds / 1000)
    parameters = sum(parameter.numel() for parameter in module.parameters())
    peak_memory_mib = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    print(
        f"variant={args.variant} backend={args.backend} mode="
        f"{'forward+backward' if args.backward else 'forward'} device={device} dtype={dtype} "
        f"params={parameters:,} latency_ms={milliseconds:.3f} "
        f"tokens_per_s={tokens_per_second:,.0f} peak_memory_mib={peak_memory_mib:.3f}"
    )


if __name__ == "__main__":
    main()
