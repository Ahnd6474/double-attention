from __future__ import annotations

import argparse
import statistics

import torch

from double_attention import DoubleAttentionLM, experiment_config


def training_step(
    model: DoubleAttentionLM,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    model.zero_grad(set_to_none=True)
    _, loss = model(token_ids, targets)
    if loss is None:
        raise RuntimeError("language-model benchmark requires a training loss")
    loss.backward()


def paired_benchmark(
    triton_model: DoubleAttentionLM,
    torch_model: DoubleAttentionLM,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    models = {"triton": triton_model, "torch": torch_model}
    for _ in range(warmup):
        training_step(triton_model, token_ids, targets)
        training_step(torch_model, token_ids, targets)
    torch.cuda.synchronize(token_ids.device)

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
            training_step(models[name], token_ids, targets)
            finished.record()
            events[name].append((started, finished))
    torch.cuda.synchronize(token_ids.device)
    medians = {
        name: statistics.median(started.elapsed_time(finished) for started, finished in pairs)
        for name, pairs in events.items()
    }
    return medians["triton"], medians["torch"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="qk4-s4")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    common = {
        "vocab_size": args.vocab_size,
        "max_sequence_length": args.sequence,
        "num_layers": args.layers,
        "dictionary_group_size": args.layers,
        "feedforward_dim": 1536,
    }
    triton_model = DoubleAttentionLM(
        config=experiment_config(args.variant, backend="triton"),
        **common,
    ).to(device=device, dtype=dtype).train()
    torch_model = DoubleAttentionLM(
        config=experiment_config(args.variant, backend="torch"),
        **common,
    ).to(device=device, dtype=dtype).train()
    torch_model.load_state_dict(triton_model.state_dict())
    token_ids = torch.randint(
        args.vocab_size,
        (args.batch, args.sequence),
        device=device,
    )
    targets = torch.randint(
        args.vocab_size,
        (args.batch, args.sequence),
        device=device,
    )
    triton_ms, torch_ms = paired_benchmark(
        triton_model,
        torch_model,
        token_ids,
        targets,
        args.warmup,
        args.iterations,
    )
    print(
        f"variant={args.variant} mode=paired-lm-forward+backward "
        f"layers={args.layers} batch={args.batch} sequence={args.sequence} "
        f"dtype={dtype} triton_ms={triton_ms:.3f} torch_ms={torch_ms:.3f} "
        f"speedup={torch_ms / triton_ms:.3f}x"
    )


if __name__ == "__main__":
    main()
