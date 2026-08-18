from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from double_attention import DoubleAttentionLM, MHA4LM, experiment_config
from double_attention.data import load_token_splits


VARIANTS = (
    "a1",
    "a1-silu",
    "a1-silu-logitnorm",
    "a1-silu-logitnorm-t1",
    "a1-r512-d512",
    "a1-r512-d1024",
    "a1-r512-d1536",
    "a1-r512-d1536-qffn",
    "a1-r512-d2855-qffn",
    "a1-d1536",
    "a1-d1536-qffn",
    "a1-no-softmax",
    "a1-no-softmax-g4",
    "a1-no-qnorm",
    "a1-no-dpnorm",
    "a1-no-norm",
    "qk2-s1",
    "qk1-s2",
    "qk2-s2",
    "qk4-s4",
    "mha4",
)


def stable_seed(name: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


@torch.no_grad()
def deterministic_initialize(model: nn.Module, base_seed: int) -> None:
    """Initialize common named tensors identically across all variants."""

    for name, parameter in model.named_parameters():
        generator = torch.Generator(device="cpu").manual_seed(stable_seed(name, base_seed))
        if name.endswith("raw_key") or name.endswith("raw_value"):
            nn.init.normal_(parameter, std=parameter.shape[0] ** -0.5, generator=generator)
        elif name.endswith("norm.weight") or ".norm.weight" in name:
            parameter.fill_(1.0)
        elif name.endswith("bias"):
            parameter.zero_()
        elif parameter.ndim >= 2:
            bound = parameter.shape[-1] ** -0.5
            parameter.uniform_(-bound, bound, generator=generator)
        else:
            # Learned temperatures/mix coefficients keep their architectural defaults.
            continue


def make_model(
    variant: str,
    vocab_size: int,
    sequence_length: int,
    backend: str,
    seed: int,
    dictionary: str,
) -> nn.Module:
    if variant == "mha4":
        model: nn.Module = MHA4LM(
            vocab_size=vocab_size,
            model_dim=512,
            num_layers=6,
            feedforward_dim=1536,
            max_sequence_length=sequence_length,
        )
    else:
        config = experiment_config(
            variant,
            model_dim=512,
            backend=backend,
            untied_dictionary=dictionary == "untied",
            output_projection=True,
        )
        model = DoubleAttentionLM(
            vocab_size=vocab_size,
            max_sequence_length=sequence_length,
            config=config,
            num_layers=6,
            dictionary_group_size=6,
            feedforward_dim=(config.dictionary_size if config.q_dictionary_feedforward else 1536),
        )
    deterministic_initialize(model, 1000 + seed)
    return model


def get_batch(
    source: torch.Tensor,
    generator: torch.Generator,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(
        0,
        len(source) - sequence_length - 1,
        (batch_size,),
        generator=generator,
    )
    x = torch.stack([source[index : index + sequence_length] for index in starts]).long()
    y = torch.stack([source[index + 1 : index + sequence_length + 1] for index in starts]).long()
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    validation: torch.Tensor,
    batches: int,
    seed: int,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    total_loss = 0.0
    total_tokens = 0
    for _ in range(batches):
        x, y = get_batch(validation, generator, batch_size, sequence_length, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits, _ = model(x)
        total_loss += F.cross_entropy(
            logits.float().flatten(0, 1),
            y.flatten(),
            reduction="sum",
        ).item()
        total_tokens += y.numel()
    model.train()
    return total_loss / total_tokens


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    tokens_seen: int,
    history: list[dict[str, float | int]],
    args: argparse.Namespace,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "history": history,
            "args": vars(args),
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--ids", type=Path, default=Path("data/docs_sp2048/docs_sp2048_ids.pt"))
    parser.add_argument("--train-ids", type=Path)
    parser.add_argument("--validation-ids", type=Path)
    parser.add_argument("--test-ids", type=Path)
    parser.add_argument(
        "--corpus-name",
        help="stable corpus label stored in result metadata (defaults to the IDs filename)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        help="fixed tokenizer vocabulary size; defaults to max observed token ID plus one",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/screen"))
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--schedule-steps", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--effective-batch", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--backend", choices=("auto", "torch", "triton"), default="triton")
    parser.add_argument(
        "--dictionary",
        choices=("tied", "untied"),
        default="untied",
        help="share one normalized dictionary for assignment/reconstruction or learn two",
    )
    parser.add_argument("--max-lr", type=float, default=6e-4)
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.effective_batch % args.micro_batch:
        raise ValueError("effective-batch must be divisible by micro-batch")
    if args.steps > args.schedule_steps:
        raise ValueError("steps cannot exceed schedule-steps")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this screening script")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    amp_dtype = torch.bfloat16 if capability[0] >= 8 and torch.cuda.is_bf16_supported() else torch.float16

    training, validation, test, observed_vocab_size = load_token_splits(
        args.ids,
        args.train_ids,
        args.validation_ids,
        args.test_ids,
    )
    for split_name, split_ids in (("training", training), ("validation", validation)):
        if len(split_ids) <= args.sequence_length + 1:
            raise ValueError(f"{split_name} split is too short for sequence length")
    if test is not None and len(test) <= args.sequence_length + 1:
        raise ValueError("test split is too short for sequence length")
    vocab_size = args.vocab_size or observed_vocab_size
    if vocab_size < observed_vocab_size:
        raise ValueError(
            f"vocab-size {vocab_size} is smaller than observed token range {observed_vocab_size}"
        )
    model = make_model(
        args.variant,
        vocab_size,
        args.sequence_length,
        args.backend,
        args.seed,
        args.dictionary,
    ).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    accumulation_steps = args.effective_batch // args.micro_batch
    generator = torch.Generator().manual_seed(7000 + args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dictionary_tag = "" if args.dictionary == "untied" or args.variant == "mha4" else "_tied"
    run_name = (
        f"{args.variant}{dictionary_tag}_s{args.seed}_steps{args.steps}_sched{args.schedule_steps}"
    )
    checkpoint_path = args.output_dir / f"{run_name}.last.pt"
    result_path = args.output_dir / f"{run_name}.json"

    start_step = 0
    tokens_seen = 0
    history: list[dict[str, float | int]] = []
    if args.resume and checkpoint_path.exists():
        # This checkpoint is written locally by save_checkpoint above and
        # contains optimizer state plus argparse Path objects, so it is not a
        # weights-only artifact.  Never use this path for untrusted files.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
        tokens_seen = int(checkpoint["tokens_seen"])
        history = list(checkpoint["history"])

    metadata = {
        "variant": args.variant,
        "dictionary": "none" if args.variant == "mha4" else args.dictionary,
        "corpus": args.corpus_name or (args.train_ids or args.ids).stem,
        "vocab_size": vocab_size,
        "seed": args.seed,
        "parameters": parameters,
        "device": torch.cuda.get_device_name(device),
        "amp_dtype": str(amp_dtype),
        "micro_batch": args.micro_batch,
        "effective_batch": args.effective_batch,
        "sequence_length": args.sequence_length,
        "training_tokens": len(training),
        "validation_tokens": len(validation),
        "test_tokens": 0 if test is None else len(test),
        "start_step": start_step,
    }
    print(json.dumps(metadata), flush=True)

    checks = {100, 500, 1000, 2000, 4000, 6000, 8000, 10000, args.steps}
    model.train()
    last_time = time.perf_counter()
    for step in range(start_step + 1, args.steps + 1):
        if step <= args.warmup:
            learning_rate = args.max_lr * step / args.warmup
        else:
            progress = (step - args.warmup) / max(1, args.schedule_steps - args.warmup)
            learning_rate = args.max_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(accumulation_steps):
            x, y = get_batch(
                training,
                generator,
                args.micro_batch,
                args.sequence_length,
                device,
            )
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _, loss = model(x, y)
                assert loss is not None
                scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            train_loss += loss.detach().item() / accumulation_steps
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
        scaler.step(optimizer)
        scaler.update()
        tokens_seen += args.effective_batch * args.sequence_length

        if step in checks:
            validation_loss = evaluate(
                model,
                validation,
                args.eval_batches,
                9876,
                args.micro_batch,
                args.sequence_length,
                device,
                amp_dtype,
            )
            now = time.perf_counter()
            record: dict[str, float | int] = {
                "step": step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "tokens_seen": tokens_seen,
                "learning_rate": learning_rate,
                "chunk_seconds": now - last_time,
                "max_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scaler,
                step,
                tokens_seen,
                history,
                args,
            )
            last_time = now

    robust_a = evaluate(
        model, validation, 64, 24680, args.micro_batch, args.sequence_length, device, amp_dtype
    )
    robust_b = evaluate(
        model, validation, 64, 13579, args.micro_batch, args.sequence_length, device, amp_dtype
    )
    test_metrics: dict[str, float] = {}
    if test is not None:
        test_a = evaluate(
            model, test, 64, 86420, args.micro_batch, args.sequence_length, device, amp_dtype
        )
        test_b = evaluate(
            model, test, 64, 97531, args.micro_batch, args.sequence_length, device, amp_dtype
        )
        test_mean = (test_a + test_b) / 2
        test_metrics = {
            "test_a": test_a,
            "test_b": test_b,
            "test_mean": test_mean,
            "test_perplexity": math.exp(test_mean),
        }
    result = {
        **metadata,
        "steps": args.steps,
        "schedule_steps": args.schedule_steps,
        "tokens_seen": tokens_seen,
        "robust_a": robust_a,
        "robust_b": robust_b,
        "robust_mean": (robust_a + robust_b) / 2,
        "perplexity": math.exp((robust_a + robust_b) / 2),
        **test_metrics,
        "history": history,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
