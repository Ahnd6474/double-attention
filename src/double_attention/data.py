from __future__ import annotations

from pathlib import Path

import torch


def load_token_splits(
    ids_path: Path,
    train_ids_path: Path | None = None,
    validation_ids_path: Path | None = None,
    test_ids_path: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
    """Load legacy concatenated IDs or explicit leakage-free data splits."""

    if (train_ids_path is None) != (validation_ids_path is None):
        raise ValueError("train-ids and validation-ids must be provided together")

    if train_ids_path is None:
        ids = torch.load(ids_path, map_location="cpu").long()
        split = int(0.95 * len(ids))
        training, validation = ids[:split], ids[split:]
        test = None
        observed_vocab_size = int(ids.max().item()) + 1
    else:
        training = torch.load(train_ids_path, map_location="cpu").long()
        validation = torch.load(validation_ids_path, map_location="cpu").long()
        test = (
            torch.load(test_ids_path, map_location="cpu").long()
            if test_ids_path is not None
            else None
        )
        splits = [training, validation] + ([] if test is None else [test])
        observed_vocab_size = max(int(split_ids.max().item()) for split_ids in splits) + 1
    return training, validation, test, observed_vocab_size
