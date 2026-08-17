from __future__ import annotations

from pathlib import Path

import pytest
import torch

from double_attention.data import load_token_splits


def save(path: Path, values: list[int]) -> Path:
    torch.save(torch.tensor(values, dtype=torch.int32), path)
    return path


def test_load_token_splits_legacy_95_5(tmp_path: Path) -> None:
    ids = save(tmp_path / "all.pt", list(range(100)))
    train, validation, test, observed_vocab = load_token_splits(ids)
    assert len(train) == 95
    assert len(validation) == 5
    assert test is None
    assert observed_vocab == 100


def test_load_token_splits_explicit_includes_test_vocabulary(tmp_path: Path) -> None:
    train_path = save(tmp_path / "train.pt", [0, 1, 2])
    validation_path = save(tmp_path / "validation.pt", [3, 4])
    test_path = save(tmp_path / "test.pt", [5, 9])
    train, validation, test, observed_vocab = load_token_splits(
        tmp_path / "unused.pt", train_path, validation_path, test_path
    )
    assert train.tolist() == [0, 1, 2]
    assert validation.tolist() == [3, 4]
    assert test is not None and test.tolist() == [5, 9]
    assert observed_vocab == 10


def test_load_token_splits_requires_train_and_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provided together"):
        load_token_splits(tmp_path / "unused.pt", train_ids_path=tmp_path / "train.pt")
