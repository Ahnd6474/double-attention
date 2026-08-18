from __future__ import annotations

import argparse
import hashlib
import json
import sysconfig
from pathlib import Path

import pyarrow.parquet as parquet
import sentencepiece as spm
import torch

from prepare_multicorpus import WIKITEXT_REPOSITORY, ensure_sources, git_head, sha256


SPLITS = ("train", "validation", "test")


def write_wikitext_split(repository: Path, split: str, output: Path) -> int:
    parts = sorted((repository / "wikitext-2-raw-v1").glob(f"{split}-*.parquet"))
    if not parts:
        raise RuntimeError(f"WikiText-2 {split} parquet is missing")
    examples = 0
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for part in parts:
            table = parquet.read_table(part, columns=["text"])
            for text in table.column("text").to_pylist():
                if text:
                    stream.write(text)
                    stream.write("\n")
                    examples += 1
    return examples


def code_split(relative: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(relative.encode()).digest()[:8], "little") % 20
    if bucket == 0:
        return "validation"
    if bucket == 1:
        return "test"
    return "train"


def write_python_splits(stdlib: Path, outputs: dict[str, Path]) -> dict[str, int]:
    files = sorted(path for path in stdlib.rglob("*.py") if "__pycache__" not in path.parts)
    if not files:
        raise RuntimeError(f"no Python files found below {stdlib}")
    handles = {name: path.open("w", encoding="utf-8", newline="\n") for name, path in outputs.items()}
    counts = {name: 0 for name in SPLITS}
    try:
        for path in files:
            relative = path.relative_to(stdlib).as_posix()
            split = code_split(relative)
            stream = handles[split]
            stream.write(f"\n\n# ===== {relative} =====\n\n")
            stream.write(path.read_text(encoding="utf-8", errors="ignore"))
            counts[split] += 1
    finally:
        for stream in handles.values():
            stream.close()
    return counts


def make_tokenizer_training(train_files: dict[str, Path], output: Path, characters: int) -> None:
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for corpus, path in train_files.items():
            stream.write(f"\n\n===== TRAIN CORPUS {corpus} =====\n\n")
            stream.write(path.read_text(encoding="utf-8", errors="ignore")[:characters])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/protocol_v2"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/multicorpus/sources"))
    parser.add_argument(
        "--python-stdlib",
        type=Path,
        default=Path(sysconfig.get_path("stdlib")),
    )
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--tokenizer-characters-per-corpus", type=int, default=1_000_000)
    args = parser.parse_args()

    root = args.output_dir.resolve()
    text_dir = root / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    wiki, _ = ensure_sources(args.source_dir.resolve())

    text_paths = {
        corpus: {split: text_dir / f"{corpus}_{split}.txt" for split in SPLITS}
        for corpus in ("wikitext2", "python_code")
    }
    wiki_examples = {
        split: write_wikitext_split(wiki, split, text_paths["wikitext2"][split])
        for split in SPLITS
    }
    code_files = write_python_splits(args.python_stdlib.resolve(), text_paths["python_code"])

    tokenizer_training = root / "tokenizer_training.txt"
    make_tokenizer_training(
        {corpus: paths["train"] for corpus, paths in text_paths.items()},
        tokenizer_training,
        args.tokenizer_characters_per_corpus,
    )
    model_prefix = root / f"shared_train_only_sp{args.vocab_size}"
    model_path = model_prefix.with_suffix(".model")
    if not model_path.exists():
        spm.SentencePieceTrainer.train(
            input=str(tokenizer_training),
            model_prefix=str(model_prefix),
            vocab_size=args.vocab_size,
            model_type="unigram",
            character_coverage=1.0,
            input_sentence_size=0,
            shuffle_input_sentence=False,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
            unk_id=0,
            hard_vocab_limit=True,
        )

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    metadata: dict[str, object] = {
        "protocol": "v2_explicit_splits_train_only_tokenizer",
        "vocab_size": processor.vocab_size(),
        "tokenizer": str(model_path),
        "tokenizer_characters_per_corpus": args.tokenizer_characters_per_corpus,
        "sources": {
            "wikitext2": {
                "url": WIKITEXT_REPOSITORY,
                "commit": git_head(wiki),
                "config": "wikitext-2-raw-v1",
                "official_splits": True,
            },
            "python_code": {
                "path": str(args.python_stdlib.resolve()),
                "split": "sha256(relative_path) modulo 20: 18 train, 1 validation, 1 test",
            },
        },
        "wikitext_examples": wiki_examples,
        "python_files": code_files,
        "corpora": {},
    }
    for corpus, paths in text_paths.items():
        corpus_metadata: dict[str, object] = {}
        for split, text_path in paths.items():
            ids = torch.tensor(
                processor.encode(text_path.read_text(encoding="utf-8")),
                dtype=torch.int32,
            )
            ids_path = root / f"{corpus}_{split}_ids.pt"
            torch.save(ids, ids_path)
            corpus_metadata[split] = {
                "text": str(text_path),
                "ids": str(ids_path),
                "bytes": text_path.stat().st_size,
                "tokens": ids.numel(),
                "sha256": sha256(text_path),
            }
            print(f"{corpus}/{split}: {ids.numel():,} tokens", flush=True)
        metadata["corpora"][corpus] = corpus_metadata

    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
