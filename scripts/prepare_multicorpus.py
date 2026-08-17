from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pyarrow.parquet as parquet
import sentencepiece as spm
import torch


WIKITEXT_REPOSITORY = "https://huggingface.co/datasets/Salesforce/wikitext"
SHAKESPEARE_REPOSITORY = "https://github.com/karpathy/char-rnn.git"


def run(*command: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def ensure_sources(source_dir: Path) -> tuple[Path, Path]:
    wiki = source_dir / "wikitext"
    if not (wiki / ".git").exists():
        clone_env = dict(os.environ)
        clone_env["GIT_LFS_SKIP_SMUDGE"] = "1"
        run(
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            WIKITEXT_REPOSITORY,
            str(wiki),
            env=clone_env,
        )
        run("git", "-C", str(wiki), "lfs", "pull", "--include=wikitext-2-raw-v1/*")

    shakespeare = source_dir / "char-rnn"
    if not (shakespeare / ".git").exists():
        run("git", "clone", "--depth", "1", SHAKESPEARE_REPOSITORY, str(shakespeare))
    return wiki, shakespeare


def write_wikitext(repository: Path, output: Path) -> None:
    parts = sorted((repository / "wikitext-2-raw-v1").glob("*.parquet"))
    if not parts:
        raise RuntimeError("WikiText-2 raw parquet files are missing")
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for part in parts:
            table = parquet.read_table(part, columns=["text"])
            for text in table.column("text").to_pylist():
                if text:
                    stream.write(text)
                    stream.write("\n")


def write_python_code(stdlib: Path, output: Path) -> int:
    files = sorted(path for path in stdlib.rglob("*.py") if "__pycache__" not in path.parts)
    if not files:
        raise RuntimeError(f"no Python files found below {stdlib}")
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for path in files:
            relative = path.relative_to(stdlib).as_posix()
            stream.write(f"\n\n# ===== {relative} =====\n\n")
            stream.write(path.read_text(encoding="utf-8", errors="ignore"))
    return len(files)


def copy_text(source: Path, output: Path) -> None:
    output.write_text(source.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def make_tokenizer_training(corpora: dict[str, Path], output: Path, characters: int) -> None:
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for name, path in corpora.items():
            text = path.read_text(encoding="utf-8", errors="ignore")[:characters]
            stream.write(f"\n\n===== CORPUS {name} =====\n\n")
            stream.write(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/multicorpus"))
    parser.add_argument("--docs", type=Path, default=Path("data/docs_sp2048/corpus.txt"))
    parser.add_argument("--python-stdlib", type=Path, default=Path("/usr/lib/python3.12"))
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--tokenizer-characters-per-corpus", type=int, default=1_000_000)
    args = parser.parse_args()

    root = args.output_dir.resolve()
    source_dir = root / "sources"
    corpus_dir = root / "corpora"
    source_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    wiki, shakespeare = ensure_sources(source_dir)

    corpora = {
        "python_docs": corpus_dir / "python_docs.txt",
        "wikitext2": corpus_dir / "wikitext2.txt",
        "shakespeare": corpus_dir / "shakespeare.txt",
        "python_code": corpus_dir / "python_code.txt",
    }
    copy_text(args.docs.resolve(), corpora["python_docs"])
    write_wikitext(wiki, corpora["wikitext2"])
    copy_text(shakespeare / "data" / "tinyshakespeare" / "input.txt", corpora["shakespeare"])
    python_files = write_python_code(args.python_stdlib.resolve(), corpora["python_code"])

    tokenizer_training = root / "tokenizer_training.txt"
    make_tokenizer_training(
        corpora,
        tokenizer_training,
        args.tokenizer_characters_per_corpus,
    )
    model_prefix = root / f"shared_sp{args.vocab_size}"
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
        "vocab_size": processor.vocab_size(),
        "tokenizer": str(model_path),
        "tokenizer_characters_per_corpus": args.tokenizer_characters_per_corpus,
        "sources": {
            "python_docs": {"description": "Python 3.14 text documentation archive"},
            "wikitext2": {
                "url": WIKITEXT_REPOSITORY,
                "commit": git_head(wiki),
                "config": "wikitext-2-raw-v1",
                "license": ["CC-BY-SA-3.0", "GFDL"],
            },
            "shakespeare": {
                "url": SHAKESPEARE_REPOSITORY,
                "commit": git_head(shakespeare),
                "path": "data/tinyshakespeare/input.txt",
                "repository_license": "MIT",
            },
            "python_code": {
                "path": str(args.python_stdlib.resolve()),
                "license": "PSF-2.0 with incorporated-software exceptions",
            },
        },
        "python_source_files": python_files,
        "corpora": {},
    }
    for name, corpus in corpora.items():
        ids = torch.tensor(processor.encode(corpus.read_text(encoding="utf-8")), dtype=torch.int32)
        ids_path = root / f"{name}_ids.pt"
        torch.save(ids, ids_path)
        metadata["corpora"][name] = {
            "text": str(corpus),
            "ids": str(ids_path),
            "bytes": corpus.stat().st_size,
            "tokens": ids.numel(),
            "sha256": sha256(corpus),
        }
        print(f"{name}: {corpus.stat().st_size:,} bytes, {ids.numel():,} tokens", flush=True)

    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
