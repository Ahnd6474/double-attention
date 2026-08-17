from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import sentencepiece as spm
import torch


DEFAULT_URL = "https://docs.python.org/3/archives/python-3.14-docs-text.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("data/docs_sp2048"))
    parser.add_argument("--vocab-size", type=int, default=2048)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "python-docs-text.zip"
    corpus = output_dir / "corpus.txt"
    model_prefix = output_dir / f"docs_sp{args.vocab_size}"
    ids_path = output_dir / f"docs_sp{args.vocab_size}_ids.pt"

    if not archive.exists():
        print(f"downloading {args.url}", flush=True)
        urllib.request.urlretrieve(args.url, archive)

    if not corpus.exists():
        with tempfile.TemporaryDirectory(prefix="double-attention-docs-") as temporary:
            root = Path(temporary)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(root)
            documents = sorted(root.rglob("*.txt"))
            if not documents:
                raise RuntimeError("documentation archive contained no .txt files")
            with corpus.open("w", encoding="utf-8", newline="\n") as output:
                for document in documents:
                    text = document.read_text(encoding="utf-8", errors="ignore")
                    output.write(f"\n\n===== {document.name} =====\n\n")
                    output.write(text)
        print(f"wrote {corpus} ({corpus.stat().st_size:,} bytes)", flush=True)

    model_path = model_prefix.with_suffix(".model")
    if not model_path.exists():
        spm.SentencePieceTrainer.train(
            input=str(corpus),
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
    token_ids = processor.encode(corpus.read_text(encoding="utf-8"), out_type=int)
    ids = torch.tensor(token_ids, dtype=torch.int32)
    torch.save(ids, ids_path)
    metadata = {
        "source_url": args.url,
        "corpus_bytes": corpus.stat().st_size,
        "tokens": ids.numel(),
        "vocab_size": processor.vocab_size(),
        "model": str(model_path),
        "ids": str(ids_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
