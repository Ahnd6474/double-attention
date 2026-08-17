from __future__ import annotations

import argparse
import json
from pathlib import Path


def label(result: dict[str, object]) -> str:
    variant = str(result["variant"])
    dictionary = str(result.get("dictionary", "none"))
    return variant.upper() if dictionary == "none" else f"{variant.upper()} {dictionary}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path, default=Path("runs/multicorpus"))
    args = parser.parse_args()
    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.run_dir.rglob("*.json")]
    if not results:
        raise SystemExit(f"no JSON results below {args.run_dir}")

    print("| corpus | model | parameters | robust loss | vs. MHA4 | perplexity |")
    print("|---|---|---:|---:|---:|---:|")
    for corpus in sorted({str(row["corpus"]) for row in results}):
        rows = [row for row in results if row["corpus"] == corpus]
        mha = next(row for row in rows if row["variant"] == "mha4")
        for row in sorted(rows, key=lambda item: float(item["robust_mean"])):
            delta = float(row["robust_mean"]) - float(mha["robust_mean"])
            print(
                f"| {corpus} | {label(row)} | {int(row['parameters']):,} | "
                f"{float(row['robust_mean']):.6f} | {delta:+.6f} | "
                f"{float(row['perplexity']):.3f} |"
            )


if __name__ == "__main__":
    main()
