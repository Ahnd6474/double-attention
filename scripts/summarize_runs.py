from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path, default=Path("runs/screen"))
    parser.add_argument(
        "--all",
        action="store_true",
        help="include shorter smoke/restart runs instead of only the longest run per variant",
    )
    args = parser.parse_args()
    rows = []
    for path in sorted(args.run_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            (
                result["variant"],
                result["parameters"],
                result["steps"],
                result["robust_mean"],
                result["perplexity"],
                path.name,
            )
        )
    if not rows:
        print(f"No completed runs in {args.run_dir}")
        return
    if not args.all:
        longest: dict[str, tuple[str, int, int, float, float, str]] = {}
        for row in rows:
            previous = longest.get(row[0])
            if previous is None or row[2] > previous[2]:
                longest[row[0]] = row
        rows = list(longest.values())
    best = min(row[3] for row in rows)
    print("variant      params       steps  robust_loss  delta_best  perplexity")
    for variant, parameters, steps, loss, perplexity, _ in sorted(rows, key=lambda row: row[3]):
        print(
            f"{variant:<11} {parameters:>11,} {steps:>7} "
            f"{loss:>12.6f} {loss-best:>11.6f} {perplexity:>11.3f}"
        )


if __name__ == "__main__":
    main()
