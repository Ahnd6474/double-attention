from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def model_label(row: dict[str, object]) -> str:
    if row["variant"] == "mha4":
        return "MHA4"
    return f"{str(row['variant']).upper()} {row['dictionary']}"


def mean_std(values: list[float]) -> str:
    return f"{statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}"


def mean(values: list[float]) -> str:
    return f"{statistics.mean(values):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path, default=Path("runs/confirmation_v2"))
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.run_dir.rglob("*.json")]
    if not rows:
        raise SystemExit(f"no results below {args.run_dir}")

    print("| corpus | model | validation loss | paired vs. MHA4 | test loss | test vs. MHA4 |")
    print("|---|---|---:|---:|---:|---:|")
    for corpus in sorted({str(row["corpus"]) for row in rows}):
        corpus_rows = [row for row in rows if row["corpus"] == corpus]
        mha_by_seed = {int(row["seed"]): row for row in corpus_rows if row["variant"] == "mha4"}
        labels = sorted({model_label(row) for row in corpus_rows}, key=lambda item: item != "MHA4")
        for label in labels:
            model_rows = sorted(
                (row for row in corpus_rows if model_label(row) == label),
                key=lambda row: int(row["seed"]),
            )
            validation = [float(row["robust_mean"]) for row in model_rows]
            test = [float(row["test_mean"]) for row in model_rows]
            validation_delta = [
                float(row["robust_mean"]) - float(mha_by_seed[int(row["seed"])]["robust_mean"])
                for row in model_rows
            ]
            test_delta = [
                float(row["test_mean"]) - float(mha_by_seed[int(row["seed"])]["test_mean"])
                for row in model_rows
            ]
            print(
                f"| {corpus} | {label} | {mean_std(validation)} | "
                f"{mean_std(validation_delta)} | {mean_std(test)} | {mean_std(test_delta)} |"
            )

    print("\n| model | parameters | training time (min) | peak allocated (MiB) |")
    print("|---|---:|---:|---:|")
    for label in sorted({model_label(row) for row in rows}, key=lambda item: item != "MHA4"):
        model_rows = [row for row in rows if model_label(row) == label]
        parameters = {int(row["parameters"]) for row in model_rows}
        if len(parameters) != 1:
            raise SystemExit(f"inconsistent parameter counts for {label}: {parameters}")
        minutes = [
            sum(float(item["chunk_seconds"]) for item in row["history"]) / 60.0
            for row in model_rows
        ]
        memory = [
            max(float(item["max_memory_mib"]) for item in row["history"])
            for row in model_rows
        ]
        print(
            f"| {label} | {parameters.pop():,} | {mean(minutes)} | {mean(memory)} |"
        )

    print("\n| corpus | model | seed | validation vs. MHA4 | test vs. MHA4 |")
    print("|---|---|---:|---:|---:|")
    for corpus in sorted({str(row["corpus"]) for row in rows}):
        corpus_rows = [row for row in rows if row["corpus"] == corpus]
        mha_by_seed = {int(row["seed"]): row for row in corpus_rows if row["variant"] == "mha4"}
        for row in sorted(
            (row for row in corpus_rows if row["variant"] != "mha4"),
            key=lambda item: (model_label(item), int(item["seed"])),
        ):
            baseline = mha_by_seed[int(row["seed"])]
            validation_delta = float(row["robust_mean"]) - float(baseline["robust_mean"])
            test_delta = float(row["test_mean"]) - float(baseline["test_mean"])
            print(
                f"| {corpus} | {model_label(row)} | {row['seed']} | "
                f"{validation_delta:+.6f} | {test_delta:+.6f} |"
            )

    print("\n| corpus | model | step | mean validation vs. MHA4 | wins |")
    print("|---|---|---:|---:|---:|")
    for corpus in sorted({str(row["corpus"]) for row in rows}):
        corpus_rows = [row for row in rows if row["corpus"] == corpus]
        mha_history = {
            int(row["seed"]): {
                int(item["step"]): float(item["validation_loss"])
                for item in row["history"]
            }
            for row in corpus_rows
            if row["variant"] == "mha4"
        }
        for label in sorted(
            {model_label(row) for row in corpus_rows if row["variant"] != "mha4"}
        ):
            model_rows = [row for row in corpus_rows if model_label(row) == label]
            steps = sorted(
                set.intersection(
                    *[
                        {int(item["step"]) for item in row["history"]}
                        for row in model_rows
                    ]
                )
            )
            for step in steps:
                deltas = []
                for row in model_rows:
                    value = next(
                        float(item["validation_loss"])
                        for item in row["history"]
                        if int(item["step"]) == step
                    )
                    deltas.append(value - mha_history[int(row["seed"])][step])
                print(
                    f"| {corpus} | {label} | {step:,} | "
                    f"{statistics.mean(deltas):+.6f} | {sum(delta < 0 for delta in deltas)}/{len(deltas)} |"
                )


if __name__ == "__main__":
    main()
