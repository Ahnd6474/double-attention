#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${DOUBLE_ATTENTION_PYTHON:-$HOME/.venvs/double-attention/bin/python}"
STEPS="${DOUBLE_ATTENTION_STEPS:-2000}"
SCHEDULE_STEPS="${DOUBLE_ATTENTION_SCHEDULE_STEPS:-3000}"
WARMUP="${DOUBLE_ATTENTION_WARMUP:-150}"
MICRO_BATCH="${DOUBLE_ATTENTION_MICRO_BATCH:-8}"
OUTPUT="${DOUBLE_ATTENTION_OUTPUT:-$ROOT/runs/multicorpus}"

USER_PYTHON_DEV="${DOUBLE_ATTENTION_PYTHON_DEV:-$HOME/.local/python3.12-dev/usr/include}"
if [[ -d "$USER_PYTHON_DEV" ]]; then
  export CPATH="$USER_PYTHON_DEV:$USER_PYTHON_DEV/python3.12${CPATH:+:$CPATH}"
fi

corpora=(python_docs wikitext2 shakespeare python_code)
models=("mha4:untied" "qk4-s4:untied" "qk2-s2:tied" "qk1-s2:tied")
for corpus in "${corpora[@]}"; do
  ids="$ROOT/data/multicorpus/${corpus}_ids.pt"
  if [[ ! -f "$ids" ]]; then
    echo "missing $ids; run scripts/prepare_multicorpus.py first" >&2
    exit 1
  fi
  output_dir="$OUTPUT/$corpus"
  mkdir -p "$output_dir"
  for spec in "${models[@]}"; do
    variant="${spec%%:*}"
    dictionary="${spec##*:}"
    log="$output_dir/${variant}_${dictionary}_screen.log"
    echo "[$(date --iso-8601=seconds)] starting $corpus $variant $dictionary" | tee -a "$log"
    "$PYTHON" "$ROOT/scripts/train_screen.py" \
      --variant "$variant" \
      --dictionary "$dictionary" \
      --corpus-name "$corpus" \
      --vocab-size 2048 \
      --ids "$ids" \
      --output-dir "$output_dir" \
      --steps "$STEPS" \
      --schedule-steps "$SCHEDULE_STEPS" \
      --warmup "$WARMUP" \
      --micro-batch "$MICRO_BATCH" \
      --effective-batch 8 \
      --backend triton \
      --resume 2>&1 | tee -a "$log"
  done
done
