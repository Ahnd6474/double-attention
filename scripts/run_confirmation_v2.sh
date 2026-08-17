#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${DOUBLE_ATTENTION_PYTHON:-$HOME/.venvs/double-attention/bin/python}"
STEPS="${DOUBLE_ATTENTION_STEPS:-8000}"
SCHEDULE_STEPS="${DOUBLE_ATTENTION_SCHEDULE_STEPS:-12000}"
WARMUP="${DOUBLE_ATTENTION_WARMUP:-600}"
MICRO_BATCH="${DOUBLE_ATTENTION_MICRO_BATCH:-8}"
OUTPUT="${DOUBLE_ATTENTION_OUTPUT:-$ROOT/runs/confirmation_v2}"

USER_PYTHON_DEV="${DOUBLE_ATTENTION_PYTHON_DEV:-$HOME/.local/python3.12-dev/usr/include}"
if [[ -d "$USER_PYTHON_DEV" ]]; then
  export CPATH="$USER_PYTHON_DEV:$USER_PYTHON_DEV/python3.12${CPATH:+:$CPATH}"
fi

corpora=(wikitext2 python_code)
models=("mha4:untied" "qk4-s4:untied" "qk2-s2:tied")
seeds=(0 1 2)
for corpus in "${corpora[@]}"; do
  train_ids="$ROOT/data/protocol_v2/${corpus}_train_ids.pt"
  validation_ids="$ROOT/data/protocol_v2/${corpus}_validation_ids.pt"
  test_ids="$ROOT/data/protocol_v2/${corpus}_test_ids.pt"
  for path in "$train_ids" "$validation_ids" "$test_ids"; do
    if [[ ! -f "$path" ]]; then
      echo "missing $path; run scripts/prepare_protocol_v2.py first" >&2
      exit 1
    fi
  done
  output_dir="$OUTPUT/$corpus"
  mkdir -p "$output_dir"
  for seed in "${seeds[@]}"; do
    for spec in "${models[@]}"; do
      variant="${spec%%:*}"
      dictionary="${spec##*:}"
      log="$output_dir/${variant}_${dictionary}_s${seed}.log"
      echo "[$(date --iso-8601=seconds)] starting $corpus $variant $dictionary seed=$seed" | tee -a "$log"
      "$PYTHON" "$ROOT/scripts/train_screen.py" \
        --variant "$variant" \
        --dictionary "$dictionary" \
        --corpus-name "$corpus" \
        --vocab-size 2048 \
        --train-ids "$train_ids" \
        --validation-ids "$validation_ids" \
        --test-ids "$test_ids" \
        --output-dir "$output_dir" \
        --seed "$seed" \
        --steps "$STEPS" \
        --schedule-steps "$SCHEDULE_STEPS" \
        --warmup "$WARMUP" \
        --micro-batch "$MICRO_BATCH" \
        --effective-batch 8 \
        --backend triton \
        --resume 2>&1 | tee -a "$log"
    done
  done
done
