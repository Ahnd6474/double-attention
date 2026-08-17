#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${DOUBLE_ATTENTION_PYTHON:-$HOME/.venvs/double-attention/bin/python}"
STEPS="${DOUBLE_ATTENTION_STEPS:-8000}"
SCHEDULE_STEPS="${DOUBLE_ATTENTION_SCHEDULE_STEPS:-12000}"
MICRO_BATCH="${DOUBLE_ATTENTION_MICRO_BATCH:-8}"
OUTPUT="${DOUBLE_ATTENTION_OUTPUT:-$ROOT/runs/tied}"

USER_PYTHON_DEV="${DOUBLE_ATTENTION_PYTHON_DEV:-$HOME/.local/python3.12-dev/usr/include}"
if [[ -d "$USER_PYTHON_DEV" ]]; then
  export CPATH="$USER_PYTHON_DEV:$USER_PYTHON_DEV/python3.12${CPATH:+:$CPATH}"
fi

mkdir -p "$OUTPUT"

# MHA4 has no dictionary and its existing 8k result is reused as the baseline.
variants=(a1 qk2-s1 qk1-s2 qk2-s2 qk4-s4)
for variant in "${variants[@]}"; do
  log="$OUTPUT/${variant}_tied_screen.log"
  echo "[$(date --iso-8601=seconds)] starting $variant tied" | tee -a "$log"
  "$PYTHON" "$ROOT/scripts/train_screen.py" \
    --variant "$variant" \
    --dictionary tied \
    --output-dir "$OUTPUT" \
    --steps "$STEPS" \
    --schedule-steps "$SCHEDULE_STEPS" \
    --micro-batch "$MICRO_BATCH" \
    --effective-batch 8 \
    --backend triton \
    --resume 2>&1 | tee -a "$log"
done
