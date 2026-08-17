#!/usr/bin/env bash

# Source this file before direct pytest/benchmark commands in the local WSL
# environment.  run_screen_all.sh applies the same setup automatically.
DOUBLE_ATTENTION_VENV="${DOUBLE_ATTENTION_VENV:-$HOME/.venvs/double-attention}"
DOUBLE_ATTENTION_PYTHON_DEV="${DOUBLE_ATTENTION_PYTHON_DEV:-$HOME/.local/python3.12-dev/usr/include}"

export PATH="$DOUBLE_ATTENTION_VENV/bin:$PATH"
if [[ -d "$DOUBLE_ATTENTION_PYTHON_DEV" ]]; then
  export CPATH="$DOUBLE_ATTENTION_PYTHON_DEV:$DOUBLE_ATTENTION_PYTHON_DEV/python3.12${CPATH:+:$CPATH}"
fi
