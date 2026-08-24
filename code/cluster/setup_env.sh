#!/usr/bin/env bash
# One-time environment setup on ophis-gpu for ngram-gap-lab.
# Creates the repository-local virtual environment.
set -euo pipefail

ROOT="${NGLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="$ROOT/.venv"
TORCH_SOURCE="${NGLAB_TORCH_SOURCE:-}"

if [ -d "$VENV" ]; then
  echo "[setup] venv already exists at $VENV"
else
  echo "[setup] creating venv..."
  python3 -m venv "$VENV"
  if [[ -n "$TORCH_SOURCE" ]]; then
    SRC="$TORCH_SOURCE/lib/python*/site-packages"
    DST="$VENV/lib/python*/site-packages"
    for pkg in torch numpy; do
      for src in $SRC/$pkg*; do
        [ -e "$src" ] && ln -sf "$src" $DST/ 2>/dev/null || true
      done
    done
  fi
fi

# Verify
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "[setup] done. venv: $VENV"
