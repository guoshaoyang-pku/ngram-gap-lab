#!/usr/bin/env bash
# One-time environment setup on ophis-gpu for ngram-gap-lab.
# Creates venv (reuses torch from existing ngram-gap-exp venv to avoid re-download).
set -euo pipefail

ROOT=/data3/guoshaoyang/ngram-gap-lab
VENV="$ROOT/.venv"

if [ -d "$VENV" ]; then
  echo "[setup] venv already exists at $VENV"
else
  echo "[setup] creating venv (reusing torch from ngram-gap-exp)..."
  # Reuse the existing venv's packages by creating a symlink-based venv
  python3 -m venv "$VENV"
  # Copy torch + numpy from existing venv site-packages
  SRC=/data3/guoshaoyang/ngram-gap-exp/.venv/lib/python*/site-packages
  DST="$VENV/lib/python*/site-packages"
  for pkg in torch numpy; do
    for src in $SRC/$pkg*; do
      [ -e "$src" ] && ln -sf "$src" $DST/ 2>/dev/null || true
    done
  done
fi

# Verify
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "[setup] done. venv: $VENV"
