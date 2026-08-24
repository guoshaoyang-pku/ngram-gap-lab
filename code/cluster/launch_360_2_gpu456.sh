#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
exec "$PY" "$SCRIPT_DIR/rerun_all.py" --execute --root "$ROOT" --gpus 4,5,6 \
  --run-ids nglab2x_opt_rmsprop_2x_b2_09999_fixed,nglab2x_opt_rmsprop_2x_b2_099999_fixed,nglab2x_opt_rmsprop_4x_b2_09999_fixed