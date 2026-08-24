#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
exec "$PY" "$SCRIPT_DIR/rerun_all.py" --execute --root "$ROOT" --gpus 0,2,3,4,5 \
  --run-ids nglab05x_b2_099_fixed,nglab025x_b2_099_fixed,nglab0_75x_e6_fixed,nglab0_25x_e6_fixed,nglab1x_opt_rmsprop_2x_s43_fixed,nglab1x_opt_adamw_090999_fixed,nglab1x_opt_adamw_090999_s44_fixed,nglab0_5x_e6_fixed