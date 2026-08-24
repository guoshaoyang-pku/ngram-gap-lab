#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
exec "$PY" "$SCRIPT_DIR/rerun_all.py" --execute --root "$ROOT" --gpus 4,5,6,7 \
  --run-ids nglab4x_input_fv_v3_fixed,nglab1x_opt_rmsprop_2x_fixed,nglab2x_opt_rmsprop_2x_b2_098_fixed,nglab025x_b2_099_fixed,nglab1x_opt_adamw_090999_s44_fixed,nglab3x_e6_fixed,nglab1x_opt_rmsprop_2x_b2_098_fixed,nglab2x_opt_rmsprop_2x_b2_099_fixed,nglab05x_b2_099_fixed,nglab1x_opt_rmsprop_2x_s43_fixed,nglab2_5x_e6_fixed,nglab0_75x_input_fv_fixed,nglab2x_opt_rmsprop_1x_b2_09999_fixed,nglab2x_opt_rmsprop_4x_b2_099_fixed,nglab1x_opt_adamw_090999_fixed,nglab0_25x_e6_fixed,nglab3x_input_fv_v3_fixed,nglab0_5x_input_fv_fixed,nglab1x_opt_rmsprop_4x_b2_098_fixed,nglab2x_opt_rmsprop_4x_fixed,nglab0_75x_e6_fixed,nglab0_5x_e6_fixed