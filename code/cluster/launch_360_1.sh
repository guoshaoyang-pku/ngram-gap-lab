#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"

exec "$PY" "$SCRIPT_DIR/rerun_all.py" --execute --root "$ROOT" --gpus 1,3,4,5,6,7