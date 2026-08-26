#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
OUT_DIR="${NGLAB_SCALING_OUT_DIR:-$ROOT/data/runs_scaling}"
ROWS_TEXT="${V5_OCCUPANCY_ROWS:-16000 22000 30000 41000 56000 76000 104000 142000 194000 265000 362000 494000 675000 922000 1259000 1719000 2000000 2347000}"

if [[ "$PY" != */* ]]; then
  PY="$(command -v "$PY")"
fi

[[ -x "$PY" ]] || { echo "python unavailable: $PY" >&2; exit 2; }

for rows in $ROWS_TEXT; do
  run_id="ctbl_v5_both_${rows}"
  result_dir="$OUT_DIR/${run_id}_fixed"
  summary="$result_dir/summary.json"
  occupancy="$result_dir/table_occupancy.json"
  [[ -f "$summary" ]] || {
    echo "[v5-occupancy] missing completed run: $summary" >&2
    exit 2
  }
  if [[ -s "$occupancy" ]]; then
    echo "[v5-occupancy] skip complete $run_id"
    continue
  fi
  echo "[v5-occupancy] backfill $run_id R=$rows"
  "$PY" -u "$ROOT/code/table_occupancy.py" \
    --data_dir "$ROOT/data/tokenized" \
    --train_shards 1 \
    --vocab_size 8192 \
    --sequence_len 2048 \
    --device_batch_size 72 \
    --epoch_batches 337 \
    --bigram_clean_table "$rows" \
    --trigram_clean_table "$rows" \
    --out "$occupancy" \
    > "$result_dir/occupancy.log" 2>&1
  test -s "$occupancy"
done

echo "[v5-occupancy] complete"