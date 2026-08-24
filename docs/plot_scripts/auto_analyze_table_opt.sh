#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-python3}"
RUNS_DIR="${NGLAB_RUNS_DIR:-$REPO/data/runs_fixed}"
DATA_DIR="${NGLAB_DATA_DIR:-$REPO/data}"
LOG="${NGLAB_EXPERIMENT_LOG:-$REPO/docs/experiment-log.md}"
MARKER="${NGLAB_TABLE_OPT_MARKER:-$DATA_DIR/table_opt_wave2_done.flag}"
REMOTE_JOBS="${NGLAB_REMOTE_JOBS:-}"
WAIT_SECONDS="${NGLAB_WAIT_SECONDS:-300}"

mkdir -p "$RUNS_DIR" "$DATA_DIR"

run_done() {
  local host="$1" remote_dir="$2" run_id
  run_id="$(basename "$remote_dir")"
  ssh -o ConnectTimeout=8 "$host" "test -f '$remote_dir/summary.json'" 2>/dev/null || return 1
  if ssh -o ConnectTimeout=8 "$host" \
    "pgrep -af -- 'train.py.*(--run_id[ =])$run_id([[:space:]]|$)' >/dev/null" \
    2>/dev/null; then
    return 1
  fi
  return 0
}

sync_one() {
  local host="$1" remote_dir="$2" run_id
  run_id="$(basename "$remote_dir")"
  mkdir -p "$RUNS_DIR/$run_id"
  rsync -az "$host:$remote_dir/" "$RUNS_DIR/$run_id/"
}

if [[ -n "$REMOTE_JOBS" ]]; then
  echo "[auto] waiting for configured remote runs at $(date)"
  while true; do
    pending=0
    while IFS= read -r job; do
      [[ -z "$job" ]] && continue
      host="${job%%:*}"
      remote_dir="${job#*:}"
      if ! run_done "$host" "$remote_dir"; then
        pending=$((pending + 1))
      fi
    done <<< "$REMOTE_JOBS"
    echo "[auto] $(date +%H:%M:%S) pending=$pending"
    [[ "$pending" -eq 0 ]] && break
    sleep "$WAIT_SECONDS"
  done

  while IFS= read -r job; do
    [[ -z "$job" ]] && continue
    sync_one "${job%%:*}" "${job#*:}"
    echo "[auto] synced $job"
  done <<< "$REMOTE_JOBS"
else
  echo "[auto] no NGLAB_REMOTE_JOBS configured; using local fixed runs"
fi

analysis_output="$DATA_DIR/table_opt_wave2_analysis.txt"
echo "[auto] running analysis at $(date)"
(
  cd "$REPO"
  NGLAB_RUNS_DIR="$RUNS_DIR" "$PY" docs/plot_scripts/analyze_table_opt.py
) | tee "$analysis_output"

{
  printf '\n### 9b. Table 优化器消融 wave2 自动分析\n\n```text\n'
  cat "$analysis_output"
  printf '```\n'
} >> "$LOG"

touch "$MARKER"
echo "[auto] marker written: $MARKER"
