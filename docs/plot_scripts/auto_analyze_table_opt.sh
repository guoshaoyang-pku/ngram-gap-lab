#!/usr/bin/env bash
# Wait for the table-optimizer wave-2 runs to finish across ophis + 360-1 + 360-2,
# then rsync results locally and run the analysis automatically.
set -u
REPO=/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab
RUNS_DIR="$REPO/data/runs"
LOG="$REPO/docs/experiment-log.md"
MARKER="$REPO/data/table_opt_wave2_done.flag"

# host:path pairs (path relative to home on 360, absolute on ophis)
JOBS=(
  "ophis-gpu:/data3/guoshaoyang/ngram-gap-lab/data/runs/nglab1x_opt_sgd_09"
  "ophis-gpu:/data3/guoshaoyang/ngram-gap-lab/data/runs/nglab1x_opt_rmsprop_2x"
  "360-1:ngram-gap-lab/data/runs/nglab1x_opt_rmsprop_2x_s43"
  "360-1:ngram-gap-lab/data/runs/nglab1x_opt_adamw_090999_s43"
  "360-1:ngram-gap-lab/data/runs/nglab1x_opt_rmsprop_4x"
  "360-2:ngram-gap-lab/data/runs/nglab1x_opt_rmsprop_2x_s44"
  "360-2:ngram-gap-lab/data/runs/nglab1x_opt_adamw_090999_s44"
)

run_done() {
  local host="$1" path="$2" run_id
  run_id=$(basename "$path")
  # 1) summary.json exists?
  if ! ssh -o ConnectTimeout=8 "$host" "test -f $path/summary.json" 2>/dev/null; then
    return 1
  fi
  # 2) no live train.py with this run_id?
  if ssh -o ConnectTimeout=8 "$host" "ps -eo cmd | grep -F -- '--run_id $run_id' | grep -v grep" 2>/dev/null | grep -q .; then
    return 1
  fi
  return 0
}

sync_one() {
  local host="$1" path="$2" run_id
  run_id=$(basename "$path")
  if [ "$host" = "ophis-gpu" ]; then
    rsync -az "ophis-gpu:$path/" "$RUNS_DIR/$run_id/"
  else
    rsync -az "$host:${path#*/}" "$RUNS_DIR/$run_id/" 2>/dev/null || \
      rsync -az "$host:~/$path/" "$RUNS_DIR/$run_id/"
  fi
}

echo "[auto] waiting for wave-2 runs at $(date)"
while true; do
  pending=0
  for job in "${JOBS[@]}"; do
    host="${job%%:*}"; path="${job#*:}"
    if ! run_done "$host" "$path"; then
      pending=$((pending+1))
    fi
  done
  echo "[auto] $(date +%H:%M:%S) pending=$pending"
  if [ "$pending" -eq 0 ]; then
    break
  fi
  sleep 300
done

echo "[auto] all done, syncing at $(date)"
for job in "${JOBS[@]}"; do
  host="${job%%:*}"; path="${job#*:}"
  sync_one "$host" "$path"
  echo "[auto] synced $host:$path"
done

echo "[auto] running analysis at $(date)"
cd "$REPO" && python3 docs/plot_scripts/analyze_table_opt.py | tee data/table_opt_wave2_analysis.txt
echo "" >> "$LOG"
echo "### 9b. Table 优化器消融 wave2 完成（2026-08-07 自动回填）" >> "$LOG"
echo "" >> "$LOG"
echo '```' >> "$LOG"
cat data/table_opt_wave2_analysis.txt >> "$LOG"
echo '```' >> "$LOG"
touch "$MARKER"
echo "[auto] marker written: $MARKER"
