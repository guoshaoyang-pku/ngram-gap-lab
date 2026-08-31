#!/usr/bin/env bash
# Three-lane queue for the absolute-table-LR-locked backbone LR experiment.
set -uo pipefail

GPU="${1:?gpu id; registered lanes are 2, 4, and 5}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$GPU" in
  2)
    RUNS=(
      "blrabs_input_lr0p0006_tlr0p0768_10k|0.0006|10000|input"
      "blrabs_input_lr0p0003_tlr0p0768_20k|0.0003|20000|input"
    )
    ;;
  4)
    RUNS=(
      "blrabs_nogram_lr0p0006_tlr0p0768_10k|0.0006|10000|nogram"
      "blrabs_nogram_lr0p0003_tlr0p0768_20k|0.0003|20000|nogram"
    )
    ;;
  5)
    # High-LR points run first because they decide whether the old Fig. 18
    # downturn survives after removing the table-LR confound.
    RUNS=(
      "blrabs_input_lr0p0040_tlr0p0768_3k|0.004|3000|input"
      "blrabs_nogram_lr0p0040_tlr0p0768_3k|0.004|3000|nogram"
      "blrabs_input_lr0p0020_tlr0p0768_3k|0.002|3000|input"
      "blrabs_nogram_lr0p0020_tlr0p0768_3k|0.002|3000|nogram"
      "blrabs_input_lr0p0010_tlr0p0768_6k|0.001|6000|input"
      "blrabs_nogram_lr0p0010_tlr0p0768_6k|0.001|6000|nogram"
      "blrabs_input_lr0p0001_tlr0p0768_3k|0.0001|3000|input"
      "blrabs_nogram_lr0p0001_tlr0p0768_3k|0.0001|3000|nogram"
      "blrabs_input_lr0p00006_tlr0p0768_3k|0.00006|3000|input"
      "blrabs_nogram_lr0p00006_tlr0p0768_3k|0.00006|3000|nogram"
    )
    ;;
  *)
    echo "unregistered lane GPU=$GPU; use 2, 4, or 5" >&2
    exit 2
    ;;
esac

failures=0
for spec in "${RUNS[@]}"; do
  IFS='|' read -r run_id backbone_lr steps arm <<< "$spec"
  echo "[blr-queue] start $(date -Iseconds) gpu=$GPU run=$run_id"
  if "$SCRIPT_DIR/run_v5_backbone_lr_abslock.sh" \
      "$GPU" "$run_id" "$backbone_lr" "$steps" "$arm"; then
    echo "[blr-queue] done  $(date -Iseconds) gpu=$GPU run=$run_id"
  else
    rc=$?
    failures=$((failures + 1))
    echo "[blr-queue] FAIL  $(date -Iseconds) gpu=$GPU run=$run_id rc=$rc" >&2
  fi
done

echo "[blr-queue] lane complete gpu=$GPU failures=$failures"
exit "$failures"
