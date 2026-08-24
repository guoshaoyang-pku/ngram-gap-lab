#!/usr/bin/env python3
"""Assign runs to GPU queues and generate launch scripts.

Strategy: sort runs by steps (descending), greedily assign each run to the
GPU queue with the least total steps. Generates one bash script per cluster
that launches all GPU queues in parallel via nohup.

Usage:
  python3 assign_gpus.py --gpus 360-2:4,5,6,7 --gpus 360-1:1,3,6,7 [--skip handover_runs]
"""
import argparse
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from rerun_all import all_runs, cmd_for_run

# Runs already being handled by handover agent (don't relaunch)
HANDOVER_ACTIVE = {
    "nglab5x_input_fv_fixed",   # handover agent, correct params
    "nglab6x_input_fv_fixed",   # handover agent, correct params
    "nglab8x_input_fv_fixed",   # handover agent, correct params
    # nglab1x_e6_fixed is running with --lr_schedule_epochs 6 (WRONG), must rerun
}

# Runs already launched on 360-2 (first batch). All were KILLED due to old
# launch script with wrong e6 val_shards. Must rerun all.
ALREADY_RUNNING = set()

def assign(runs, gpu_lists, skip_running=True):
    """Assign runs to GPUs using LPT (Longest Processing Time first) scheduling."""
    skip = set(HANDOVER_ACTIVE)
    if skip_running:
        skip |= ALREADY_RUNNING
    to_assign = [(fam, spec) for fam, spec in runs
                 if spec['run_id'] not in skip]

    # Sort by steps descending (LPT)
    to_assign.sort(key=lambda x: x[1]['steps'], reverse=True)

    # Initialize GPU queues
    queues = {}  # gpu_key -> {cluster, gpu_id, runs: [], total_steps}
    for cluster, gpus in gpu_lists:
        for g in gpus:
            key = f"{cluster}:{g}"
            queues[key] = {'cluster': cluster, 'gpu': g, 'runs': [], 'total_steps': 0}

    # Greedy assignment
    for fam, spec in to_assign:
        # Find least loaded queue
        key = min(queues, key=lambda k: queues[k]['total_steps'])
        queues[key]['runs'].append((fam, spec))
        queues[key]['total_steps'] += spec['steps']

    return queues

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", action="append", required=True,
                        help="format: cluster:gpu1,gpu2,...")
    parser.add_argument(
        "--root",
        default=os.environ.get("NGLAB_ROOT", str(REPO_ROOT)),
        help="repository root used in generated commands",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR),
        help="directory for generated launch scripts",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_lists = []
    for spec in args.gpus:
        cluster, gpu_str = spec.split(":")
        gpus = [int(g) for g in gpu_str.split(",")]
        gpu_lists.append((cluster, gpus))

    host_root = os.path.abspath(args.root)
    runs = all_runs(host_root)
    queues = assign(runs, gpu_lists)

    # Print assignment
    total = 0
    for key, q in sorted(queues.items()):
        print(f"\n{'='*60}")
        print(f"GPU {key} (total_steps={q['total_steps']}, est={q['total_steps']*0.8/60:.0f}min)")
        print(f"{'='*60}")
        for fam, spec in q['runs']:
            print(f"  {spec['run_id']:45s} steps={spec['steps']:5d} [{fam}]")
            total += 1

    print(f"\nTotal assigned: {total} runs")
    print(f"Handover active (skipped): {len(HANDOVER_ACTIVE)} runs")

    # Generate per-cluster launch scripts
    clusters = {}
    for key, q in queues.items():
        c = q['cluster']
        if c not in clusters:
            clusters[c] = []
        clusters[c].append(q)

    for cluster, cl_queues in clusters.items():
        fname = output_dir / f"launch_{cluster.replace('-','_')}.sh"
        with fname.open('w') as f:
            f.write("#!/usr/bin/env bash\n")
            f.write(f"# Auto-generated launch script for {cluster}\n")
            f.write("# Each GPU runs its queue serially; all GPUs run in parallel.\n")
            f.write("# A failed run logs an error but does NOT stop the queue.\n")
            f.write("set -uo pipefail\n\n")
            f.write('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n')
            f.write('ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"\n')
            f.write('PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"\n')
            f.write('OUT_DIR="${NGLAB_OUT_DIR:-$ROOT/data/runs_fixed}"\n')
            f.write('LOGDIR="$ROOT/logs/rerun"\n')
            f.write('mkdir -p "$LOGDIR"\n\n')

            for q in cl_queues:
                gpu = q['gpu']
                f.write(f"# GPU {gpu}: {len(q['runs'])} runs, est {q['total_steps']*0.8/60:.0f}min\n")
                f.write(f'(\n')
                for i, (fam, spec) in enumerate(q['runs']):
                    cmd = cmd_for_run(spec, '"$ROOT"', '"$ROOT"', '"$PY"')
                    cmd = cmd.replace("{GPU}", str(gpu))
                    f.write(f'  echo "[GPU {gpu}] {i+1}/{len(q["runs"])}: {spec["run_id"]} at $(date)"\n')
                    f.write(f'  mkdir -p "$OUT_DIR/{spec["run_id"]}"\n')
                    f.write(f'  {cmd} > "$OUT_DIR/{spec["run_id"]}/train.log" 2>&1 || echo "[GPU {gpu}] {spec["run_id"]} FAILED"\n')
                f.write(f'  echo "[GPU {gpu}] queue done at $(date)"\n')
                f.write(f') &\n\n')

            f.write('echo "All GPU queues launched. Waiting..."\n')
            f.write('wait\n')
            f.write('echo "All done at $(date)"\n')

        print(f"\nGenerated: {fname}")

if __name__ == "__main__":
    main()
