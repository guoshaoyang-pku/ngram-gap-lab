#!/usr/bin/env python3
"""Rerun all ngram-gap-lab experiments with the fixed train.py (freq-bin bug fix).

Generates launch commands for all experiment families. Every run uses:
  - freq_eval_interval=10, val_interval=10 (unified v10 standard)
  - NO --lr_schedule_epochs (pure step-based LR, no extra params)
  - _fixed suffix on run_id (never overwrites old data)

Usage:
  python3 rerun_all.py --list              # list all experiments
  python3 rerun_all.py --generate <cluster> # generate bash script for cluster
"""
import json
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import textwrap

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

# Shard chunk counts (verified from train.log train_chunks= fields).
# Shards 4,5,7,8 individual sizes unknown, but only needed as sums (see bpe_for_shards below).
SHARD_CHUNKS = {
    1: 24264, 2: 23976, 3: 23760,
    60: 12132, 61: 12132, 62: 6066, 63: 18198, 64: 12132,
}
# Verified sums for multi-shard training sets (from train.log headers)
SHARD_SET_CHUNKS = {
    (1, 2): 48240,
    (1, 2, 3): 72000,
    (1, 2, 3, 4, 5): 119808,       # 5x
    (1, 2, 3, 4, 5, 6): 143568,    # 6x
    (1, 2, 3, 4, 5, 6, 7, 8): 191016,  # 8x
}

def bpe_for_shards(shards):
    """batches per epoch for a list of shard ids."""
    key = tuple(sorted(shards))
    if key in SHARD_SET_CHUNKS:
        return SHARD_SET_CHUNKS[key] // 72
    return sum(SHARD_CHUNKS[s] for s in shards) // 72

def steps_for_epochs(shard_str, n_epochs):
    """Compute steps for n epochs given a comma-separated shard string."""
    shards = [int(s) for s in shard_str.split(",")]
    return bpe_for_shards(shards) * n_epochs

# Common args shared by ALL runs
COMMON = dict(
    seed=42,
    device_batch_size=72,
    total_batch_size=147456,
    val_interval=10,
    val_batches=4,
    table_norm_interval=10,
    lr=0.004,
    enable_unigram=0,
    enable_bigram=1,
    enable_trigram=1,
    n_layer=8,
    n_head=6,
    n_embd=768,
    vocab_size=8192,
    sequence_len=2048,
    freq_eval_interval=10,
    freq_eval_batches=4,
    table_optimizer="rmsprop",
    table_betas="0.0,0.99",
    table_lr_scale=2.0,
)

# Standard val shard sets
STD_VAL = "2,3,4,5,6,7,8,9,10,6542"
BIG_VAL = "3,4,5,6,7,8,9,10,6542"  # when train includes shard 2

def make_run(run_id, inj, steps, train_shards, val_shards, freq_index,
             extra=None, nogram=False):
    """Build a run spec dict."""
    spec = dict(COMMON)
    spec['run_id'] = run_id
    spec['injection_position'] = inj
    spec['steps'] = steps
    spec['train_shards'] = train_shards
    spec['val_shards'] = val_shards
    spec['freq_index'] = freq_index
    if nogram:
        spec['enable_bigram'] = 0
        spec['enable_trigram'] = 0
    if extra:
        spec.update(extra)
    return spec

# ---------------------------------------------------------------------------
# Experiment families
# ---------------------------------------------------------------------------

def all_runs(host_data_dir):
    """Return list of (family, run_spec) for all experiments to rerun."""
    FI = lambda name: f"{host_data_dir}/data/{name}"
    runs = []

    # 1. injpos_v10_1x: 4 runs, 2000 steps, shard 1
    for run_id, inj in [
        ("nglab1x_v10_v_fixed", "v"),
        ("nglab1x_v10_y_fixed", "y"),
        ("nglab1x_v10_input_fixed", "input"),
    ]:
        runs.append(("injpos_v10_1x", make_run(
            run_id, inj, 2000, "1", STD_VAL, FI("freq_index.npz"))))
    # nogram variant
    runs.append(("injpos_v10_1x", make_run(
        "nglab1x_v10_nogram_fixed", "input", 2000, "1", STD_VAL,
        FI("freq_index.npz"), nogram=True)))

    # 2. epoch_scale_v10: 2 runs, 2000 steps
    runs.append(("epoch_scale_v10", make_run(
        "nglab2x_input_v10_fv_fixed", "input", 2000, "1,2",
        "3,4,5,6,7,8,9,10,6542", FI("freq_index_train2x_fine.npz"))))
    runs.append(("epoch_scale_v10", make_run(
        "nglab0_5x_input_fv_fixed", "input", 2000, "60",
        STD_VAL, FI("freq_index_train0_5x.npz"))))

    # 3. shard_sweep: 6 runs (0.25x to 4x)
    runs.append(("shard_sweep", make_run(
        "nglab0_25x_input_fv_fixed", "input", 2000, "62",
        STD_VAL, FI("freq_index_train0_25x.npz"))))
    runs.append(("shard_sweep", make_run(
        "nglab0_75x_input_fv_fixed", "input", 2000, "63",
        STD_VAL, FI("freq_index_train0_75x.npz"))))
    runs.append(("shard_sweep", make_run(
        "nglab1_5x_input_fv_fixed", "input", 2000, "1,61",
        BIG_VAL, FI("freq_index_train1_5x.npz"))))
    runs.append(("shard_sweep", make_run(
        "nglab2_5x_input_fv_v3_fixed", "input", 3200, "1,2,64",
        "4,5,6,7,8,9,10,6542", FI("freq_index_train2_5x.npz"))))
    runs.append(("shard_sweep", make_run(
        "nglab3x_input_fv_v3_fixed", "input", 3800, "1,2,3",
        "4,5,6,7,8,9,10,6542", FI("freq_index_train3x.npz"))))
    runs.append(("shard_sweep", make_run(
        "nglab4x_input_fv_v3_fixed", "input", 5000, "1,2,3,4",
        "5,6,7,8,9,10,6542", FI("freq_index_train4x.npz"))))

    # 4. shard_sweep_360: 3 runs (5x/6x/8x), 2000 steps
    runs.append(("shard_sweep_360", make_run(
        "nglab5x_input_fv_fixed", "input", 2000, "1,2,3,4,5",
        "6,7,8,9,10,6542", FI("freq_index_train5x.npz"))))
    runs.append(("shard_sweep_360", make_run(
        "nglab6x_input_fv_fixed", "input", 2000, "1,2,3,4,5,6",
        "7,8,9,10,6542", FI("freq_index_train6x.npz"))))
    runs.append(("shard_sweep_360", make_run(
        "nglab8x_input_fv_fixed", "input", 2000, "1,2,3,4,5,6,7,8",
        "9,10,6542", FI("freq_index_train8x.npz"))))

    # 5. e6_series: 8 runs, 5 real epochs each
    # Original e6 targeted buggy-epoch=7; now we target real-epoch=5.
    # Val shards: must NOT overlap with train. Use the same val set as original e6 runs.
    # train=[62]→val=2..10,6542; [60]→val=2..10,6542; [63]→val=2..10,6542;
    # [1]→val=2..10,6542; [1,61]→val=3..10,6542; [1,2]→val=4..10,6542;
    # [1,2,64]→val=4..10,6542; [1,2,3]→val=5..10,6542
    e6_configs = [
        ("nglab0_25x_e6_fixed", "62",  STD_VAL),
        ("nglab0_5x_e6_fixed",  "60",  STD_VAL),
        ("nglab0_75x_e6_fixed", "63",  STD_VAL),
        ("nglab1x_e6_fixed",    "1",   STD_VAL),
        ("nglab1_5x_e6_fixed",  "1,61", BIG_VAL),
        ("nglab2x_e6_fixed",    "1,2",   "3,4,5,6,7,8,9,10,6542"),
        ("nglab2_5x_e6_fixed",  "1,2,64", "4,5,6,7,8,9,10,6542"),
        ("nglab3x_e6_fixed",    "1,2,3",  "4,5,6,7,8,9,10,6542"),
    ]
    for run_id, shard_str, val_shards in e6_configs:
        steps = steps_for_epochs(shard_str, 5)
        runs.append(("e6_series", make_run(
            run_id, "input", steps, shard_str,
            val_shards, FI("freq_index.npz"))))

    # 6. table_opt_1x: 13 runs
    #    adamw/sgd runs were 1000 steps; rmsprop runs were 2000 steps.
    #    Keep original steps.
    to1x = [
        ("nglab1x_opt_adamw_080950_fixed", 1000, 42, dict(table_optimizer="adamw", table_betas="0.8,0.95")),
        ("nglab1x_opt_adamw_090999_fixed", 1000, 42, dict(table_optimizer="adamw", table_betas="0.9,0.999")),
        ("nglab1x_opt_adamw_090999_s43_fixed", 1000, 43, dict(table_optimizer="adamw", table_betas="0.9,0.999")),
        ("nglab1x_opt_adamw_090999_s44_fixed", 1000, 44, dict(table_optimizer="adamw", table_betas="0.9,0.999")),
        ("nglab1x_opt_rmsprop_2x_fixed", 2000, 42, dict(table_optimizer="rmsprop", table_lr_scale=2.0)),
        ("nglab1x_opt_rmsprop_2x_b2_098_fixed", 2000, 42, dict(table_optimizer="rmsprop", table_lr_scale=2.0, table_betas="0.0,0.98")),
        ("nglab1x_opt_rmsprop_2x_b2_099_fixed", 2000, 42, dict(table_optimizer="rmsprop", table_lr_scale=2.0, table_betas="0.0,0.99")),
        ("nglab1x_opt_rmsprop_2x_s43_fixed", 1000, 43, dict(table_optimizer="rmsprop", table_lr_scale=2.0)),
        ("nglab1x_opt_rmsprop_2x_s44_fixed", 1000, 44, dict(table_optimizer="rmsprop", table_lr_scale=2.0)),
        ("nglab1x_opt_rmsprop_4x_fixed", 2000, 42, dict(table_optimizer="rmsprop", table_lr_scale=4.0)),
        ("nglab1x_opt_rmsprop_4x_b2_098_fixed", 2000, 42, dict(table_optimizer="rmsprop", table_lr_scale=4.0, table_betas="0.0,0.98")),
        ("nglab1x_opt_rmsprop_4x_b2_099_fixed", 2000, 42, dict(table_optimizer="rmsprop", table_lr_scale=4.0, table_betas="0.0,0.99")),
        ("nglab1x_opt_sgd_09_fixed", 1000, 42, dict(table_optimizer="sgd", table_betas="0.9,0.0")),
    ]
    for run_id, steps, seed, extra in to1x:
        e = dict(extra)
        e['seed'] = seed
        runs.append(("table_opt_1x", make_run(
            run_id, "input", steps, "1", STD_VAL,
            FI("freq_index.npz"), extra=e)))

    # 7. table_opt_2x: 10 runs, 2000 steps, shards 1,2
    to2x = [
        ("nglab2x_opt_rmsprop_1x_b2_09999_fixed", dict(table_optimizer="rmsprop", table_lr_scale=1.0, table_betas="0.0,0.9999")),
        ("nglab2x_opt_rmsprop_2x_fixed", dict(table_optimizer="rmsprop", table_lr_scale=2.0, table_betas="0.0,0.999")),
        ("nglab2x_opt_rmsprop_2x_b2_098_fixed", dict(table_optimizer="rmsprop", table_lr_scale=2.0, table_betas="0.0,0.98")),
        ("nglab2x_opt_rmsprop_2x_b2_099_fixed", dict(table_optimizer="rmsprop", table_lr_scale=2.0, table_betas="0.0,0.99")),
        ("nglab2x_opt_rmsprop_2x_b2_09999_fixed", dict(table_optimizer="rmsprop", table_lr_scale=2.0, table_betas="0.0,0.9999")),
        ("nglab2x_opt_rmsprop_2x_b2_099999_fixed", dict(table_optimizer="rmsprop", table_lr_scale=2.0, table_betas="0.0,0.99999")),
        ("nglab2x_opt_rmsprop_4x_fixed", dict(table_optimizer="rmsprop", table_lr_scale=4.0, table_betas="0.0,0.999")),
        ("nglab2x_opt_rmsprop_4x_b2_098_fixed", dict(table_optimizer="rmsprop", table_lr_scale=4.0, table_betas="0.0,0.98")),
        ("nglab2x_opt_rmsprop_4x_b2_099_fixed", dict(table_optimizer="rmsprop", table_lr_scale=4.0, table_betas="0.0,0.99")),
        ("nglab2x_opt_rmsprop_4x_b2_09999_fixed", dict(table_optimizer="rmsprop", table_lr_scale=4.0, table_betas="0.0,0.9999")),
    ]
    for run_id, extra in to2x:
        runs.append(("table_opt_2x", make_run(
            run_id, "input", 2000, "1,2",
            "3,4,5,6,7,8,9,10,6542",
            FI("freq_index_train2x_fine.npz"), extra=extra)))

    # 8. short_b2: 2 runs (0.25x and 0.5x with b2=0.99), 2000 steps
    runs.append(("short_b2", make_run(
        "nglab025x_b2_099_fixed", "input", 2000, "62",
        STD_VAL, FI("freq_index_train0_25x.npz"),
        extra=dict(table_optimizer="rmsprop", table_betas="0.0,0.99"))))
    runs.append(("short_b2", make_run(
        "nglab05x_b2_099_fixed", "input", 2000, "60",
        STD_VAL, FI("freq_index_train0_5x.npz"),
        extra=dict(table_optimizer="rmsprop", table_betas="0.0,0.99"))))

    # 9. Smoke verify + no-freq-bin control
    runs.append(("verify", make_run(
        "smoke_fixed_verify", "input", 400, "62",
        STD_VAL, FI("freq_index_train0_25x.npz"))))
    # No-freq-bin control: same as nglab1x_v10_input but without --freq_index
    ctrl = dict(COMMON)
    ctrl.update(dict(
        run_id="nglab1x_v10_input_nofb_fixed",
        injection_position="input",
        steps=2000,
        train_shards="1",
        val_shards=STD_VAL,
    ))
    ctrl.pop('freq_eval_interval')
    ctrl.pop('freq_eval_batches')
    runs.append(("verify", ctrl))

    return runs

def cmd_for_run(spec, host_root, host_data_dir, py="python3"):
    """Build the bash command string for a single run."""
    parts = [
        f'CUDA_VISIBLE_DEVICES="{{GPU}}" {py} -u {host_root}/code/train.py',
        f'--run_id {spec["run_id"]}',
        f'--injection_position {spec["injection_position"]}',
        f'--steps {spec["steps"]}',
        f'--seed {spec["seed"]}',
        f'--data_dir {host_data_dir}/data/tokenized',
        f'--out_dir {host_root}/data/runs_fixed',
        f'--train_shards {spec["train_shards"]}',
        f'--val_shards {spec["val_shards"]}',
        f'--device_batch_size {spec["device_batch_size"]}',
        f'--total_batch_size {spec["total_batch_size"]}',
        f'--val_interval {spec["val_interval"]}',
        f'--val_batches {spec["val_batches"]}',
        f'--table_norm_interval {spec["table_norm_interval"]}',
        f'--lr {spec["lr"]}',
        f'--enable_unigram {spec["enable_unigram"]}',
        f'--enable_bigram {spec["enable_bigram"]}',
        f'--enable_trigram {spec["enable_trigram"]}',
        f'--n_layer {spec["n_layer"]}',
        f'--n_head {spec["n_head"]}',
        f'--n_embd {spec["n_embd"]}',
        f'--vocab_size {spec["vocab_size"]}',
        f'--sequence_len {spec["sequence_len"]}',
    ]
    if 'freq_index' in spec and spec['freq_index']:
        parts.append(f'--freq_index {spec["freq_index"]}')
        parts.append(f'--freq_eval_interval {spec["freq_eval_interval"]}')
        parts.append(f'--freq_eval_batches {spec["freq_eval_batches"]}')
    if spec.get('table_optimizer'):
        parts.append(f'--table_optimizer {spec["table_optimizer"]}')
    if spec.get('table_lr_scale'):
        parts.append(f'--table_lr_scale {spec["table_lr_scale"]}')
    if spec.get('table_betas'):
        parts.append(f'--table_betas {spec["table_betas"]}')
    return " \\\n    ".join(parts)


def execute_runs(runs, root, gpus, selected_ids=None):
    by_id = {spec["run_id"]: (family, spec) for family, spec in runs}
    if selected_ids:
        missing = sorted(set(selected_ids) - set(by_id))
        if missing:
            raise SystemExit(f"unknown run id(s): {', '.join(missing)}")
        selected = [by_id[run_id] for run_id in selected_ids]
    else:
        selected = runs
    queues = [[] for _ in gpus]
    for index, item in enumerate(selected):
        queues[index % len(queues)].append(item)
    failures = []

    def run_queue(index):
        gpu = gpus[index]
        for family, spec in queues[index]:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"[GPU {gpu}] start {spec['run_id']} ({family})", flush=True)
            command = cmd_for_run(
                spec,
                str(root),
                str(root),
                os.environ.get("NGLAB_PY", "python3"),
            )
            result = subprocess.run(
                command.replace("{GPU}", str(gpu)),
                env=env,
                shell=True,
                executable="/bin/bash",
            )
            if result.returncode:
                failures.append(spec["run_id"])
                print(
                    f"[GPU {gpu}] failed {spec['run_id']} "
                    f"(exit={result.returncode})",
                    flush=True,
                )
            else:
                print(f"[GPU {gpu}] done {spec['run_id']}", flush=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(run_queue, range(len(queues))))
    if failures:
        raise SystemExit(
            "one or more runs failed: " + ", ".join(sorted(failures))
        )


def render_script(cluster):
    gpu_defaults = {
        "360-1": "1,3,4,5,6,7",
        "360-2": "4,5,6,7",
    }
    if cluster == "both":
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            'ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"',
            'PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"',
            'GPU_SET="${NGLAB_GPUS:-0}"',
            "",
            'exec "$PY" "$SCRIPT_DIR/rerun_all.py" --execute '
            '--root "$ROOT" --gpus "$GPU_SET"',
        ]
    else:
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            'ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"',
            'PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"',
            f'GPU_SET="${{NGLAB_GPUS:-{gpu_defaults[cluster]}}}"',
            "",
            'exec "$PY" "$SCRIPT_DIR/rerun_all.py" --execute '
            '--root "$ROOT" --gpus "$GPU_SET"',
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--generate", choices=["360-1", "360-2", "both"])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--root",
        default=os.environ.get("NGLAB_ROOT", str(REPO_ROOT)),
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument(
        "--run-ids",
        default="",
        help="comma-separated subset; omit to execute the full registered queue",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    host_data = str(root)

    runs = all_runs(host_data)

    if args.execute:
        gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
        if not gpus:
            raise SystemExit("--gpus must contain at least one GPU id")
        selected_ids = [
            run_id.strip() for run_id in args.run_ids.split(",") if run_id.strip()
        ]
        execute_runs(runs, root, gpus, selected_ids or None)
        raise SystemExit(0)

    if args.list:
        print(f"Total runs: {len(runs)}\n")
        cur_fam = None
        for fam, spec in runs:
            if fam != cur_fam:
                print(f"\n=== {fam} ===")
                cur_fam = fam
            fb = "yes" if spec.get('freq_index') else "NO"
            extra_str = ""
            if spec.get('table_optimizer'):
                extra_str += f" opt={spec['table_optimizer']}"
            if spec.get('table_lr_scale'):
                extra_str += f" lr{spec['table_lr_scale']}x"
            if spec.get('table_betas'):
                extra_str += f" b2={spec['table_betas'].split(',')[1]}"
            print(f"  {spec['run_id']:45s} steps={spec['steps']:5d} shards={spec['train_shards']:12s} fb={fb}{extra_str}")
        raise SystemExit(0)

    if args.generate:
        print(render_script(args.generate), end="")
        raise SystemExit(0)

    parser.error("choose one of --list, --generate, or --execute")
