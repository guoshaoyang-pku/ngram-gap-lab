#!/usr/bin/env python3
"""extract_data.py — 从 data/runs_fixed/ 提取本附录需要的全部数据

输出：results/appendix_data.json
包含所有消融 run 的：配置 + 每 10 步的 train/val/gap 轨迹 + epoch 边界。

用法：
  python extract_data.py                 # 本地跑（从仓库根读取）
  python extract_data.py --remote ophis-gpu   # 先 rsync 集群上尚未拉回的 _fixed run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = REPO / "data" / "runs_fixed"
RUNS_PARTIAL = REPO / "data" / "runs_partial"   # 补点进行中的部分结果（快照）
OUT = HERE / "results" / "appendix_data.json"

# 本附录覆盖的全部 run（β₂ 消融 + 表学习率消融 + 短 epoch 对照）
RUN_IDS = {
    # --- β₂ 消融 · 1x shard · LR×2 ---
    "b2_098_1x_lr2": "nglab1x_opt_rmsprop_2x_b2_098_fixed",
    "b2_099_1x_lr2": "nglab1x_opt_rmsprop_2x_b2_099_fixed",
    "b2_0999_1x_lr2": "nglab1x_opt_rmsprop_2x_fixed",
    # --- β₂ 消融 · 1x shard · LR×4 ---
    "b2_098_1x_lr4": "nglab1x_opt_rmsprop_4x_b2_098_fixed",
    "b2_099_1x_lr4": "nglab1x_opt_rmsprop_4x_b2_099_fixed",
    "b2_0999_1x_lr4": "nglab1x_opt_rmsprop_4x_fixed",
    # --- β₂ 消融 · 2-epoch shard · LR×2 ---
    "b2_098_2ep_lr2": "nglab2x_opt_rmsprop_2x_b2_098_fixed",
    "b2_099_2ep_lr2": "nglab2x_opt_rmsprop_2x_b2_099_fixed",
    "b2_0999_2ep_lr2": "nglab2x_opt_rmsprop_2x_fixed",
    "b2_09999_2ep_lr2": "nglab2x_opt_rmsprop_2x_b2_09999_fixed",
    "b2_099999_2ep_lr2": "nglab2x_opt_rmsprop_2x_b2_099999_fixed",
    # --- β₂ 消融 · 2-epoch shard · LR×4 ---
    "b2_098_2ep_lr4": "nglab2x_opt_rmsprop_4x_b2_098_fixed",
    "b2_099_2ep_lr4": "nglab2x_opt_rmsprop_4x_b2_099_fixed",
    "b2_0999_2ep_lr4": "nglab2x_opt_rmsprop_4x_fixed",
    "b2_09999_2ep_lr4": "nglab2x_opt_rmsprop_4x_b2_09999_fixed",
    # --- β₂=0.99 · LR×1 补点（本附录补跑的） ---
    "b2_099_1x_lr1": "nglab1x_opt_rmsprop_b2_099_lr1_fixed",
    "b2_099_2ep_lr1": "nglab2x_opt_rmsprop_b2_099_lr1_fixed",
    # --- 学习率消融基线（β₂=0.999 默认） ---
    "lr1_1x": "nglab1x_v10_input_fixed",
    "lr1_2ep": "nglab2x_input_v10_fv_fixed",
    "nogram": "nglab1x_v10_nogram_fixed",
    # --- 短 epoch 家族 ---
    "short_025x_b2_099": "nglab025x_b2_099_fixed",
    "short_05x_b2_099": "nglab05x_b2_099_fixed",
    "short_025x_b2_0999": "nglab0_25x_input_fv_fixed",
    "short_05x_b2_0999": "nglab0_5x_input_fv_fixed",
}


def load_run(run_id: str) -> dict | None:
    d = RUNS / run_id
    partial = False
    if not d.exists():
        d = RUNS_PARTIAL / run_id          # 补点进行中：用部分快照
        partial = True
    if not d.exists():
        return None
    sp = d / "summary.json"
    if sp.exists():
        s = json.loads(sp.read_text())
        rec = {"run_id": run_id, "final_gap": s.get("final_gap"),
               "steps": s.get("steps"), "partial": False,
               "config": {k: s["config"][k] for k in
                          ("table_optimizer", "table_betas", "table_lr_scale",
                           "train_shards", "val_shards", "max_steps")
                          if k in s.get("config", {})}}
    else:
        # 部分快照：配置按补点定义写死（β₂=0.99 · 表 LR×1 · 极简 setting）
        rec = {"run_id": run_id, "final_gap": None, "steps": None,
               "partial": True,
               "config": {"table_optimizer": "rmsprop",
                          "table_betas": [0.0, 0.99], "table_lr_scale": 1.0}}
    traj = {"step": [], "train_loss": [], "val_loss": [], "gap": [],
            "epoch": [], "epoch_bounds": []}
    prev_ep = 0
    for line in (d / "train_log.jsonl").read_text().splitlines():
        r = json.loads(line)
        traj["step"].append(r["step"])
        traj["train_loss"].append(r["train_loss"])
        traj["val_loss"].append(r["val_loss"])
        traj["gap"].append(r["gap"])
        ep = r.get("epoch", 0)
        traj["epoch"].append(ep)
        if ep != prev_ep:
            if prev_ep > 0:
                traj["epoch_bounds"].append(r["step"])
            prev_ep = ep
    rec["traj"] = traj
    return rec


def maybe_sync_remote(host: str) -> None:
    """补点可能只在集群上。先把远端存在的 _fixed / b2_099_lr1 run 拉回。"""
    for rid in ("nglab1x_opt_rmsprop_b2_099_lr1",
                "nglab2x_opt_rmsprop_b2_099_lr1"):
        src = f"{host}:/data3/guoshaoyang/ngram-gap-lab/data/runs/{rid}/summary.json"
        test = subprocess.run(["rsync", "--dry-run", "-q", src, "/tmp/"],
                              capture_output=True)
        if test.returncode == 0:
            src_dir = f"{host}:/data3/guoshaoyang/ngram-gap-lab/data/runs/{rid}/"
            dst = RUNS / (rid + "_fixed") / ""
            dst_local = RUNS / (rid + "_fixed")
            dst_local.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-az", src_dir, str(dst) ], check=True)
            print(f"synced {rid} -> {dst_local.name}/")
        else:
            print(f"(remote {rid} not finished yet)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default=None,
                    help="sync unfinished runs from this ssh host first")
    args = ap.parse_args()

    if args.remote:
        maybe_sync_remote(args.remote)

    data, missing = {}, []
    for key, rid in RUN_IDS.items():
        rec = load_run(rid)
        if rec is None:
            missing.append(rid)
        else:
            data[key] = rec
    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT}  ({len(data)} runs)")
    if missing:
        print(f"missing (not finished / not pulled yet): {missing}")
    return None


if __name__ == "__main__":
    sys.exit(main())
