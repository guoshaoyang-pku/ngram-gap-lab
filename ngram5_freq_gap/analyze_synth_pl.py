#!/usr/bin/env python3
"""Per-r-bucket gap analysis for the clean power-law toy (synth_pl).

For the final probe step, reads the train and val probe npz (same fixed
batches every step) and computes per-exact-r bucket:
    val CE(r), train CE(r), gap(r) = val CE - train CE, excess(r) = val CE - H,
and fits the log-log slope of gap(r) vs r (expect ~ -1, i.e. gap ~ (K_eff-1)/r).

Usage:
  python3 analyze_synth_pl.py --run synth_pl_A_nanogpt_s42 --dataset synth_pl_A \
      --out summary_pl.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RUNS_ROOT = Path("/data/home/guoshaoyang/ngram-gap-exp/runs/ngram5")
DATA_ROOT = Path("/data/home/guoshaoyang/ngram-gap-exp/ngram5_data")


def load_entropy_and_k(dataset: str) -> tuple[dict, dict, float]:
    t = np.load(DATA_ROOT / dataset / "transition_matrix.npz")
    P = t["probabilities"]
    H = -np.sum(P * np.log(np.maximum(P, 1e-30)), axis=1)
    Keff = np.exp(H)
    return (
        {tuple(map(int, c)): float(h) for c, h in zip(t["contexts"], H)},
        {tuple(map(int, c)): float(k) for c, k in zip(t["contexts"], Keff)},
        float(np.mean(Keff)),
    )


def loglog_fit(rs: np.ndarray, gs: np.ndarray):
    mask = (rs > 0) & (gs > 1e-12)
    x = np.log(rs[mask]); y = np.log(gs[mask])
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), r2, float(np.exp(intercept))


def analyze_run(run: str, dataset: str, entropy: dict, keff: dict, k_mean: float) -> dict:
    rd = RUNS_ROOT / run
    steps = []
    per_step: dict[int, dict] = {}
    for line in (rd / "probe_details.jsonl").read_text().splitlines():
        rec = json.loads(line)
        step = rec["step"]
        split = rec["split"]
        z = np.load(rd / "probe_details" / Path(rec["path"]).name)
        key = f"{split}_{step}"
        per_step.setdefault(step, {})[split] = z
    final_step = max(per_step)
    ztr = per_step[final_step]["train"]
    zva = per_step[final_step]["val"]
    for step, splits in sorted(per_step.items()):
        n = 0; ce = 0.0; excess = 0.0
        for split in ("train", "val"):
            z = splits[split]
            n += int(z["target_losses"].size)
            ce += float(z["target_losses"].mean())
        # only val has the entropy reference
        z = splits["val"]
        ctx = z["contexts"].reshape(-1, 5)
        H = np.array([entropy[tuple(map(int, c))] for c in ctx])
        excess = float((z["target_losses"].reshape(-1) - H).mean())
        steps.append({"step": step, "n": n, "ce_train": float(ztr["target_losses"].mean()) if step == final_step else None,
                      "ce_val": float(zva["target_losses"].mean()) if step == final_step else None,
                      "excess_ce": excess})
    steps.sort(key=lambda d: d["step"])

    # per-r bucket at final step
    def bucket_stats(z, Hmap=None):
        losses = z["target_losses"].reshape(-1)
        ctx = z["contexts"].reshape(-1, 5)
        freq = z["frequencies"].reshape(-1)
        out = {}
        for r in sorted(set(int(v) for v in freq)):
            m = freq == r
            d = {"n": int(m.sum()), "ce": float(losses[m].mean())}
            if Hmap is not None:
                H = np.array([Hmap[tuple(map(int, c))] for c in ctx[m]])
                d["entropy"] = float(H.mean())
                d["excess_ce"] = float(losses[m].mean() - H.mean())
            out[r] = d
        return out

    tr = bucket_stats(ztr)
    va = bucket_stats(zva, entropy)
    rs = sorted(set(tr) | set(va))
    rows = []
    for r in rs:
        g = va[r]["ce"] - tr[r]["ce"] if r in va and r in tr else float("nan")
        rows.append({
            "r": r,
            "n_val": va.get(r, {}).get("n", 0),
            "n_train": tr.get(r, {}).get("n", 0),
            "val_ce": va.get(r, {}).get("ce", float("nan")),
            "train_ce": tr.get(r, {}).get("ce", float("nan")),
            "gap": g,
            "excess_ce": va.get(r, {}).get("excess_ce", float("nan")),
            "entropy": va.get(r, {}).get("entropy", float("nan")),
            "pred_gap_k1_over_r": (k_mean - 1.0) / r,
        })
    rs_a = np.array([x["r"] for x in rows if x["n_val"] >= 20 and x["n_train"] >= 20])
    gs_a = np.array([x["gap"] for x in rows if x["n_val"] >= 20 and x["n_train"] >= 20])
    slope, r2, scale = loglog_fit(rs_a, gs_a)
    return {
        "run": run,
        "dataset": dataset,
        "k_mean_exp_h": k_mean,
        "final_step": final_step,
        "loglog_slope_gap": slope,
        "loglog_r2_gap": r2,
        "loglog_scale": scale,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    entropy, keff, k_mean = load_entropy_and_k(args.dataset)
    results = [analyze_run(r, args.dataset, entropy, keff, k_mean)
               for r in args.runs.split(",")]
    Path(args.out).write_text(json.dumps({"dataset": args.dataset, "runs": results},
                                         indent=2) + "\n")
    for res in results:
        print(f"== {res['run']}: K_eff={res['k_mean_exp_h']:.2f} slope={res['loglog_slope_gap']:.3f} "
              f"R2={res['loglog_r2_gap']:.4f} (scale={res['loglog_scale']:.2f})")
        print(f"   {'r':>5} {'n_val':>6} {'n_tr':>6} {'valCE':>7} {'trCE':>7} {'gap':>8} "
              f"{'excess':>8} {'(K-1)/r':>8}")
        for row in res["rows"]:
            print(f"   {row['r']:>5} {row['n_val']:>6} {row['n_train']:>6} {row['val_ce']:>7.3f} "
                  f"{row['train_ce']:>7.3f} {row['gap']:>8.4f} {row['excess_ce']:>8.4f} "
                  f"{row['pred_gap_k1_over_r']:>8.2f}")


if __name__ == "__main__":
    main()
