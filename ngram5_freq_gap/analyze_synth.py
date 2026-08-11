#!/usr/bin/env python3
"""Summarize synthetic-transition pilot runs (order=5) on 360-2.

For each run and each probe step (val, constructed blocks) computes:
  - mean model CE on target positions (nats)
  - mean per-context conditional entropy H(c) (Bayes reference, nats)
  - excess CE = model CE - H(c)  (the signal to compare across schemes)
  - per-frequency mean gap (excess CE) with sample counts, at the final step

Usage:
  python3 analyze_synth.py --runs A_s42,B_s42 --dataset synth_A_sparse_restart \
      --out summary_synth.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RUNS_ROOT = Path("/data/home/guoshaoyang/ngram-gap-exp/runs/ngram5")
DATA_ROOT = Path("/data/home/guoshaoyang/ngram-gap-exp/ngram5_data")


def load_entropy(dataset: str) -> dict[tuple[int, ...], float]:
    t = np.load(DATA_ROOT / dataset / "transition_matrix.npz")
    P = t["probabilities"]
    H = -np.sum(P * np.log(np.maximum(P, 1e-30)), axis=1)
    return {tuple(map(int, c)): float(h) for c, h in zip(t["contexts"], H)}


def bucket_of(r: int, edges: list[int]) -> str:
    for i in range(len(edges) - 1):
        if edges[i] <= r < edges[i + 1]:
            return f"{edges[i]}-{edges[i+1]-1}"
    return f"{edges[-1]}+"


def analyze_run(run: str, dataset: str, entropy: dict, edges: list[int]) -> dict:
    rd = RUNS_ROOT / run
    steps = []
    per_step: dict[int, dict] = {}
    for line in (rd / "probe_details.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec["split"] != "val":
            continue
        step = rec["step"]
        z = np.load(rd / "probe_details" / Path(rec["path"]).name)
        losses = z["target_losses"].reshape(-1)
        ctx = z["contexts"].reshape(-1, 5)
        freq = z["frequencies"].reshape(-1)
        H = np.array([entropy[tuple(map(int, c))] for c in ctx])
        excess = losses - H
        per_step[step] = {
            "step": step,
            "n": int(len(losses)),
            "ce": float(losses.mean()),
            "entropy": float(H.mean()),
            "excess_ce": float(excess.mean()),
            "excess_std": float(excess.std()),
        }
    steps = [per_step[s] for s in sorted(per_step)]

    # final-step per-frequency breakdown
    last = per_step[max(per_step)]
    z = np.load(rd / "probe_details" / f"step_{last['step']:05d}_val.npz")
    losses = z["target_losses"].reshape(-1)
    ctx = z["contexts"].reshape(-1, 5)
    freq = z["frequencies"].reshape(-1)
    H = np.array([entropy[tuple(map(int, c))] for c in ctx])
    excess = losses - H

    per_freq: dict[int, dict] = {}
    for r in sorted(set(int(v) for v in freq)):
        m = freq == r
        per_freq[r] = {
            "frequency": int(r),
            "bucket": bucket_of(int(r), edges),
            "n": int(m.sum()),
            "ce": float(losses[m].mean()),
            "entropy": float(H[m].mean()),
            "excess_ce": float(excess[m].mean()),
        }
    return {
        "run": run,
        "dataset": dataset,
        "steps": steps,
        "final_step": last["step"],
        "per_frequency": list(per_freq.values()),
        "overall_excess_ce": last["excess_ce"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--edges", default="0,1,2,3,4,5,6,11,21,51,101,201,501,1001,5001")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    edges = [int(x) for x in args.edges.split(",")]
    entropy = load_entropy(args.dataset)
    results = [analyze_run(r, args.dataset, entropy, edges)
               for r in args.runs.split(",")]
    payload = {"dataset": args.dataset, "bucket_edges": edges, "runs": results}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    print(f"== {args.dataset} ==")
    for res in results:
        last = res["steps"][-1]
        print(f"{res['run']}: final step {last['step']} | CE={last['ce']:.4f} "
              f"H={last['entropy']:.4f} excess={last['excess_ce']:.4f}")
        for pf in res["per_frequency"]:
            print(f"  r={pf['frequency']:<5} bucket={pf['bucket']:<8} "
                  f"n={pf['n']:<4} CE={pf['ce']:.4f} H={pf['entropy']:.4f} "
                  f"excess={pf['excess_ce']:+.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
