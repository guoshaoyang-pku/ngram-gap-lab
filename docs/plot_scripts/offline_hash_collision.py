#!/usr/bin/env python3
"""§46 offline hash-collision measurement: owned token mass vs table rows R.

Zero-GPU: uses only data/freq_index.npz (exact context hit counts on the
standard train shards) and the EXACT hash row assignment of the clean table
(train.py `_bigram_row_indices` / `_trigram_row_indices`, first hash family,
K=1 clean single table):

    bigram row  = ((prev * p1) ^ (cur  * p2)) % R
    trigram row = ((p2t * q1) ^ (p1 * q2) ^ (cur * q3)) % R

For each R we report three complementary quantities (context-count weighted vs
token-mass weighted -- the gap follows token mass, not context count):

  coll_rate   = (K - occupied) / K            context-level collision rate
  solo_frac   = token mass in rows with exactly 1 context
  owner_frac  = token mass where the row's dominant (max-f) context = owner

Source of truth: freq_index.npz (shard 1, data_seed 42, vocab 8192).
Output: docs/figs/theory/theory_hash_collision_mass.csv
        docs/figs/theory/fig_v5_hash_collision_mass.{png,svg}
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v5_style import apply_style, BRANCH_COLORS, save  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
NPZ = os.path.join(ROOT, "data", "freq_index.npz")
OUT_DIR = os.path.join(ROOT, "docs", "figs", "theory")

# exact primes from code/train.py (clean K=1 -> first hash family only)
BP1, BP2 = 2654435761, 2246822519
TQ1, TQ2, TQ3 = 16777619, 2166136261, 3432918353

RS = [2 ** k for k in range(10, 21)]


def rows_for(branch, keys):
    V = 8192
    if branch == "bigram":
        prev, cur = keys // V, keys % V
        return ((prev * BP1) ^ (cur * BP2)).astype(np.int64)
    prev2 = keys // (V * V)
    rem = keys % (V * V)
    prev1, cur = rem // V, rem % V
    return ((prev2 * TQ1) ^ (prev1 * TQ2) ^ (cur * TQ3)).astype(np.int64)


def measure(branch, keys, counts, R):
    rows = rows_for(branch, keys) % R
    ctx_row = np.bincount(rows, minlength=R)
    mass_row = np.bincount(rows, weights=counts.astype(np.float64), minlength=R)
    K = keys.size
    T = counts.sum()
    occ = int((ctx_row > 0).sum())
    solo_mass = mass_row[ctx_row == 1].sum()
    order = np.argsort(rows, kind="stable")
    rs, cs = rows[order], counts[order]
    idx = np.flatnonzero(np.r_[True, rs[1:] != rs[:-1]])
    owner_mass = np.maximum.reduceat(cs, idx).sum()
    return {
        "R": R,
        "K": K,
        "T": int(T),
        "occupied": occ,
        "coll_rate": (K - occ) / K,
        "solo_frac": float(solo_mass / T),
        "owner_frac": float(owner_mass / T),
    }


def main():
    apply_style()
    z = np.load(NPZ, allow_pickle=True)
    results = {"bigram": [], "trigram": []}
    for branch in ("bigram", "trigram"):
        keys = z[f"{branch}_keys"]
        counts = z[f"{branch}_counts"].astype(np.int64)
        for R in RS:
            m = measure(branch, keys, counts, R)
            m["branch"] = branch
            results[branch].append(m)

    csv_path = os.path.join(OUT_DIR, "theory_hash_collision_mass.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["branch", "R", "K", "T", "occupied", "coll_rate", "solo_frac", "owner_frac"],
        )
        w.writeheader()
        for branch in ("bigram", "trigram"):
            for m in results[branch]:
                w.writerow(m)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), sharey=True)
    for ax, branch in zip(axes, ("bigram", "trigram")):
        rows_ = results[branch]
        x = [m["R"] for m in rows_]
        c = BRANCH_COLORS[branch]
        ax.plot(x, [m["owner_frac"] for m in rows_], "-", color=c, lw=2.0,
                marker="o", ms=4, label="owner token mass (dominant ctx)")
        ax.plot(x, [m["solo_frac"] for m in rows_], "--", color=c, lw=1.4,
                marker="s", ms=3, label="solo token mass (no collision)")
        ax.plot(x, [m["coll_rate"] for m in rows_], ":", color="#666666", lw=1.4,
                marker="^", ms=3, label="context collision rate (K-weighted)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.4)
        ax.set_title(f"{branch}  (K={rows_[0]['K']:,}, T={rows_[0]['T']:,})", fontsize=10)
        ax.set_xlabel("table rows $R$ (per branch)")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("fraction")
    axes[0].legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.suptitle(
        "Offline hash-collision audit (exact training hash): owned token mass vs $R$",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, OUT_DIR, "fig_v5_hash_collision_mass")
    print("wrote", csv_path)
    for branch in ("bigram", "trigram"):
        m = results[branch][-1]
        print(f"{branch} @R=2^20: coll={m['coll_rate']:.4f} solo={m['solo_frac']:.4f} owner={m['owner_frac']:.4f}")


if __name__ == "__main__":
    main()
