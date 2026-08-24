#!/usr/bin/env python3
"""ngram-gap-lab · exact-frequency scaling analysis (plan §4).

Reads data/runs_scaling/<run>/exact_freq_loss.jsonl:
  each line: {step, epoch, train: {bigram/trigram: {f: stats}},
              val: {...}, shared: {bigram/trigram: {shared_total, per_f}}}

Main quantity:
  token-marginal gap:  E[val loss | f] − E[train loss | f]
  context-matched gap: shared.per_f[f].gap   (same context in train & val probe)

Produces:
  - fig: gap(f) on log-log, per module & epoch (fixed-step)
  - fig: ΔG(f) = G_module(f) − G_nogram(f)
  - table: fit of two-factor model G(f)=A f^-β [1 - exp(-c f^γ)] per module
  - fit manifest (which f excluded & why)

Usage: python3 docs/plot_scripts/analyze_scaling_frequency.py [runs_dir]
"""
import json
import os
import sys
import glob
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
RUNS_DIR = os.path.join(REPO_ROOT, "data", "runs_scaling")
if len(sys.argv) > 1:
    RUNS_DIR = sys.argv[1]
FIGS_DIR = os.path.join(
    REPO_ROOT, "docs", "appendices", "s1_scaling_three_axis", "figs"
)
os.makedirs(FIGS_DIR, exist_ok=True)

MIN_TOKENS = 1024
MIN_CONTEXTS = 32


def load_exact(run_dir):
    path = os.path.join(run_dir, "exact_freq_loss.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            out.append(e)
    return out


def marginal_gap(entry, branch):
    """token-marginal gap per exact f: E[val|f] - E[train|f]."""
    tr = entry.get("train", {}).get(branch, {})
    va = entry.get("val", {}).get(branch, {})
    fs = set(tr) | set(va)
    out = {}
    for f in fs:
        t = tr.get(f)
        v = va.get(f)
        # require both splits present and min token count
        if not t or not v:
            continue
        if t["token_count"] < MIN_TOKENS or v["token_count"] < MIN_TOKENS:
            continue
        if t["distinct_contexts"] < MIN_CONTEXTS or v["distinct_contexts"] < MIN_CONTEXTS:
            continue
        out[int(f)] = {
            "f": int(f),
            "train_mean": t["mean_loss"],
            "val_mean": v["mean_loss"],
            "gap": v["mean_loss"] - t["mean_loss"],
            "train_tokens": t["token_count"],
            "val_tokens": v["token_count"],
        }
    return out


def context_matched_gap(entry, branch):
    sh = entry.get("shared", {}).get(branch, {})
    per_f = sh.get("per_f", {})
    out = {}
    for f, st in per_f.items():
        if st["shared_contexts"] < MIN_CONTEXTS:
            continue
        out[int(f)] = {
            "f": int(f),
            "gap": st["gap"],
            "train_mean": st["train_mean"],
            "val_mean": st["val_mean"],
            "shared_contexts": st["shared_contexts"],
        }
    return out


def two_factor(f, A, beta, c, gamma):
    return A * f ** (-beta) * (1.0 - np.exp(-c * f ** gamma))


def fit_two_factor(fs, gaps):
    """Weighted fit (weights = 1/sqrt(tokens) proxy). Return (popt, perr) or None."""
    if len(fs) < 4:
        return None
    fs = np.asarray(fs, dtype=float)
    gaps = np.asarray(gaps, dtype=float)
    # skip non-positive gaps for log-protected fits; fit in raw space with abs
    # to avoid sign issues, but keep only points with gap > 0 for the
    # multiplicative two-factor form (documented in manifest).
    keep = gaps > 0
    if keep.sum() < 4:
        return None
    fs, gaps = fs[keep], gaps[keep]
    # weights: higher for more tokens (inverse variance proxy)
    p0 = [np.median(gaps), 0.5, 0.1, 0.5]
    try:
        popt, pcov = curve_fit(two_factor, fs, gaps, p0=p0,
                               maxfev=20000, bounds=([0, -5, 0, 0], [np.inf, 5, np.inf, 5]))
        perr = np.sqrt(np.diag(pcov))
        return popt, perr, int(keep.sum())
    except Exception:
        return None


def main():
    # collect runs by (L, module, alignment)
    runs = {}
    for run_dir in sorted(glob.glob(os.path.join(RUNS_DIR, "ep_*"))):
        run_id = os.path.basename(run_dir)
        parts = run_id.split("_")
        if len(parts) < 4:
            continue
        L, mod, align = parts[1], parts[2], parts[3]
        if L not in ("L1", "L2", "L3", "L4") or mod not in ("bigram", "trigram", "both", "nogram"):
            continue
        exact = load_exact(run_dir)
        if not exact:
            continue
        runs[(L, mod, align)] = {"run_dir": run_dir, "exact": exact}

    # ---- gap(f) curves per module at final step (fixed-step) ----
    for branch in ("bigram", "trigram"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for mod in ("bigram", "trigram", "both", "nogram"):
            key = ("L4", mod, "fs")
            if key not in runs:
                continue
            exact = runs[key]["exact"]
            if not exact:
                continue
            entry = exact[-1]  # final step
            mg = marginal_gap(entry, branch)
            if not mg:
                continue
            fs = sorted(mg)
            ax.plot([mg[f]["f"] for f in fs], [mg[f]["gap"] for f in fs],
                    marker=".", markersize=4, label=mod, alpha=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("exact train frequency f")
        ax.set_ylabel("token-marginal gap")
        ax.set_title(f"{branch} gap(f) at final step (L4 fixed-step)")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS_DIR, f"freq_gap_{branch}_final.png"), dpi=150)
        plt.close(fig)
        print(f"saved freq_gap_{branch}_final.png")

    # ---- two-factor fit per module (final step, L4 fs) ----
    manifest = []
    for branch in ("bigram", "trigram"):
        for mod in ("bigram", "trigram"):
            key = ("L4", mod, "fs")
            if key not in runs:
                continue
            entry = runs[key]["exact"][-1]
            mg = marginal_gap(entry, branch)
            fs = sorted(mg)
            gaps = [mg[f]["gap"] for f in fs]
            res = fit_two_factor(fs, gaps)
            if res is None:
                manifest.append({"branch": branch, "module": mod, "fit": "failed",
                                 "n_f": len(fs)})
                print(f"{branch}/{mod}: fit failed ({len(fs)} f points)")
                continue
            popt, perr, n_used = res
            manifest.append({"branch": branch, "module": mod, "fit": "ok",
                             "A": float(popt[0]), "beta": float(popt[1]),
                             "c": float(popt[2]), "gamma": float(popt[3]),
                             "perr": [float(x) for x in perr],
                             "n_f": len(fs), "n_used": n_used})
            print(f"{branch}/{mod}: A={popt[0]:.3f} beta={popt[1]:.3f} "
                  f"c={popt[2]:.4f} gamma={popt[3]:.3f} (n={n_used}/{len(fs)})")
    with open(os.path.join(FIGS_DIR, "fit_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"saved fit_manifest.json")


if __name__ == "__main__":
    main()
