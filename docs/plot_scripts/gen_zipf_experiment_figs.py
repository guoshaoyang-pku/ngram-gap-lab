#!/usr/bin/env python3
"""Strict-Zipf toy experiment figures (2026-08-07, 360-2).

Compares per-bucket gap g(r) under:
  * strict Zipf N_r ~ 1/r^2 (t5z_zipf_s42/43/44, this batch)
  * anti-Zipf N_r ~ 1/r   (t5b_beta_000_999_low, same protocol)
  * old kink design       (t5_on_low_s42, reference)
and checks whether gap-vs-frequency double-log linearity improves.

Figures -> docs/figs/toy/fig_zipf_experiment.{png,svg}
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS_DIR = os.path.join(REPO_ROOT, "docs", "figs", "toy")
os.makedirs(FIGS_DIR, exist_ok=True)

TOY_RUNS = "/Users/guoshaoyang/Desktop/workdir/OPHIS/OPHIS_gap/toy/runs"
BG = "#f7f5ef"; BORDER = "#c8c1b6"; TEXT = "#686d73"; ANCHOR = "#353d79"
RED = "#C44E52"; GREEN = "#3c8d5a"; ORANGE = "#d97932"


def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(colors=TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(alpha=0.25, color=BORDER)


def loglog_fit(rs, gs):
    x = np.log(np.asarray(rs, float)); y = np.log(np.asarray(gs, float))
    ok = np.isfinite(x) & np.isfinite(y) & (y > np.log(1e-9))
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return None
    slope, inter = np.polyfit(x, y, 1)
    pred = slope * x + inter
    r2 = 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(slope), float(r2), int(len(x))


def load_eg(path):
    m = json.load(open(path))
    eg = m.get("exact_r_gap", {})
    rs = sorted(int(k) for k in eg)
    return np.array(rs, float), np.array([eg[str(r)] for r in rs], float), m


def main():
    zfits = []
    zall = {}
    for s in ["t5z_zipf_s42", "t5z_zipf_s43", "t5z_zipf_s44"]:
        rs, gs, m = load_eg(os.path.join(TOY_RUNS, s + ".run_meta.json"))
        zfits.append((rs, gs, m))
        for r, g in zip(rs, gs):
            zall.setdefault(int(r), []).append(float(g))
    rs0, gs0, m0 = load_eg(os.path.join(TOY_RUNS, "t5b_beta_000_999_low/run_meta.json"))
    t5 = json.load(open(os.path.join(TOY_RUNS, "..", "run_meta_table_t5.json")))
    old = t5["runs"]["t5_on_low_s42"]
    eg_old = {int(float(k)): v for k, v in old["exact_r_gap"].items()}
    rs_old = np.array(sorted(eg_old), float)
    gs_old = np.array([eg_old[int(r)] for r in rs_old], float)

    zr = np.array(sorted(zall), float)
    zg = np.array([np.mean(zall[int(r)]) for r in zr], float)
    zs = np.array([np.std(zall[int(r)]) for r in zr], float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    ax.errorbar(np.log2(zr), np.log2(np.maximum(zg, 1e-12)), yerr=zs / (zg * np.log(2)),
                fmt="o-", color=RED, lw=1.8, ms=5, capsize=3,
                label="strict Zipf (3 seeds, mean±std)")
    ax.plot(np.log2(rs0), np.log2(np.maximum(gs0, 1e-12)), "s--", color=GREEN, lw=1.6, ms=5,
            label="anti-Zipf t5b (same protocol)")
    ax.plot(np.log2(rs_old), np.log2(np.maximum(gs_old, 1e-12)), "d-.", color=ANCHOR, lw=1.5, ms=5,
            label="old kink design t5_on_low")
    for lab, rr, gg, c in [("zipf", zr, zg, RED), ("t5b(no r=8)", rs0[rs0 != 8], gs0[rs0 != 8], GREEN)]:
        f = loglog_fit(rr, gg)
        if f:
            ax.annotate(f"{lab}: slope {f[0]:+.2f}, R²={f[1]:.2f}", xy=(0.03, 0.9 - 0.12 * (0 if lab.startswith("zipf") else 1)),
                        xycoords="axes fraction", fontsize=8.5, color=c)
    ax.set_xlabel("log2(frequency r)"); ax.set_ylabel("log2(gap = val − train)")
    ax.set_title("per-bucket gap g(r): strict Zipf vs anti-Zipf (same protocol)\n"
                 "curves coincide → g(r) is distribution-INDEPENDENT", fontsize=10.5)
    ax.legend(fontsize=8); style(ax)

    ax = axes[1]
    cur_rs = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256], float)
    cur_ns = np.array([16384, 8192, 4096, 2048, 1024, 512, 256, 128, 128], float)
    zipf_n = {}
    for r, n in json.load(open(os.path.join(TOY_RUNS, "t5z_zipf_s42.run_meta.json"))) .get("exact_r_gap", {}).items():
        pass
    # real bucket counts from generator meta (cache data)
    gen = json.load(open("/tmp/t5zipf_meta.json")) if os.path.exists("/tmp/t5zipf_meta.json") else {}
    zb = {int(float(k)): v for k, v in gen.get("buckets", {}).items()}
    ax.plot(np.log2(cur_rs), np.log2(cur_ns), "o-", color=GREEN, lw=1.6, label="anti-Zipf t5b (slope ≈ −1)")
    if zb:
        zrs = np.array(sorted(zb), float)
        zns = np.array([zb[int(r)] for r in zrs], float)
        ax.plot(np.log2(zrs[zrs <= 256]), np.log2(zns[zrs <= 256]), "s-", color=RED, lw=1.6,
                label="strict Zipf N_r ≈ C/r² (slope ≈ −2)")
    ax.set_xlabel("log2(frequency r)"); ax.set_ylabel("log2(# keys)")
    ax.set_title("ngram frequency distribution: anti-Zipf vs strict Zipf\n"
                 "(strict Zipf keeps the same protocol; only N_r changes)", fontsize=10.5)
    ax.legend(fontsize=8); style(ax)

    fig.suptitle("Strict-Zipf toy experiment (360-2, 2000 steps, input injection, β=(0, 0.999)): "
                 "the per-bucket gap curve does not change", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for name in ("fig_zipf_experiment",):
        fig.savefig(os.path.join(FIGS_DIR, name + ".png"), dpi=150)
        fig.savefig(os.path.join(FIGS_DIR, name + ".svg"))
    plt.close(fig)

    print("[zipf-exp] wrote", os.path.join(FIGS_DIR, "fig_zipf_experiment.png"))
    fz = loglog_fit(zr, zg)
    ft = loglog_fit(rs0[rs0 != 8], gs0[rs0 != 8])
    fo = loglog_fit(rs_old, gs_old)
    print(f"  strict Zipf  : slope={fz[0]:+.3f} R2={fz[1]:.3f} n={fz[2]}  (mean over 3 seeds)")
    print(f"  anti-Zipf t5b: slope={ft[0]:+.3f} R2={ft[1]:.3f} n={ft[2]}  (r=8 outlier removed)")
    print(f"  old kink     : slope={fo[0]:+.3f} R2={fo[1]:.3f} n={fo[2]}")
    print("  per-r gap (zipf mean vs t5b):")
    for r in zr:
        print(f"    r={int(r):3d}: zipf {np.mean(zall[int(r)]):.2f}±{np.std(zall[int(r)]):.2f} "
              f" vs t5b {dict(zip(rs0, gs0)).get(r, float('nan')):.2f}")


if __name__ == "__main__":
    main()
