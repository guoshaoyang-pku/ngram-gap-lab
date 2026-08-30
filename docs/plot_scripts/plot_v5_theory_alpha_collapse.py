#!/usr/bin/env python3
"""Theory audit: (1) Zipf exponent alpha of true context frequencies from
data/freq_index.npz (1x shard-1 index), compared against the exponent-triangle
prediction gamma = 1 - alpha*(1-beta); (2) pass-count collapse figure that
overlays the 10-epoch long replay, the wrap-around (>1xL4) epoch-length points
and the dose axis on a single "completed passes" axis.

Inputs (all existing artifacts, no new runs):
  data/freq_index.npz                                    -- exact train counts, shard 1
  docs/appendices/s1_scaling_three_axis/s1_epoch_long_replay_points.csv
  docs/appendices/s1_scaling_three_axis/s1_epoch_length_points.csv
  docs/appendices/s1_scaling_three_axis/s1_dose_points.csv

Outputs:
  docs/figs/theory/fig_v5_zipf_context_alpha.png
  docs/figs/main/fig_v5_pass_collapse.png
  docs/figs/theory/theory_zipf_triangle.csv
"""
import csv
import hashlib
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "..", "data", "freq_index.npz")
NPZ = os.path.normpath(NPZ)
APP = os.path.join(ROOT, "appendices", "s1_scaling_three_axis")
FIG_THEORY = os.path.join(ROOT, "figs", "theory")
FIG_MAIN = os.path.join(ROOT, "figs", "main")

# measured v5 exponents (single seed 42, descriptive fits; see s1_scaling_analysis.md)
BETA = {"bigram": 0.252746, "trigram": 0.318121}
GAMMA_NET = {"bigram": 0.5761, "trigram": 0.6648}
GAMMA_RAW = {"bigram": 0.5010, "trigram": 0.6526}
RANK_WINDOW = {"bigram": (2e3, 2e5), "trigram": (1e5, 9.3e5)}  # = table-size fit windows


def sha256(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def loglog_slope(x, y):
    lx, ly = np.log(x), np.log(y)
    a, b = np.polyfit(lx, ly, 1)
    pred = a * lx + b
    ss = 1.0 - ((ly - pred) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    return a, ss


def main():
    print("freq_index.npz sha256 =", sha256(NPZ))
    z = np.load(NPZ)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, br in zip(axes, ("bigram", "trigram")):
        c = np.sort(z[br + "_counts"].astype(np.int64))[::-1]
        K, T = c.size, int(c.sum())
        ranks = np.arange(1, K + 1, dtype=np.float64)
        lo, hi = RANK_WINDOW[br]
        m = (ranks >= lo) & (ranks <= hi)
        alpha_win, r2_win = loglog_slope(ranks[m], c[m].astype(float))
        alpha_win = -alpha_win
        # sensitivity windows
        sens = {}
        for wlo, whi in ((1e3, 1e5), (1e4, 1e6)):
            mm = (ranks >= wlo) & (ranks <= min(whi, K))
            s, _ = loglog_slope(ranks[mm], c[mm].astype(float))
            sens[f"{wlo:g}-{whi:g}"] = -s
        # count-of-counts tail exponent -> alpha_cc via n(f) ~ f^-(1+1/alpha)
        cc = np.bincount(c[c <= 1000].astype(np.int64))
        fvals = np.arange(1, 101)
        nz = cc[1:101] > 0
        s_cc, r2_cc = loglog_slope(fvals[nz].astype(float), cc[1:101][nz].astype(float))
        alpha_cc = 1.0 / (-s_cc - 1.0)
        # f at window edges + token-mass fractions
        f_edges = [int(c[int(r) - 1]) for r in (lo, hi)]
        mass = {t: c[c <= t].sum() / T for t in (1, 2, 4, 8, 200)}
        # triangle closure
        a_need_net = (1 - GAMMA_NET[br]) / (1 - BETA[br])
        a_need_raw = (1 - GAMMA_RAW[br]) / (1 - BETA[br])
        g_pred = 1 - alpha_win * (1 - BETA[br])
        print(f"\n[{br}] K={K:,} T={T:,} load@2^20={K/2**20:.2f}")
        print(f"  rank-window {lo:g}-{hi:g}: alpha={alpha_win:.3f} (R2={r2_win:.4f}), "
              f"f at window edges={f_edges}")
        print(f"  sensitivity: {sens}")
        print(f"  count-of-counts alpha_cc={alpha_cc:.3f} (slope {s_cc:.3f}, R2={r2_cc:.4f})")
        print(f"  token-mass fraction f<=1/2/4/8/200: "
              + "/".join(f"{mass[t]:.3f}" for t in (1, 2, 4, 8, 200)))
        print(f"  triangle: gamma_pred=1-alpha(1-beta)={g_pred:.3f} "
              f"vs gamma_net={GAMMA_NET[br]} gamma_raw={GAMMA_RAW[br]}; "
              f"alpha needed: net={a_need_net:.3f} raw={a_need_raw:.3f}")
        rows.append(dict(branch=br, K=K, T=T, alpha_rank_window=round(alpha_win, 4),
                         r2=round(r2_win, 4), window_lo=lo, window_hi=hi,
                         f_at_lo=f_edges[0], f_at_hi=f_edges[1],
                         alpha_count_of_counts=round(alpha_cc, 4),
                         beta_meas=BETA[br], gamma_net=GAMMA_NET[br],
                         gamma_raw=GAMMA_RAW[br], gamma_pred=round(g_pred, 4),
                         alpha_needed_net=round(a_need_net, 4),
                         alpha_needed_raw=round(a_need_raw, 4),
                         mass_f_le_8=round(float(mass[8]), 4)))
        # plot rank-frequency (subsampled for file size)
        idx = np.unique(np.round(np.logspace(0, np.log10(K - 1), 4000)).astype(int))
        ax.loglog(ranks[idx], c[idx - 1], lw=1.2, color="#1f77b4", label="rank-frequency")
        ax.axvspan(lo, hi, color="orange", alpha=0.18,
                   label=f"table-size fit window\nlocal alpha={alpha_win:.3f}")
        ax.set_title(f"{br} contexts (K={K:,}, shard-1 train index)")
        ax.set_xlabel("rank")
        ax.set_ylabel("train hit count f")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)
    fig.suptitle("True context Zipf structure vs table-size fit windows "
                 "(data/freq_index.npz, seed-42 1x shard)", fontsize=11)
    fig.tight_layout()
    out1 = os.path.join(FIG_THEORY, "fig_v5_zipf_context_alpha.png")
    fig.savefig(out1, dpi=160)
    print("saved", out1)

    with open(os.path.join(FIG_THEORY, "theory_zipf_triangle.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------------- pass-count collapse ----------------
    def read_csv(name):
        with open(os.path.join(APP, name)) as f:
            return list(csv.DictReader(f))

    replay = [(float(r["epoch"]), float(r["gap"])) for r in read_csv("s1_epoch_long_replay_points.csv")
              if r["arm"] == "trigram-only"]
    nog = [(float(r["epoch"]), float(r["gap"])) for r in read_csv("s1_epoch_long_replay_points.csv")
           if r["arm"] == "nogram"]
    wrap = []
    for r in read_csv("s1_epoch_length_points.csv"):
        mult = float(r["epoch_multiplier_L4"])
        if mult <= 1.0:
            continue  # true epoch-length variation, different data volume -> not overlayable
        for e, key in ((1, "gap_epoch_1"), (2, "gap_epoch_2"), (3, "gap_epoch_3")):
            wrap.append((mult * e, float(r[key])))

    # dose axis: use the 128x batch (nglab*_input_v5_128x_freq10), NOT the 2x
    # batch recorded in s1_dose_points.csv (config table_lr_scale=2.0).
    import json
    runs_fixed = os.path.normpath(os.path.join(ROOT, "..", "data", "runs_fixed"))
    dose_names = {0.25: "0_25x", 0.5: "0_5x", 0.75: "0_75x", 1.5: "1_5x",
                  2.0: "2x", 2.5: "2_5x", 3.0: "3x", 4.0: "4x", 5.0: "5x",
                  6.0: "6x", 8.0: "8x"}
    dose, dose_rows = [], []
    for d in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
        rid = "nglab1x_input_v5_128x_freq10" if d == 1.0 else f"nglab{dose_names[d]}_input_v5_128x_freq10"
        s = json.load(open(os.path.join(runs_fixed, rid + "_fixed", "summary.json")))
        dose.append((2000.0 / (337.0 * d), s["final_gap"]))
        dose_rows.append(dict(axis="dose_128x", run_id=rid, seed=42, dose=d, steps=2000,
                              final_train_loss=s.get("final_train_loss"),
                              final_val_loss=s.get("final_val_loss"),
                              final_gap=s["final_gap"],
                              source=f"data/runs_fixed/{rid}_fixed"))
    with open(os.path.join(APP, "s1_dose_points_128x.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dose_rows[0].keys()))
        w.writeheader()
        w.writerows(dose_rows)
    pos = [(d["dose"], d["final_gap"]) for d in dose_rows if d["final_gap"] > 0 and d["dose"] <= 5]
    sl, r2p = loglog_slope(np.array([p[0] for p in pos]), np.array([p[1] for p in pos]))
    print(f"dose 128x batch: log-log slope={sl:.4f} R2={r2p:.4f} n={len(pos)}")

    fig2, ax = plt.subplots(figsize=(8.6, 5.4))
    for pts, lab, kw in (
        (replay, "10-epoch long replay, trigram-only (s1v5_128_ep_tri_1xL4_10ep)",
         dict(marker="o", color="#d62728", s=42, zorder=4)),
        (wrap, ">1xL4 'epoch-length' points = wrap-around replay of shard 1 (trigram-only)",
         dict(marker="s", color="#ff9f40", s=34, zorder=3)),
        (dose, "dose axis 128x, input arm bi+tri, step 2000 (nglab*_input_v5_128x_freq10)",
         dict(marker="^", color="#1f77b4", s=38, zorder=2)),
        (nog, "no-gram control, same replay schedule",
         dict(marker="x", color="#7f7f7f", s=30, zorder=1)),
    ):
        xs, ys = zip(*sorted(pts))
        ax.scatter(xs, ys, label=lab, **kw)
    rx, ry = zip(*sorted(replay))
    ax.plot(rx, ry, color="#d62728", lw=0.8, alpha=0.6)
    ax.axvline(1.0, color="k", lw=0.8, ls="--")
    ax.text(1.04, 6.2, "1 pass:\ngap crosses 0", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("completed passes over the training set (log)")
    ax.set_ylabel("online gap = fixed val - current-batch train")
    ax.set_title("Pass-count collapse: raw points, no smoothing (seed 42)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    fig2.tight_layout()
    out2 = os.path.join(FIG_MAIN, "fig_v5_pass_collapse.png")
    fig2.savefig(out2, dpi=160)
    print("saved", out2)


if __name__ == "__main__":
    main()
