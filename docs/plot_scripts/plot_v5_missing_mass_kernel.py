#!/usr/bin/env python3
"""kappa(f) microscopic-origin test: missing continuation mass (Good-Turing).

Hypothesis H-KAPPA: per-context gap kernel kappa(f) is proportional to the
leave-one-out novel-continuation rate M(f) of contexts with train count f:
    M(f) = N1(f) / (f * n(f))
where N1(f) = number of (context, next-token) types occurring exactly once in
train whose context has total count f. For the bigram branch the continuation
types are exactly the trigram types in data/freq_index.npz (prefix = key//8192),
so M(f) is computable exactly with zero new training.

Checks performed (bigram branch, single free amplitude C per comparison):
  1. shape of measured per-f gap g(f) (s1v5_128_frequency_main exact_freq_loss,
     step 1000, both tables, R=2^20) vs  C*M(f)  and  C*M(f)*s(f,R0),
     s = f/(f+T/R0) the collision-dilution share at R0=2^20.
  2. R-axis reconvolution with kappa=M(f) (NO fitted shape exponent at all):
     G(R) ~ sum_f n(f)*(f/T)*M(f)*f/(f+T/R); window slope vs measured 0.576.
  3. beta(R) prediction: local log-slope of M(f)*f/(f+T/R) over f in [4,4096]
     vs the measured beta(R) from single-table runs (beta_by_table_R.csv).

Outputs: docs/figs/theory/fig_v5_missing_mass_kernel.png
         docs/figs/theory/theory_missing_mass_bigram.csv
"""
import csv
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.normpath(os.path.join(ROOT, "..", "data", "freq_index.npz"))
FREQ_RUN = os.path.normpath(os.path.join(
    ROOT, "..", "data", "runs_scaling", "s1v5_128_frequency_main_fixed", "exact_freq_loss.jsonl"))
BETA_CSV = os.path.join(ROOT, "appendices", "s1_scaling_three_axis", "beta_by_table_R.csv")
OUTDIR = os.path.join(ROOT, "figs", "theory")
R0 = 2 ** 20


def loglog_slope(x, y):
    a, b = np.polyfit(np.log(x), np.log(y), 1)
    return a


def main():
    z = np.load(NPZ)
    bi_keys = z["bigram_keys"].astype(np.int64)
    bi_counts = z["bigram_counts"].astype(np.int64)
    tri_keys = z["trigram_keys"].astype(np.int64)
    tri_counts = z["trigram_counts"].astype(np.int64)
    T = int(bi_counts.sum())

    # --- M(f): group trigram continuation types by their bigram-prefix count f
    prefix = tri_keys // 8192
    pos = np.searchsorted(bi_keys, prefix)
    assert np.all(bi_keys[pos] == prefix), "prefix lookup failed"
    f_of_type = bi_counts[pos]                      # context count f for each continuation type
    ones = tri_counts == 1
    fmax = int(bi_counts.max())
    N1 = np.bincount(f_of_type[ones], minlength=fmax + 1).astype(np.float64)
    n_f = np.bincount(bi_counts, minlength=fmax + 1).astype(np.float64)
    fgrid = np.arange(fmax + 1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.where(fgrid * n_f > 0, N1 / (fgrid * n_f), np.nan)
    # sanity prints
    for f in (1, 2, 4, 8, 32, 128, 1024, 8192):
        if f <= fmax and n_f[f] > 0:
            print(f"f={f:6d}  n(f)={int(n_f[f]):9d}  M(f)={M[f]:.4f}")

    # --- measured per-f gap from the frequency run (last record)
    last = None
    for line in open(FREQ_RUN):
        last = line
    rec = json.loads(last)
    tr = rec["train"]["bigram"]
    va = rec["val"]["bigram"]
    pts = []
    for k, v in va.items():
        f = int(k)
        if f < 1 or k not in tr:
            continue
        if v["token_count"] < 200 or tr[k]["token_count"] < 200:
            continue
        pts.append((f, v["mean_loss"] - tr[k]["mean_loss"],
                    v["token_count"]))
    pts.sort()
    # geometric binning of both data and model with val-token weights
    edges = np.unique(np.round(np.logspace(0, np.log10(3e4), 25)).astype(int))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [(f, g, w) for f, g, w in pts if lo <= f < hi]
        if not sel:
            continue
        w = np.array([s[2] for s in sel], dtype=float)
        g = np.array([s[1] for s in sel])
        fs = np.array([s[0] for s in sel], dtype=float)
        fbar = float(np.exp((w * np.log(fs)).sum() / w.sum()))
        gbar = float((w * g).sum() / w.sum())
        # model values averaged over the same f with token weights n(f)*f? use exact f of data pts
        Mbar = float((w * M[fs.astype(int)]).sum() / w.sum())
        rows.append((fbar, gbar, Mbar, float(w.sum())))
    fb = np.array([r[0] for r in rows])
    gb = np.array([r[1] for r in rows])
    Mb = np.array([r[2] for r in rows])

    # --- distributional collision share s_dist(f,R): co-occupants ~ Poisson(K/R)
    # types drawn uniformly from the empirical count distribution (heavy-tailed),
    # so E[f/(f+S_other)] >> f/(f+T/R) (Jensen). Monte Carlo on the histogram.
    K = int(bi_counts.size)
    rng = np.random.default_rng(0)
    probs = n_f / K
    support = np.arange(n_f.size)
    NMC = 40000

    def sample_S(R, budget=4e7):
        lam = K / R
        nmc = int(min(NMC, max(400, budget / max(lam, 1))))
        n_co = rng.poisson(lam, nmc)
        cap = int(min(n_co.max(), lam + 6 * np.sqrt(lam) + 20))
        draws = rng.choice(support, size=(nmc, max(cap, 1)), p=probs)
        m = np.arange(max(cap, 1))[None, :] < np.minimum(n_co, cap)[:, None]
        return (draws * m).sum(axis=1)

    def s_dist(fvals, S):
        return (fvals[:, None] / (fvals[:, None] + S[None, :])).mean(axis=1)

    S0 = sample_S(R0)
    s0 = s_dist(fb, S0)
    ok = gb > 0
    # single-amplitude fits in log space
    C1 = float(np.exp(np.mean(np.log(gb[ok]) - np.log(Mb[ok]))))
    C2 = float(np.exp(np.mean(np.log(gb[ok]) - np.log(Mb[ok] * s0[ok]))))
    def r2(pred):
        ly, lp = np.log(gb[ok]), np.log(pred[ok])
        return 1 - ((ly - lp) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    win = (fb >= 4) & (fb <= 4096) & ok
    print(f"\nbigram branch, {ok.sum()} bins, f in [{fb.min():.0f},{fb.max():.0f}]")
    print(f"  model M(f):          C={C1:.3f}  logR2={r2(C1*Mb):.4f}")
    print(f"  model M(f)*s_dist:   C={C2:.3f}  logR2={r2(C2*Mb*s0):.4f}")
    print(f"  s_dist(f=1,R0)={s0[0]:.3f}  (mean-field would be {fb[0]/(fb[0]+T/R0):.3f})")
    print(f"  window f in [4,4096] slopes: data {loglog_slope(fb[win], gb[win]):.3f}, "
          f"M {loglog_slope(fb[win], Mb[win]):.3f}, "
          f"M*s_dist {loglog_slope(fb[win], (Mb*s0)[win]):.3f}")
    print(f"  full-range slopes: data {loglog_slope(fb[ok], gb[ok]):.3f}, "
          f"M {loglog_slope(fb[ok], Mb[ok]):.3f}")

    # --- two-component kernel: kappa(f) = B*M(f) + V*S_eff(f)/f
    # S_eff(f) = mean distinct continuations per context of count f (from index);
    # second term is the plug-in / estimation-variance component ~ (S-1)/f.
    Ntypes = np.bincount(f_of_type, minlength=fmax + 1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        S_eff = np.where(n_f > 0, Ntypes / n_f, np.nan)
    Sb = np.array([S_eff[int(round(f))] if int(round(f)) <= fmax and n_f[int(round(f))] > 0
                   else np.nan for f in fb])
    # fill missing S_eff at binned fbar via nearest populated f
    for i, f in enumerate(fb):
        if not np.isfinite(Sb[i]):
            cand = np.where(n_f > 0)[0]
            Sb[i] = S_eff[cand[np.argmin(np.abs(cand - f))]]
    X = np.stack([Mb[ok], np.minimum(Sb[ok], fb[ok]) / fb[ok]], axis=1)
    coef, *_ = np.linalg.lstsq(X, gb[ok], rcond=None)
    pred2 = X @ coef
    ss = 1 - ((gb[ok] - pred2) ** 2).sum() / ((gb[ok] - gb[ok].mean()) ** 2).sum()
    print(f"  two-component fit: B={coef[0]:.3f} V={coef[1]:.3f}  linR2={ss:.4f} "
          f"(one-component M-only linR2="
          f"{1 - ((gb[ok] - C1*Mb[ok])**2).sum() / ((gb[ok]-gb[ok].mean())**2).sum():.4f})")

    # --- R-axis reconvolution with kappa = M(f), distributional share (zero shape params)
    mask = (fgrid >= 1) & (n_f > 0) & np.isfinite(M)
    fv, nv, Mv = fgrid[mask], n_f[mask], M[mask]
    massf = fv * nv / T
    Rgrid = np.logspace(3, 7, 17)
    G = []
    for R in Rgrid:
        S = sample_S(R)
        # subsample f support for speed: exact for f<=1024, log-spaced above
        sh = s_dist(fv, S[:8000])
        G.append((massf * Mv * sh).sum())
    G = np.array(G)
    win = (Rgrid >= 2e3) & (Rgrid <= 2e5)
    print(f"  R-axis window slope with kappa=M(f), s_dist: "
          f"{loglog_slope(Rgrid[win], G[win]):.3f}  (measured net-gap 0.576)")

    # --- beta(R) prediction vs measured single-table betas
    beta_meas = []
    if os.path.exists(BETA_CSV):
        for r in csv.DictReader(open(BETA_CSV)):
            if r["branch"] == "bigram" and r["identifiable"] == "True":
                beta_meas.append((float(r["R"]), float(r["beta"])))
    fwin = (fv >= 4) & (fv <= 4096)
    beta_pred = []
    for R in np.logspace(3.2, 6.5, 9):
        S = sample_S(R)
        kern = Mv * s_dist(fv, S[:6000])
        beta_pred.append((R, -loglog_slope(fv[fwin], kern[fwin])))

    # --- figure
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    ax = axes[0]
    ax.loglog(fb, gb, "o", ms=5, color="#1f77b4", label="measured per-f gap (binned, step 1000)")
    ax.loglog(fb, C1 * Mb, "-", color="#2ca02c", lw=1.5, label=f"C·M(f), C={C1:.2f}")
    ax.loglog(fb, C2 * Mb * s0, "--", color="#d62728", lw=1.5,
              label=f"C·M(f)·s_dist(f,R0), C={C2:.2f} [falsified at low f]")
    two = coef[0] * Mb + coef[1] * np.minimum(Sb, fb) / fb
    ax.loglog(fb, two, "-", color="#9467bd", lw=1.8,
              label=f"B·M(f)+V·S_eff/f, B={coef[0]:.2f} V={coef[1]:.2f} (linR2={ss:.3f})")
    ax.set_xlabel("exact train context count f (bigram branch)")
    ax.set_ylabel("gap (val - train probe)")
    ax.set_title("kernel shape: missing continuation mass")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    ax.loglog(Rgrid, G / G[win][-1], "-", color="#d62728", lw=1.6,
              label="G(R) with kappa=M(f) (normalized)")
    ax.axvspan(2e3, 2e5, color="orange", alpha=0.15, label="measured fit window (slope .576)")
    ax.set_xlabel("table rows R"); ax.set_title("R-axis reconvolution, zero shape params")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    ax = axes[2]
    if beta_meas:
        bx, by = zip(*sorted(beta_meas))
        ax.semilogx(bx, by, "o", color="#1f77b4", label="measured beta(R), single-table runs")
    px, py = zip(*beta_pred)
    ax.semilogx(px, py, "-", color="#d62728", lw=1.5, label="model local slope of M(f)·s(f,R)")
    ax.set_xlabel("table rows R"); ax.set_ylabel("beta (positive)")
    ax.set_title("beta drifts with R: model vs data")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.suptitle("H-KAPPA: kappa(f) = missing continuation mass (Good-Turing), bigram branch", fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "fig_v5_missing_mass_kernel.png")
    fig.savefig(out, dpi=160)
    print("saved", out)

    with open(os.path.join(OUTDIR, "theory_missing_mass_bigram.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fbar", "gap_meas", "M", "s_R0", "val_tokens"])
        for (fbar, gbar, Mbar, wt), sv in zip(rows, s0):
            w.writerow([round(fbar, 2), round(gbar, 5), round(Mbar, 6), round(sv, 5), int(wt)])


if __name__ == "__main__":
    main()
