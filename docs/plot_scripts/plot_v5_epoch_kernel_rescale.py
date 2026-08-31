#!/usr/bin/env python3
"""§45 zero-GPU check: does the exact-f kernel g_e(f) translate along e*f?

Question (challenge ① on kappa(f) micro-origin): a context with train
hit-count f that has lived through e epochs has seen e*f total exposures.
If per-context memory state depended ONLY on total exposure, the epoch-e
snapshot of the frequency kernel would equal the epoch-1 snapshot shifted
horizontally by e: g_e(f) = g_1(e*f).  The two-state model instead predicts
a vertical amplification g_e(f) ~ A(e) * kappa(f) (backbone readout grows
with passes, kernel shape fixed).  We test both collapses against the raw
curves.

Kernel used here: g_e(f) = val_both_e(f) - val_nogram_e(f) per exact train
hit-count f (same fixed val population and grouping in both runs; positive
= the table raises val loss at that f).  Data: 10 epoch-boundary rows of
exact_freq_loss.jsonl from the §33 10-epoch L4 replay (both + nogram).

Run:  NGLAB_KERNEL_DATA=/tmp/ng_data python3 plot_v5_epoch_kernel_rescale.py
"""
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from v5_style import apply_style, BRANCH_COLORS, save

DATA_DIR = Path(os.environ.get("NGLAB_KERNEL_DATA", "/tmp/ng_data"))
OUT_DIR = Path(os.environ.get("NGLAB_FIG_DIR",
                              "/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/figs/theory"))

RUN_BOTH = "s1v5_128_ep1xL4_10ep_both_boundaries.json"
RUN_NOGRAM = "s1v5_128_ep1xL4_10ep_nogram_boundaries.json"

MIN_TOKENS = 200      # drop under-populated f bins from curves
F_MIN, F_MAX = 2, 71147
FIT_WINDOW = (8.0, 512.0)   # reference window for vertical amplification A_e
CMAP = plt.get_cmap("viridis")


def load_rows(path):
    rows = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            rows[d["epoch"]] = d
    return rows


def kernel(rows_both, rows_nogram, branch, epoch):
    """Return (f, g, tokens) arrays for one epoch: g = val_both - val_nogram."""
    b = rows_both[epoch]["val"][branch]
    n = rows_nogram[epoch]["val"][branch]
    fs, gs, ws = [], [], []
    for f, e_b in b.items():
        fi = int(f)
        if not (F_MIN <= fi <= F_MAX):
            continue
        e_n = n.get(f)
        if e_n is None:
            continue
        tok = e_b["token_count"]
        if tok < MIN_TOKENS:
            continue
        fs.append(fi)
        gs.append(e_b["mean_loss"] - e_n["mean_loss"])
        ws.append(tok)
    order = np.argsort(fs)
    return (np.array(fs, float)[order], np.array(gs, float)[order],
            np.array(ws, float)[order])


def interp_log(f_src, y_src, f_query):
    """Linear interpolation in log-log space (y_query undefined where out of range)."""
    m = y_src > 0
    x1, y1 = np.log(f_src[m]), np.log(y_src[m])
    o = np.argsort(x1)
    inside = (np.log(f_query) >= x1[o][0]) & (np.log(f_query) <= x1[o][-1])
    out = np.full_like(f_query, np.nan)
    out[inside] = np.exp(np.interp(np.log(f_query[inside]), x1[o], y1[o]))
    return out


def epoch_color(e, epochs):
    frac = (e - 1) / max(1, epochs[-1] - 1)
    return CMAP(0.15 + 0.8 * frac)


def main():
    apply_style()
    rows_both = load_rows(DATA_DIR / RUN_BOTH)
    rows_nogram = load_rows(DATA_DIR / RUN_NOGRAM)
    epochs = sorted(rows_both)
    print(f"epoch-boundary rows: {epochs}")

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    csv_rows = []
    for r, branch in enumerate(("bigram", "trigram")):
        ax_raw, ax_ef, ax_vert = axes[r]
        data = {e: kernel(rows_both, rows_nogram, branch, e) for e in epochs}

        # ---- vertical anchor: first epoch with >=3 positive in-window bins
        ref_e, y1_win = None, None
        for e in epochs:
            f, g, w = data[e]
            m = (f >= FIT_WINDOW[0]) & (f <= FIT_WINDOW[1]) & (g > 0)
            if m.sum() >= 3:
                ref_e = e
                y1_win = float(np.average(np.log(g[m]), weights=w[m]))
                break
        print(f"\n[{branch}] vertical anchor epoch = {ref_e} "
              f"(window f in {FIT_WINDOW}); A_e relative to it")

        # ---- plots
        for e in epochs:
            f, g, w = data[e]
            c = epoch_color(e, epochs)
            ax_raw.plot(f, g, lw=0.9, color=c, alpha=0.85)
            ax_ef.plot(f * e, g, lw=0.9, color=c, alpha=0.85)
            if ref_e is not None and e >= ref_e:
                m = (f >= FIT_WINDOW[0]) & (f <= FIT_WINDOW[1]) & (g > 0)
                if m.sum() >= 3:
                    A = float(np.exp(np.average(np.log(g[m]), weights=w[m]) - y1_win))
                    ax_vert.plot(f, g / A, lw=0.9, color=c, alpha=0.85)

        # ---- verdict numbers per epoch
        print(f"{'epoch':>5} {'A_vert':>8} {'RMSE e*f':>10} {'RMSE vert':>10}")
        f_ref, y_ref = data[ref_e][0], data[ref_e][1] if ref_e else (None, None)
        for e in epochs:
            f, g, w = data[e]
            A = np.nan
            if ref_e is not None and e >= ref_e:
                m = (f >= FIT_WINDOW[0]) & (f <= FIT_WINDOW[1]) & (g > 0)
                if m.sum() >= 3:
                    A = float(np.exp(np.average(np.log(g[m]), weights=w[m]) - y1_win))
            rmse_ef = rmse_v = np.nan
            if e != ref_e and f_ref is not None:
                fm, gm, wm = f, g, w
                ok = (gm > 0) & (fm * e <= f_ref.max())
                if ok.sum() >= 4:
                    y_h = interp_log(f_ref, y_ref, fm[ok] * e)
                    y_v = interp_log(f_ref, y_ref, fm[ok])
                    good = ~np.isnan(y_h) & ~np.isnan(y_v) & (y_v > 0)
                    if good.sum() >= 4:
                        A_fit = np.exp(np.average(np.log(gm[ok][good]) - np.log(y_v[good]),
                                                  weights=wm[ok][good]))
                        rmse_ef = float(np.sqrt(np.average(
                            (np.log(gm[ok][good]) - np.log(y_h[good])) ** 2,
                            weights=wm[ok][good])))
                        rmse_v = float(np.sqrt(np.average(
                            (np.log(gm[ok][good]) - np.log(A_fit * y_v[good])) ** 2,
                            weights=wm[ok][good])))
            print(f"{e:>5} {A:>8.3f} {rmse_ef:>10.3f} {rmse_v:>10.3f}")
            csv_rows.append({
                "branch": branch, "epoch": e, "A_vertical": A,
                "rmse_ef": rmse_ef, "rmse_vertical": rmse_v,
                "n_f_bins": len(f), "ref_epoch": ref_e or "",
            })

        for ax, title in ((ax_raw, "raw  $g_e(f)$"),
                          (ax_ef, r"horizontal rescale: $g_e(f)$ vs $e\cdot f$"),
                          (ax_vert, r"vertical rescale: $g_e(f)/A_e$")):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{branch} · {title}", fontsize=9.5)
            ax.set_xlabel(r"train hit-count $f$" if ax is ax_raw else
                          (r"$e\cdot f$" if ax is ax_ef else r"$f$"))
        ax_raw.set_ylabel(r"$g(f)=$val$_{\rm both}-$val$_{\rm nogram}$")
        ax_vert.set_ylabel(r"$g(f)/A_e$")
        handles = [plt.Line2D([], [], color=epoch_color(e, epochs), lw=1.2,
                              label=f"e={e}") for e in (1, 3, 5, 10)]
        ax_raw.legend(handles=handles, loc="best")
    fig.suptitle("kernel snapshot rescale test · 10-epoch L4 replay (both vs nogram, epoch boundaries)",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png, svg = save(fig, OUT_DIR, "fig_v5_epoch_kernel_rescale")

    csv_path = OUT_DIR / "theory_epoch_kernel_rescale.csv"
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=sorted({k for row in csv_rows for k in row}))
        wr.writeheader()
        wr.writerows(csv_rows)
    print(f"\nsaved: {png}\n       {svg}\n       {csv_path}")


if __name__ == "__main__":
    main()
