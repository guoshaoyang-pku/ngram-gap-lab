#!/usr/bin/env python3
"""S1 three-axis scaling · relationship figure set (three-seed, online gap).

Follows the ngram-gap-plotting skill palette (warm paper, muted semantic
colors) and the P4 online-gap convention.  Produces a coherent set of
relationship figures that show *how* the table-induced gap depends on:

  1. table size (logical addresses 2R), per module and seed        -> gap vs 2R
  2. collision / occupancy (the *physical* table state)            -> gap vs (1-coll)
  3. exposure (epoch length / replay count), per alignment         -> ΔG vs L, both alignments
  4. exact token frequency f, per branch and seed (two-factor)     -> gap(f) with fit
  5. a one-page "relationship map" tying all four views together   -> summary panel

All gaps are ONLINE (val_loss - train_loss from train_log.jsonl or summary).
Non-positive gaps are excluded from every log-scale view; f=0 (novel) is never
plotted.  Fit lines are guides (see analyze_scaling_*), not scaling-law claims.
"""
import json
import os
import glob
import math
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.environ.get("NGLAB_SCALING_RUNS_DIR",
                          os.path.join(REPO_ROOT, "data", "runs_scaling"))
FIGS_DIR = os.path.join(REPO_ROOT, "docs", "appendices",
                        "s1_scaling_three_axis", "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

# ---- ngram-gap-plotting skill palette ----
PAPER = "#f7f5ef"
BORDER = "#c8c1b6"
MUTED = "#686d73"
ANCHOR = "#353d79"
BIGRAM_C = "#2d6f9f"
TRIGRAM_C = "#c4493d"
BOTH_C = "#353d79"
NOGRAM_C = "#8a8f8a"
MOD_COLORS = {"bigram": BIGRAM_C, "trigram": TRIGRAM_C,
              "both": BOTH_C, "nogram": NOGRAM_C}
SEED_MARKERS = {42: "o", 43: "s", 44: "^"}
SEED_COLORS = {42: "#2d6f9f", 43: "#c4493d", 44: "#b67524"}
EPB = {"L1": 42, "L2": 84, "L3": 168, "L4": 337}
SEEDS = (42, 43, 44)
MIN_TOKENS = 1024
MIN_CONTEXTS = 32


def load_online_final(run_dir):
    path = os.path.join(run_dir, "train_log.jsonl")
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    if not last:
        return None
    return {"gap": float(last["gap"]), "train": float(last["train_loss"]),
            "val": float(last["val_loss"])}


def load_summary(run_dir):
    with open(os.path.join(run_dir, "summary.json")) as f:
        return json.load(f)


def load_exact(run_dir):
    path = os.path.join(run_dir, "exact_freq_loss.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def is_scaling_summary(summary, physical_id):
    config = summary.get("config", {})
    dense = (config.get("val_interval_steps") == 10 and not config.get("val_steps"))
    sparse = bool(config.get("val_steps"))
    return (
        summary.get("run_id") == physical_id
        and config.get("table_optimizer") == "rmsprop"
        and config.get("table_lr_scale") == 2.0
        and config.get("table_betas") == [0.0, 0.99]
        and (dense or sparse)
        and summary.get("compute_dtype") == "bf16"
        and summary.get("torch_compile") is False
    )


def canonical_runs():
    runs = []
    for run_dir in sorted(glob.glob(os.path.join(RUNS_DIR, "*"))):
        if not os.path.isdir(run_dir):
            continue
        pid = os.path.basename(run_dir)
        if not pid.endswith("_fixed"):
            continue
        rid = pid[:-len("_fixed")]
        if not (rid.startswith("tbl_") or rid.startswith("ep_") or rid.startswith("freq_")):
            continue
        try:
            summary = load_summary(run_dir)
        except Exception:
            continue
        if not is_scaling_summary(summary, pid):
            continue
        runs.append((rid, pid, run_dir, summary))
    return runs


def parse_seed(rid):
    for s in (43, 44):
        if rid.endswith(f"_s{s}"):
            return rid.rsplit("_s", 1)[0], s
    return rid, 42


def collect_table():
    """{ (module, mult, seed): {gap, occ, logical} }"""
    out = {}
    for rid, pid, run_dir, summary in canonical_runs():
        base, seed = parse_seed(rid)
        if not base.startswith("tbl_"):
            continue
        parts = base.split("_")
        if len(parts) != 3:
            continue
        _, mult_s, mod = parts
        if mod not in MOD_COLORS:
            continue
        try:
            mult = int(mult_s)
        except ValueError:
            continue
        if int(summary.get("seed", seed)) != seed:
            continue
        online = load_online_final(run_dir)
        gap = None
        if online:
            gap = online["gap"]
        if gap is None:
            gap = summary.get("final_gap")
        if gap is None:
            continue
        occ = None
        opath = os.path.join(run_dir, "table_occupancy.json")
        if os.path.exists(opath):
            with open(opath) as f:
                occ = json.load(f)
        logical = 16384 * mult
        out[(mod, mult, seed)] = {
            "gap": float(gap), "logical": logical,
            "mult": mult, "occ": occ,
        }
    return out


def collect_epoch():
    """{ (L, mod, align, seed): final online gap }"""
    out = {}
    for rid, pid, run_dir, summary in canonical_runs():
        base, seed = parse_seed(rid)
        if not base.startswith("ep_"):
            continue
        parts = base.split("_")
        if len(parts) != 4:
            continue
        _, L, mod, align = parts
        if L not in EPB or mod not in MOD_COLORS or align not in ("fs", "fe"):
            continue
        if int(summary.get("seed", seed)) != seed:
            continue
        online = load_online_final(run_dir)
        if online is None:
            continue
        out[(L, mod, align, seed)] = float(online["gap"])
    return out


def _sem(stats):
    n = stats["token_count"]
    if n <= 0:
        return float("nan")
    mean = stats["mean_loss"]
    var = max(stats["loss_sq_sum"] / n - mean * mean, 0.0)
    return math.sqrt(var) / math.sqrt(n)


def marginal_gap_rows(entry, branch):
    tr = entry.get("train", {}).get(branch, {})
    va = entry.get("val", {}).get(branch, {})
    rows = []
    for f in set(tr) & set(va):
        t, v = tr[f], va[f]
        if t["token_count"] < MIN_TOKENS or v["token_count"] < MIN_TOKENS:
            continue
        if t["distinct_contexts"] < MIN_CONTEXTS or v["distinct_contexts"] < MIN_CONTEXTS:
            continue
        rows.append({
            "f": int(f),
            "gap": v["mean_loss"] - t["mean_loss"],
            "sem": math.hypot(_sem(t), _sem(v)),
            "n": t["token_count"] + v["token_count"],
        })
    return rows


def freq_step_entry(run_dir, step):
    """Return the exact-frequency record nearest to (and not after) step."""
    best = None
    for entry in load_exact(run_dir):
        if entry.get("step", 0) <= step:
            best = entry
        else:
            break
    return best


def collect_freq():
    """{(branch, mod, align, seed): [exact rows at final step]}"""
    out = {}
    for rid, pid, run_dir, summary in canonical_runs():
        base, seed = parse_seed(rid)
        if not base.startswith("freq_"):
            continue
        parts = base.split("_")
        if len(parts) != 3:
            continue
        _, mod, align = parts
        if mod not in MOD_COLORS or align not in ("fs", "fe"):
            continue
        if int(summary.get("seed", seed)) != seed:
            continue
        exact = load_exact(run_dir)
        if not exact:
            continue
        for branch in ("bigram", "trigram"):
            rows = marginal_gap_rows(exact[-1], branch)
            if rows:
                out[(branch, mod, align, seed)] = rows
    return out


def collect_freq_snapshots():
    """{(branch, mod, seed, snapshot): [exact rows]} for the fe axis."""
    out = {}
    targets = (("E1", 337), ("E3", 1012), ("E6", 2022))
    for rid, pid, run_dir, summary in canonical_runs():
        base, seed = parse_seed(rid)
        if not base.startswith("freq_"):
            continue
        parts = base.split("_")
        if len(parts) != 3:
            continue
        _, mod, align = parts
        if mod not in MOD_COLORS or align != "fe":
            continue
        if int(summary.get("seed", seed)) != seed:
            continue
        for snapshot, target_step in targets:
            exact = freq_step_entry(run_dir, target_step)
            if exact is None:
                continue
            for branch in ("bigram", "trigram"):
                rows = marginal_gap_rows(exact, branch)
                if rows:
                    out[(branch, mod, seed, snapshot)] = rows
    return out


def style_ax(ax):
    ax.set_facecolor(PAPER)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(MUTED)
    ax.grid(alpha=0.3, which="both", color=BORDER)


def log_bin(xs, ys, n=8):
    """Equal-count log-f bins -> geometric midpoint + mean gap."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    order = np.argsort(np.log(xs))
    xs, ys = xs[order], ys[order]
    edges = np.array_split(np.arange(len(xs)), min(n, len(xs)))
    bx, by = [], []
    for idx in edges:
        bx.append(np.exp(np.mean(np.log(xs[idx]))))
        by.append(np.mean(ys[idx]))
    return np.array(bx), np.array(by)


def two_factor(f, A, beta, c, gamma):
    return A * f ** (-beta) * (1.0 - np.exp(-c * f ** gamma))


def fit_two_factor(fs, gaps):
    from scipy.optimize import curve_fit
    fs = np.asarray(fs, float)
    gaps = np.asarray(gaps, float)
    keep = gaps > 0
    if keep.sum() < 4:
        return None
    fs, gaps = fs[keep], gaps[keep]
    p0 = [np.median(gaps), 0.5, 0.1, 0.5]
    try:
        popt, pcov = curve_fit(two_factor, fs, gaps, p0=p0, maxfev=30000,
                               bounds=([0, -5, 0, 0], [np.inf, 5, np.inf, 5]))
        return popt
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Figure 1: gap vs logical addresses (per module, per seed, log-log)
# ---------------------------------------------------------------------------
def fig_gap_vs_2R(table):
    fig, ax = plt.subplots(figsize=(9.5, 6))
    fig.patch.set_facecolor(PAPER)
    for mod, color in MOD_COLORS.items():
        if mod == "nogram":
            continue
        for seed in SEEDS:
            pts = [(v["logical"], v["gap"]) for (m, mm, s), v in table.items()
                   if m == mod and s == seed]
            if not pts:
                continue
            pts.sort()
            xs = np.array([p[0] for p in pts], float)
            ys = np.array([p[1] for p in pts], float)
            pos = ys > 0
            if pos.sum() < 2:
                continue
            ax.plot(xs[pos], ys[pos], marker=SEED_MARKERS[seed], markersize=5.5,
                    linewidth=1.4, alpha=0.9 if seed == 42 else 0.55,
                    color=color, label=f"{mod} s{seed}")
            if pos.sum() >= 3:
                k, b = np.polyfit(np.log(xs[pos]), np.log(ys[pos]), 1)
                xf = np.linspace(xs[pos].min(), xs[pos].max(), 80)
                ax.plot(xf, np.exp(b) * xf ** k, "--", linewidth=1.0,
                        alpha=0.45, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("logical addresses 2R (per n-gram per layer; log)")
    ax.set_ylabel("final online gap = val − train (log)")
    ax.set_title("Table size → gap  (dashed = log-log guide, not a claim)")
    ax.legend(fontsize=8, ncol=3)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "rel_gap_vs_2R_multiseed.png"), dpi=160)
    plt.close(fig)
    print("saved rel_gap_vs_2R_multiseed.png")


# ---------------------------------------------------------------------------
# Figure 2: gap vs physical table state (1 - collision, occupancy)
# ---------------------------------------------------------------------------
def _occ_state(occ, branch="bigram"):
    if not occ:
        return None, None
    b = occ["branches"][branch]["0"][0]
    return 1.0 - b["collision_rate"], b["occupancy"]


def fig_gap_vs_physical(table):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.patch.set_facecolor(PAPER)
    for mod, color in MOD_COLORS.items():
        if mod not in ("bigram", "trigram"):
            continue
        for seed in SEEDS:
            pts = []
            for (m, mm, s), v in table.items():
                if m != mod or s != seed:
                    continue
                branch = mod if mod in ("bigram", "trigram") else "bigram"
                c, _ = _occ_state(v["occ"], branch)
                if c is not None:
                    pts.append((c, v["gap"]))
            if not pts:
                continue
            pts.sort()
            xs = np.array([p[0] for p in pts], float)
            ys = np.array([p[1] for p in pts], float)
            pos = ys > 0
            if pos.sum() < 2:
                continue
            ax.plot(xs[pos], ys[pos], marker=SEED_MARKERS[seed], markersize=5,
                    linewidth=1.3, alpha=0.9 if seed == 42 else 0.5,
                    color=color, label=f"{mod} s{seed}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("1 − collision rate (module-matched L0 h0; log)")
    ax.set_ylabel("final online gap (log)")
    ax.legend(fontsize=7, ncol=3)
    style_ax(ax)
    ax.set_title("Table physical state → gap  (collision complement)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "rel_gap_vs_physical.png"), dpi=160)
    plt.close(fig)
    print("saved rel_gap_vs_physical.png")


# ---------------------------------------------------------------------------
# Figure 3: ΔG vs epoch length (both alignments, per module, 3 seeds)
# ---------------------------------------------------------------------------
def fig_deltaG_vs_epoch(epoch):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True)
    fig.patch.set_facecolor(PAPER)
    Ls = ["L1", "L2", "L3", "L4"]
    for align, ax in (("fs", axes[0]), ("fe", axes[1])):
        for mod, color in MOD_COLORS.items():
            if mod == "nogram":
                continue
            for seed in SEEDS:
                dgs = []
                for L in Ls:
                    gm = epoch.get((L, mod, align, seed))
                    gn = epoch.get((L, "nogram", align, seed))
                    if gm is None or gn is None:
                        dgs.append(None)
                    else:
                        dgs.append(gm - gn)
                xs = np.arange(len(Ls))
                valid = [(x, d) for x, d in zip(xs, dgs) if d is not None]
                if not valid:
                    continue
                ax.plot([x for x, _ in valid], [d for _, d in valid],
                        marker=SEED_MARKERS[seed], markersize=5.5, linewidth=1.3,
                        alpha=0.9 if seed == 42 else 0.55, color=color,
                        label=f"{mod} s{seed}")
        ax.axhline(0, color=BORDER, lw=1.0)
        ax.set_xticks(np.arange(len(Ls)))
        ax.set_xticklabels(Ls)
        ax.set_xlabel("epoch length L")
        ax.set_title(f"{'fixed-step (1000 steps)' if align == 'fs' else 'fixed-epoch (6 epochs)'}")
        ax.legend(fontsize=7, ncol=3)
        style_ax(ax)
    axes[0].set_ylabel("final ΔG = G(module) − G(no-ngram)")
    fig.suptitle("Epoch length → table-induced gap  (3 seeds; all ΔG > 0 → direction seed-stable)",
                 color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIGS_DIR, "rel_deltaG_vs_epoch.png"), dpi=160)
    plt.close(fig)
    print("saved rel_deltaG_vs_epoch.png")


# ---------------------------------------------------------------------------
# Figure 4: gap(f) two-factor, per branch, with fit + error bars
# ---------------------------------------------------------------------------
def fig_gap_vs_frequency(freq):
    for branch, bcolor in (("bigram", BIGRAM_C), ("trigram", TRIGRAM_C)):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.patch.set_facecolor(PAPER)
        for ax, align, title in (
            (axes[0], "fs", "fixed-step (1000 steps)"),
            (axes[1], "fe", "fixed-epoch (6 epochs, 2022 steps)"),
        ):
            for mod, color in MOD_COLORS.items():
                if mod == "nogram":
                    continue
                for seed in SEEDS:
                    key = (branch, mod, align, seed)
                    if key not in freq:
                        continue
                    rows = freq[key]
                    fs = np.array([r["f"] for r in rows])
                    gs = np.array([r["gap"] for r in rows])
                    pos = gs > 0
                    if pos.sum() < 4:
                        continue
                    bx, by = log_bin(fs[pos], gs[pos])
                    # The binning error bar is a diagnostic of within-bin
                    # variation, not uncertainty of a population estimate.
                    bin_x, bin_y = [], []
                    bin_yerr = []
                    order = np.argsort(np.log(fs[pos]))
                    fsp, gsp = fs[pos][order], gs[pos][order]
                    for idx in np.array_split(np.arange(len(fsp)),
                                              min(8, len(fsp))):
                        bin_x.append(np.exp(np.mean(np.log(fsp[idx]))))
                        bin_y.append(np.mean(gsp[idx]))
                        bin_yerr.append(np.std(gsp[idx], ddof=1) /
                                        math.sqrt(len(idx)) if len(idx) > 1 else 0.0)
                    ax.errorbar(bin_x, bin_y, yerr=bin_yerr,
                                fmt=SEED_MARKERS[seed], markersize=4,
                                color=MOD_COLORS[mod], alpha=0.8,
                                label=f"{mod} s{seed}", capsize=2, lw=0.8)
                    fit = fit_two_factor(fs, gs)
                    if fit is not None:
                        xf = np.geomspace(fs[pos].min(), fs[pos].max(), 80)
                        ax.plot(xf, two_factor(xf, *fit), "--", linewidth=1.0,
                                alpha=0.35, color=MOD_COLORS[mod])
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(f"exact {branch} frequency f (log)")
            ax.set_ylabel(f"token-marginal gap(f) (log)")
            ax.set_title(title)
            ax.legend(fontsize=6.5, ncol=2)
            style_ax(ax)
        fig.suptitle(f"{branch} branch · exact frequency → gap  "
                     "(8 log-f bins, geometric midpoints; dashed = two-factor guide)",
                     color=MUTED)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(os.path.join(FIGS_DIR, f"rel_gap_vs_frequency_{branch}.png"), dpi=160)
        plt.close(fig)
        print(f"saved rel_gap_vs_frequency_{branch}.png")


def fig_frequency_snapshots(freq_snapshots):
    """Show how the frequency relation changes across replay exposure."""
    targets = ("E1", "E3", "E6")
    fit_rows = []
    for branch in ("bigram", "trigram"):
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
        fig.patch.set_facecolor(PAPER)
        for ax, snapshot in zip(axes, targets):
            for mod, color in MOD_COLORS.items():
                if mod == "nogram":
                    continue
                for seed in SEEDS:
                    rows = freq_snapshots.get((branch, mod, seed, snapshot), [])
                    rows = [r for r in rows if r["f"] > 0 and r["gap"] > 0]
                    if len(rows) < 4:
                        continue
                    fs = np.array([r["f"] for r in rows])
                    gs = np.array([r["gap"] for r in rows])
                    bx, by = log_bin(fs, gs)
                    # Match the final-frequency figures: show the within-bin
                    # SEM rather than connecting noisy per-f points.
                    order = np.argsort(np.log(fs))
                    fsp, gsp = fs[order], gs[order]
                    yerr = []
                    for idx in np.array_split(np.arange(len(fsp)),
                                              min(8, len(fsp))):
                        yerr.append(np.std(gsp[idx], ddof=1) /
                                    math.sqrt(len(idx)) if len(idx) > 1 else 0.0)
                    ax.errorbar(bx, by, yerr=yerr,
                                fmt=SEED_MARKERS[seed], markersize=4,
                                linewidth=0.8, alpha=0.75, color=color,
                                capsize=1.5, label=f"{mod} s{seed}")
                    fit = fit_two_factor(fs, gs)
                    if fit is not None:
                        xf = np.geomspace(fs.min(), fs.max(), 80)
                        ax.plot(xf, two_factor(xf, *fit), "--", linewidth=0.9,
                                alpha=0.35, color=color)
                        fit_rows.append({
                            "branch": branch, "module": mod, "seed": seed,
                            "snapshot": snapshot, "target_step": {
                                "E1": 337, "E3": 1012, "E6": 2022
                            }[snapshot],
                            "A": fit[0], "beta": fit[1], "c": fit[2],
                            "gamma": fit[3],
                        })
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("exact frequency f (log)")
            ax.set_title({"E1": "epoch 1 (~337 steps)", "E3": "epoch 3 (~1012 steps)",
                          "E6": "epoch 6 (2022 steps)"}[snapshot])
            ax.legend(fontsize=6.5, ncol=2)
            style_ax(ax)
        axes[0].set_ylabel(f"{branch} token-marginal gap(f) (log)")
        fig.suptitle(f"{branch} branch · exposure × frequency → gap  "
                     "(fixed-epoch, positive gaps; dashed = two-factor guide)",
                     color=MUTED)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out = os.path.join(FIGS_DIR, f"rel_gap_vs_frequency_epoch_{branch}.png")
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"saved rel_gap_vs_frequency_epoch_{branch}.png")
    with open(os.path.join(FIGS_DIR, "frequency_snapshot_fit.csv"), "w",
              newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "branch", "module", "seed", "snapshot", "target_step",
            "A", "beta", "c", "gamma"
        ])
        writer.writeheader()
        writer.writerows(fit_rows)
    print("saved frequency_snapshot_fit.csv")


# ---------------------------------------------------------------------------
# Interactive table relationship: legend-controlled traces + axis buttons.
# ---------------------------------------------------------------------------
def fig_table_interactive(table):
    import plotly.graph_objects as go
    fig = go.Figure()
    for mod, color in MOD_COLORS.items():
        if mod == "nogram":
            continue
        for seed in SEEDS:
            pts = sorted((v["logical"], v["gap"], mm)
                         for (m, mm, s), v in table.items()
                         if m == mod and s == seed and v["gap"] > 0)
            if not pts:
                continue
            fig.add_trace(go.Scatter(
                x=[p[0] for p in pts], y=[p[1] for p in pts],
                customdata=[[p[2], mod, seed] for p in pts],
                mode="lines+markers", name=f"{mod} s{seed}",
                marker={"symbol": {42: "circle", 43: "square", 44: "triangle-up"}[seed]},
                line={"color": color},
                hovertemplate="mult=%{customdata[0]}<br>module=%{customdata[1]}<br>seed=%{customdata[2]}<br>2R=%{x}<br>gap=%{y:.4f}<extra></extra>",
            ))
    fig.update_layout(
        template="plotly_white", paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        title="Table size → final online gap (3 seeds)",
        xaxis={"title": "logical addresses 2R", "type": "log"},
        yaxis={"title": "final online gap = val − train", "type": "log"},
        legend={"orientation": "h"},
        updatemenus=[{"type": "buttons", "direction": "right", "x": 0.0, "y": 1.16,
                      "buttons": [
                          {"label": "both log", "method": "relayout", "args": [{"xaxis.type": "log", "yaxis.type": "log"}]},
                          {"label": "x log / y linear", "method": "relayout", "args": [{"xaxis.type": "log", "yaxis.type": "linear"}]},
                          {"label": "both linear", "method": "relayout", "args": [{"xaxis.type": "linear", "yaxis.type": "linear"}]},
                      ]}],
        margin={"t": 120, "b": 70, "l": 80, "r": 25},
    )
    out = os.path.join(FIGS_DIR, "rel_gap_vs_2R_multiseed.html")
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    print("saved rel_gap_vs_2R_multiseed.html")


def fig_frequency_interactive(freq):
    import plotly.graph_objects as go
    fig = go.Figure()
    for branch, bcolor in (("bigram", BIGRAM_C), ("trigram", TRIGRAM_C)):
        for mod, color in MOD_COLORS.items():
            if mod == "nogram":
                continue
            for seed in SEEDS:
                rows = freq.get((branch, mod, "fe", seed), [])
                rows = [r for r in rows if r["f"] > 0 and r["gap"] > 0]
                if not rows:
                    continue
                bx, by = log_bin([r["f"] for r in rows], [r["gap"] for r in rows])
                fig.add_trace(go.Scatter(
                    x=bx, y=by, mode="lines+markers", name=f"{branch}/{mod} s{seed}",
                    marker={"symbol": {42: "circle", 43: "square", 44: "triangle-up"}[seed]},
                    line={"color": color},
                    hovertemplate="f=%{x:.3g}<br>gap=%{y:.4f}<extra></extra>",
                    legendgroup=branch,
                ))
    fig.update_layout(
        template="plotly_white", paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        title="Exact frequency → token-marginal gap (fe; positive gaps only)",
        xaxis={"title": "exact frequency f", "type": "log"},
        yaxis={"title": "gap(f)", "type": "log"}, legend={"orientation": "h"},
        updatemenus=[{"type": "buttons", "direction": "right", "x": 0.0, "y": 1.16,
                      "buttons": [
                          {"label": "both log", "method": "relayout", "args": [{"xaxis.type": "log", "yaxis.type": "log"}]},
                          {"label": "x log / y linear", "method": "relayout", "args": [{"xaxis.type": "log", "yaxis.type": "linear"}]},
                          {"label": "both linear", "method": "relayout", "args": [{"xaxis.type": "linear", "yaxis.type": "linear"}]},
                      ]}],
        margin={"t": 120, "b": 70, "l": 80, "r": 25},
    )
    out = os.path.join(FIGS_DIR, "rel_gap_vs_frequency_multiseed.html")
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    print("saved rel_gap_vs_frequency_multiseed.html")


# ---------------------------------------------------------------------------
# Figure 5: relationship map — one panel per relationship, annotated
# ---------------------------------------------------------------------------
def fig_relationship_map(table, epoch, freq):
    fig = plt.figure(figsize=(15, 10.5))
    fig.patch.set_facecolor(PAPER)
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.3)

    # (a) table size -> gap
    ax = fig.add_subplot(gs[0, 0])
    for mod, color in MOD_COLORS.items():
        if mod == "nogram":
            continue
        for seed in SEEDS:
            pts = [(v["logical"], v["gap"]) for (m, mm, s), v in table.items()
                   if m == mod and s == seed and v["gap"] > 0]
            if not pts:
                continue
            pts.sort()
            xs = np.array([p[0] for p in pts], float)
            ys = np.array([p[1] for p in pts], float)
            ax.plot(xs, ys, marker=SEED_MARKERS[seed], markersize=4,
                    linewidth=1.2, alpha=0.75, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("(a) table size → gap\n(bigger table ⇒ bigger gap; no plateau resolved ≤1M)")
    ax.set_xlabel("logical addresses 2R (log)")
    ax.set_ylabel("final online gap (log)")
    style_ax(ax)

    # (b) collision -> gap (trigram only, cleaner)
    ax = fig.add_subplot(gs[0, 1])
    for seed in SEEDS:
        pts = []
        for (m, mm, s), v in table.items():
            if m != "trigram" or s != seed or v["gap"] <= 0:
                continue
            c, _ = _occ_state(v["occ"], "trigram")
            if c is not None:
                pts.append((c, v["gap"]))
        if not pts:
            continue
        pts.sort()
        xs = np.array([p[0] for p in pts], float)
        ys = np.array([p[1] for p in pts], float)
        ax.plot(xs, ys, marker=SEED_MARKERS[seed], markersize=4,
                linewidth=1.2, alpha=0.75, color=SEED_COLORS[seed],
                label=f"trigram s{seed}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("(b) collision complement → gap (trigram)\n"
                 "larger 1−collision ⇒ larger gap")
    ax.set_xlabel("1 − collision rate (log)")
    ax.set_ylabel("final online gap (log)")
    ax.legend(fontsize=7)
    style_ax(ax)

    # (c) epoch length -> ΔG (fixed-epoch; the stable alignment)
    ax = fig.add_subplot(gs[1, 0])
    Ls = ["L1", "L2", "L3", "L4"]
    for mod, color in MOD_COLORS.items():
        if mod == "nogram":
            continue
        for seed in SEEDS:
            dgs = []
            for L in Ls:
                gm = epoch.get((L, mod, "fe", seed))
                gn = epoch.get((L, "nogram", "fe", seed))
                dgs.append((gm - gn) if (gm is not None and gn is not None) else None)
            valid = [(i, d) for i, d in enumerate(dgs) if d is not None]
            if valid:
                ax.plot([i for i, _ in valid], [d for _, d in valid],
                        marker=SEED_MARKERS[seed], markersize=4,
                        linewidth=1.2, alpha=0.75, color=color)
    ax.axhline(0, color=BORDER, lw=0.9)
    ax.set_xticks(range(len(Ls)))
    ax.set_xticklabels(Ls)
    ax.set_title("(c) exposure → gap\n(fixed-epoch ΔG; L4 trigram ≈ 5.7, cv≈2%)")
    ax.set_xlabel("epoch length L")
    ax.set_ylabel("ΔG (module − no-ngram)")
    style_ax(ax)

    # (d) frequency -> gap (two-factor, trigram branch, fe align, 3 seeds)
    ax = fig.add_subplot(gs[1, 1])
    for mod, color in MOD_COLORS.items():
        if mod == "nogram":
            continue
        for seed in SEEDS:
            key = ("trigram", mod, "fe", seed)
            if key not in freq:
                continue
            rows = freq[key]
            fs = np.array([r["f"] for r in rows])
            gs = np.array([r["gap"] for r in rows])
            pos = gs > 0
            if pos.sum() < 4:
                continue
            bx, by = log_bin(fs[pos], gs[pos], 6)
            ax.plot(bx, by, marker=SEED_MARKERS[seed], markersize=4,
                    linewidth=1.2, alpha=0.75, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("(d) frequency → gap\n(trigram branch, fe; β stable across seeds)")
    ax.set_xlabel("exact trigram frequency f (log)")
    ax.set_ylabel("gap(f) (log)")
    style_ax(ax)

    fig.suptitle("S1 three-axis scaling · relationship map (3 seeds, online gap)",
                 color=MUTED, fontsize=13)
    fig.savefig(os.path.join(FIGS_DIR, "rel_relationship_map.png"), dpi=160)
    plt.close(fig)
    print("saved rel_relationship_map.png")


def main():
    table = collect_table()
    epoch = collect_epoch()
    freq = collect_freq()
    freq_snapshots = collect_freq_snapshots()
    print(f"table points: {len(table)}, epoch points: {len(epoch)}, "
          f"freq groups: {len(freq)}, freq snapshots: {len(freq_snapshots)}")
    fig_gap_vs_2R(table)
    fig_gap_vs_physical(table)
    fig_deltaG_vs_epoch(epoch)
    fig_gap_vs_frequency(freq)
    fig_relationship_map(table, epoch, freq)
    fig_table_interactive(table)
    fig_frequency_interactive(freq)
    fig_frequency_snapshots(freq_snapshots)


if __name__ == "__main__":
    main()
