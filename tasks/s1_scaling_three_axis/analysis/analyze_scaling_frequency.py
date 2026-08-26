#!/usr/bin/env python3
"""ngram-gap-lab · exact-frequency scaling analysis (plan §4).

Reads data/runs_scaling/<run>/exact_freq_loss.jsonl:
  each line: {step, epoch, train: {bigram/trigram: {f: stats}},
              val: {...}, shared: {bigram/trigram: {shared_total, per_f}}}

Main quantity:
  token-marginal gap:  E[val loss | f] − E[train loss | f]

Produces (per branch, final fixed-step checkpoint of the L4 freq runs):
  - freq_gap_<branch>_final_raw.png : ALL per-f points (debug; ugly, but shows
    every point with a vertical SEM bar).  Log-log by default.
  - freq_gap_<branch>_final.png     : binned summary.  Each bin spans a
    geometric f-range [lo, hi] (equal-count bins on log-f); point x = geometric
    midpoint sqrt(lo*hi), horizontal error bar = bin f-range, vertical error
    bar = pooled SEM of the token-marginal gap.  Log-log by default.
  - fit_manifest.json               : two-factor fit
    G(f)=A f^-β [1 - exp(-c f^γ)] per module, plus which f were excluded.

Bin rules (source of truth: the plotting skill):
  - never subdivide an existing aggregate bucket; bins are built from the
    per-f entries themselves, so finer bins are always exact.
  - log views drop `novel` (f=0), f without both train & val buckets, and
    non-positive gaps (log undefined).  State this in the figure.

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
RUNS_DIR = os.environ.get(
    "NGLAB_SCALING_RUNS_DIR",
    os.environ.get("NGLAB_RUNS_DIR", os.path.join(REPO_ROOT, "data", "runs_scaling")),
)
if len(sys.argv) > 1:
    RUNS_DIR = sys.argv[1]
FIGS_DIR = os.path.join(
    REPO_ROOT, "docs", "appendices", "s1_scaling_three_axis", "figs"
)
os.makedirs(FIGS_DIR, exist_ok=True)

MIN_TOKENS = 1024
MIN_CONTEXTS = 32
N_BUCKETS = 8

# plotting skill palette
PAPER = "#f7f5ef"
BORDER = "#c8c1b6"
MUTED = "#686d73"
ANCHOR = "#353d79"
BIGRAM_C = "#2d6f9f"
TRIGRAM_C = "#c4493d"
MOD_COLORS = {"bigram": "#2d6f9f", "trigram": "#c4493d", "both": "#353d79", "nogram": "#8a8f8a"}


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


def canonical_run_dirs(runs_dir):
    for run_dir in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        if not os.path.isdir(run_dir):
            continue
        physical_id = os.path.basename(run_dir)
        if not physical_id.endswith("_fixed"):
            continue
        yield physical_id[:-len("_fixed")], physical_id, run_dir


def is_current_scaling_summary(summary, physical_id):
    config = summary.get("config", {})
    exact_interval = config.get("exact_freq_eval_interval")
    # Older frequency runs may predate the config key, but current runs must
    # align exact-frequency evaluation with the standard 10-step cadence.
    exact_ok = exact_interval is None or exact_interval in {10}
    return (
        summary.get("run_id") == physical_id
        and config.get("table_optimizer") == "rmsprop"
        and config.get("table_lr_scale") == 2.0
        and config.get("table_betas") == [0.0, 0.99]
        and config.get("val_interval_steps") == 10
        and config.get("table_norm_interval_steps") == 10
        and exact_ok
        and summary.get("compute_dtype") == "bf16"
        and summary.get("torch_compile") is False
    )


def _sem(stats):
    """Standard error of the mean loss from loss_sum/loss_sq_sum."""
    n = stats["token_count"]
    mean = stats["mean_loss"]
    var = max(stats["loss_sq_sum"] / n - mean * mean, 0.0)
    return math.sqrt(var) / math.sqrt(n)


def marginal_gap_rows(entry, branch):
    """Eligible per-f rows.

    Returns list of dicts {f, gap, train_sem, val_sem, gap_sem, train_n, val_n}
    plus a list of rejected {f, reason}.
    """
    tr = entry.get("train", {}).get(branch, {})
    va = entry.get("val", {}).get(branch, {})
    fs = set(tr) | set(va)
    out = []
    rejected = []
    for f in fs:
        t = tr.get(f)
        v = va.get(f)
        if not t or not v:
            rejected.append({"f": int(f), "reason": "missing train or val bucket"})
            continue
        reasons = []
        if t["token_count"] < MIN_TOKENS:
            reasons.append(f"train token_count < {MIN_TOKENS}")
        if v["token_count"] < MIN_TOKENS:
            reasons.append(f"val token_count < {MIN_TOKENS}")
        if t["distinct_contexts"] < MIN_CONTEXTS:
            reasons.append(f"train distinct_contexts < {MIN_CONTEXTS}")
        if v["distinct_contexts"] < MIN_CONTEXTS:
            reasons.append(f"val distinct_contexts < {MIN_CONTEXTS}")
        if reasons:
            rejected.append({"f": int(f), "reason": "; ".join(reasons)})
            continue
        tsem = _sem(t)
        vsem = _sem(v)
        out.append({
            "f": int(f),
            "gap": v["mean_loss"] - t["mean_loss"],
            "train_sem": tsem,
            "val_sem": vsem,
            "gap_sem": math.hypot(tsem, vsem),
            "train_n": t["token_count"],
            "val_n": v["token_count"],
        })
    return out, rejected


def bucket_bounds(fs, n_buckets=N_BUCKETS):
    """Equal-count log-f bins over positive f. Returns [(lo, hi)] with lo<=hi
    (lo==hi for single-f bins)."""
    fs = np.sort(np.asarray(fs, dtype=float))
    fs = fs[fs > 0]
    if len(fs) == 0:
        return []
    if len(fs) <= n_buckets:
        return [(float(f), float(f)) for f in fs]
    logf = np.log(fs)
    edges = np.unique(np.quantile(logf, np.linspace(0.0, 1.0, n_buckets + 1)))
    bounds = []
    for i in range(len(edges) - 1):
        lo = float(np.exp(edges[i]))
        hi = float(np.exp(edges[i + 1]))
        mask = (fs >= lo) & (fs <= hi)
        if mask.sum() == 0:
            continue
        bounds.append((float(fs[mask].min()), float(fs[mask].max())))
    return bounds


def bin_summary(rows, lo, hi):
    """Weighted-mean token-marginal gap over rows with lo<=f<=hi.

    Weight = train+val tokens.  gap_sem = sqrt(pooled per-f gap variance +
    between-f spread) / sqrt(n_f).  mid = geometric midpoint of the bin's f.
    """
    sub = [r for r in rows if lo <= r["f"] <= hi]
    if not sub:
        return None
    fs = np.array([r["f"] for r in sub], dtype=float)
    gaps = np.array([r["gap"] for r in sub])
    gap_var = np.array([r["gap_sem"] ** 2 for r in sub])
    n = np.array([r["train_n"] + r["val_n"] for r in sub], dtype=float)
    w = n / n.sum()
    mean_gap = float((w * gaps).sum())
    pooled = float((w * (gap_var + (gaps - mean_gap) ** 2)).sum())
    sem_gap = math.sqrt(max(pooled, 0.0)) / math.sqrt(len(sub))
    return {
        "lo": float(lo), "hi": float(hi),
        "mid": math.sqrt(lo * hi),
        "n_f": len(sub),
        "gap": mean_gap,
        "gap_sem": sem_gap,
    }


def two_factor(f, A, beta, c, gamma):
    return A * f ** (-beta) * (1.0 - np.exp(-c * f ** gamma))


def fit_two_factor(fs, gaps):
    """Raw-space weighted fit of the two-factor form. Keep only positive gaps.
    Return (popt, perr, n_used) or None."""
    if len(fs) < 4:
        return None
    fs = np.asarray(fs, dtype=float)
    gaps = np.asarray(gaps, dtype=float)
    keep = gaps > 0
    if keep.sum() < 4:
        return None
    fs, gaps = fs[keep], gaps[keep]
    p0 = [np.median(gaps), 0.5, 0.1, 0.5]
    try:
        popt, pcov = curve_fit(two_factor, fs, gaps, p0=p0,
                               maxfev=20000, bounds=([0, -5, 0, 0], [np.inf, 5, np.inf, 5]))
        perr = np.sqrt(np.diag(pcov))
        return popt, perr, int(keep.sum())
    except Exception:
        return None


def style_ax(ax):
    ax.set_facecolor(PAPER)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(MUTED)
    ax.grid(alpha=0.3, which="both", color=BORDER)


def parse_run_key(run_id):
    seed = 42
    if run_id.endswith("_s43") or run_id.endswith("_s44"):
        base, seed_text = run_id.rsplit("_s", 1)
        run_id = base
        seed = int(seed_text)
    parts = run_id.split("_")
    if parts and parts[0] == "pilot":
        parts = parts[1:]
    if len(parts) == 4 and parts[0] == "ep":
        L, mod, align = parts[1:]
    elif len(parts) == 3 and parts[0] == "freq":
        L, mod, align = "L4", parts[1], parts[2]
    else:
        return None
    if L not in ("L1", "L2", "L3", "L4") or mod not in ("bigram", "trigram", "both", "nogram"):
        return None
    return L, mod, align, seed


def main():
    # collect runs by (L, module, alignment, seed)
    runs = {}
    legacy_count = 0
    rejected_count = 0
    for run_id, physical_id, run_dir in canonical_run_dirs(RUNS_DIR):
        parsed = parse_run_key(run_id)
        if parsed is None:
            continue
        L, mod, align, seed = parsed
        summary_path = os.path.join(run_dir, "summary.json")
        if not os.path.exists(summary_path):
            rejected_count += 1
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        if not is_current_scaling_summary(summary, physical_id):
            rejected_count += 1
            continue
        if int(summary.get("seed", seed)) != int(seed):
            rejected_count += 1
            continue
        exact = load_exact(run_dir)
        if not exact:
            continue
        runs[(L, mod, align, seed)] = {
            "run_id": physical_id,
            "run_dir": run_dir,
            "exact": exact,
        }
    for run_dir in glob.glob(os.path.join(RUNS_DIR, "*")):
        if os.path.isdir(run_dir) and not os.path.basename(run_dir).endswith("_fixed"):
            legacy_count += 1
    if legacy_count:
        print(f"ignored {legacy_count} non-canonical scaling directories (expected *_fixed)")
    if rejected_count:
        print(f"ignored {rejected_count} scaling directories with an invalid run contract")

    # ---- per-branch figures: raw (debug) + binned (canonical), log-log ----
    for branch in ("bigram", "trigram"):
        # collect per-f rows from all four module arms at the final step
        raw_rows = {}   # mod -> rows
        for mod in ("bigram", "trigram", "both", "nogram"):
            key = ("L4", mod, "fs", 42)
            if key not in runs:
                continue
            exact = runs[key]["exact"]
            if not exact:
                continue
            rows, _ = marginal_gap_rows(exact[-1], branch)
            # drop non-positive gaps (log undefined) and f=0 (novel)
            rows = [r for r in rows if r["f"] > 0 and r["gap"] > 0]
            if rows:
                raw_rows[mod] = rows

        if not raw_rows:
            print(f"no eligible points for branch={branch}")
            continue

        # ---- RAW debug plot: every per-f point with SEM ----
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor(PAPER)
        for mod, rows in raw_rows.items():
            xs = [r["f"] for r in rows]
            ys = [r["gap"] for r in rows]
            ye = [r["gap_sem"] for r in rows]
            ax.errorbar(xs, ys, yerr=ye, fmt=".", markersize=3, alpha=0.55,
                        color=MOD_COLORS[mod], label=f"{mod} (n={len(rows)})",
                        elinewidth=0.8, capsize=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("exact train frequency f (log)")
        ax.set_ylabel("token-marginal gap (log)")
        ax.set_title(f"{branch} · raw per-f points (debug; final step, L4 fs)")
        ax.legend()
        style_ax(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS_DIR, f"freq_gap_{branch}_final_raw.png"), dpi=150)
        plt.close(fig)
        print(f"saved freq_gap_{branch}_final_raw.png")

        # ---- BINNED canonical plot: geometric midpoint + h/v error bars ----
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor(PAPER)
        for mod, rows in raw_rows.items():
            fs_vals = sorted(r["f"] for r in rows)
            bins = bucket_bounds(fs_vals, N_BUCKETS)
            summaries = [s for s in (bin_summary(rows, lo, hi) for lo, hi in bins) if s]
            if not summaries:
                continue
            xs = [s["mid"] for s in summaries]
            ys = [s["gap"] for s in summaries]
            xerr = [(s["mid"] - s["lo"], s["hi"] - s["mid"]) for s in summaries]
            yerr = [s["gap_sem"] for s in summaries]
            ax.errorbar(xs, ys, xerr=np.array(xerr).T, yerr=yerr, fmt="o-",
                        markersize=5, linewidth=1.6, capsize=2.5,
                        color=MOD_COLORS[mod],
                        label=f"{mod} (n={len(fs_vals)} f, {len(summaries)} bins)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("exact train frequency f (log)")
        ax.set_ylabel("token-marginal gap (log)")
        ax.set_title(f"{branch} · binned gap(f), {N_BUCKETS} log-f bins "
                     f"(final step, L4 fs)\n"
                     "x = geometric bin midpoint ± f-range; y = pooled SEM; "
                     "non-positive gaps & f=0 excluded")
        ax.legend()
        style_ax(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS_DIR, f"freq_gap_{branch}_final.png"), dpi=150)
        plt.close(fig)
        print(f"saved freq_gap_{branch}_final.png")

    # ---- two-factor fit per module (final step, L4 fs), per seed ----
    manifest = []
    for branch in ("bigram", "trigram"):
        for mod in ("bigram", "trigram"):
            for seed in sorted({k[3] for k in runs}):
                key = ("L4", mod, "fs", seed)
                if key not in runs:
                    continue
                entry = runs[key]["exact"][-1]
                rows, rejected = marginal_gap_rows(entry, branch)
                fs = np.array([r["f"] for r in rows])
                gaps = np.array([r["gap"] for r in rows])
                fit_rejected = [
                    {"f": int(f), "reason": "non-positive gap or f=0; excluded from multiplicative fit"}
                    for f in fs if f <= 0 or gaps[fs == f][0] <= 0
                ]
                res = fit_two_factor(fs, gaps)
                if res is None:
                    manifest.append({"branch": branch, "module": mod, "seed": seed, "fit": "failed",
                                     "n_f": int(len(fs)), "n_used": 0,
                                     "excluded_frequency_bins": rejected + fit_rejected})
                    print(f"{branch}/{mod} s{seed}: fit failed ({len(fs)} f points)")
                    continue
                popt, perr, n_used = res
                manifest.append({"branch": branch, "module": mod, "seed": seed, "fit": "ok",
                                 "A": float(popt[0]), "beta": float(popt[1]),
                                 "c": float(popt[2]), "gamma": float(popt[3]),
                                 "perr": [float(x) for x in perr],
                                 "n_f": int(len(fs)), "n_used": int(n_used),
                                 "excluded_frequency_bins": rejected + fit_rejected})
                print(f"{branch}/{mod} s{seed}: A={popt[0]:.3f} beta={popt[1]:.3f} "
                      f"c={popt[2]:.4f} gamma={popt[3]:.3f} (n={n_used}/{len(fs)})")
    with open(os.path.join(FIGS_DIR, "fit_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print("saved fit_manifest.json")


if __name__ == "__main__":
    main()
