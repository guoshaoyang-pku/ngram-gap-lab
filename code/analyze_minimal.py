#!/usr/bin/env python3
"""Analyze the minimal 5-gram matrix runs (6 arms).

For each run dir under RUNS_DIR, read:
  - probe_details.jsonl (manifest: step/split/npz path)
  - probe_details/step_*.npz (target_losses, frequencies at target position)

Outputs a comparison table + per-frequency gap analysis:

  runs_dir/
    matrix_summary.csv      # per arm: mean target loss (train/val), gap
    matrix_summary.json
    matrix_gap_curves.png   # target-position train/val loss curves per arm
    matrix_freq_gap.csv     # per-frequency gap at final probe, per arm
    matrix_freq_gap.png

Usage:
  python code/analyze_minimal.py --runs-dir <runs_dir> [--out-dir <out>]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_probe_manifest(run_dir: Path):
    """Return {step: {train: path, val: path}} from probe_details.jsonl."""
    manifest_path = run_dir / "probe_details.jsonl"
    if not manifest_path.exists():
        return {}
    steps: dict[int, dict] = {}
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        step = int(rec["step"])
        split = rec["split"]
        path = Path(rec["path"])
        if not path.is_absolute():
            path = run_dir / path
        steps.setdefault(step, {})[split] = path
    return steps


def target_mean_loss(npz_path: Path) -> float:
    with np.load(npz_path, allow_pickle=True) as z:
        return float(z["target_losses"].mean())


def target_freq_gap(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (freqs, mean_train_loss_per_freq, mean_val_loss_per_freq)."""
    with np.load(npz_path, allow_pickle=True) as z:
        freqs = z["frequencies"].ravel()
        losses = z["target_losses"].ravel()
    order = np.argsort(freqs)
    freqs_s = freqs[order]
    losses_s = losses[order]
    # group by frequency
    uniq, starts = np.unique(freqs_s, return_index=True)
    splits = np.split(losses_s, starts[1:])
    means = np.array([s.mean() for s in splits])
    return uniq, means, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True,
                    help="dir containing the 6 arm run dirs")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: <runs_dir>)")
    ap.add_argument("--arms", default=None,
                    help="optional comma-separated list of arm subdirs to include")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else runs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_dirs = []
    if args.arms:
        arm_dirs = [runs_dir / a for a in args.arms.split(",") if a.strip()]
    else:
        # auto-detect: any subdir with probe_details.jsonl
        for d in sorted(runs_dir.iterdir()):
            if d.is_dir() and (d / "probe_details.jsonl").exists():
                arm_dirs.append(d)

    if not arm_dirs:
        raise SystemExit(f"No arm run dirs found under {runs_dir}")

    rows = []
    freq_rows = []
    for arm_dir in arm_dirs:
        name = arm_dir.name
        manifest = load_probe_manifest(arm_dir)
        if not manifest:
            print(f"[warn] {name}: no probe_details.jsonl, skipping")
            continue
        steps = sorted(manifest)
        # overall target-position loss at each probe step
        per_step = []
        for step in steps:
            splits = manifest[step]
            rec = {"step": step}
            for split, path in splits.items():
                if path.exists():
                    rec[f"{split}_loss"] = target_mean_loss(path)
            per_step.append(rec)
        final = per_step[-1] if per_step else {}
        gap = (final.get("val_loss", float("nan"))
               - final.get("train_loss", float("nan")))
        rows.append({
            "arm": name,
            "final_step": final.get("step", ""),
            "train_loss": final.get("train_loss", float("nan")),
            "val_loss": final.get("val_loss", float("nan")),
            "gap": gap,
            "n_probes": len(per_step),
        })

        # per-frequency gap at final step (train + val)
        if final.get("step") is not None:
            tr_path = manifest[final["step"]].get("train")
            va_path = manifest[final["step"]].get("val")
            if tr_path and tr_path.exists() and va_path and va_path.exists():
                with np.load(tr_path, allow_pickle=True) as z:
                    tr_f = z["frequencies"].ravel()
                    tr_l = z["target_losses"].ravel()
                with np.load(va_path, allow_pickle=True) as z:
                    va_f = z["frequencies"].ravel()
                    va_l = z["target_losses"].ravel()
                freqs = np.unique(np.concatenate([tr_f, va_f]))
                for f in freqs:
                    freq_rows.append({
                        "arm": name,
                        "frequency": int(f),
                        "train_loss": float(tr_l[tr_f == f].mean()),
                        "val_loss": float(va_l[va_f == f].mean()),
                        "gap": float(va_l[va_f == f].mean()
                                      - tr_l[tr_f == f].mean()),
                        "train_count": int((tr_f == f).sum()),
                        "val_count": int((va_f == f).sum()),
                    })

    # ---- CSV summary ----
    with open(out_dir / "matrix_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["arm"])
        w.writeheader()
        w.writerows(rows)
    print("\n=== MATRIX SUMMARY ===")
    print(f"{'arm':<26}{'step':>7}{'train':>9}{'val':>9}{'gap':>9}")
    for r in rows:
        print(f"{r['arm']:<26}{r['final_step']:>7}"
              f"{r['train_loss']:>9.3f}{r['val_loss']:>9.3f}{r['gap']:>9.3f}")

    with open(out_dir / "matrix_summary.json", "w") as fh:
        json.dump(rows, fh, indent=2)

    # ---- per-frequency CSV ----
    if freq_rows:
        with open(out_dir / "matrix_freq_gap.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(freq_rows[0].keys()))
            w.writeheader()
            w.writerows(freq_rows)
        print(f"\n=== PER-FREQUENCY GAP (final probe, {len(freq_rows)} rows) ===")
        print("  -> matrix_freq_gap.csv")

    # ---- gap curves PNG ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        for arm_dir in arm_dirs:
            name = arm_dir.name
            manifest = load_probe_manifest(arm_dir)
            steps = sorted(manifest)
            xs, tr, va = [], [], []
            for step in steps:
                splits = manifest[step]
                tr_path = splits.get("train")
                va_path = splits.get("val")
                if tr_path and tr_path.exists() and va_path and va_path.exists():
                    xs.append(step)
                    tr.append(target_mean_loss(tr_path))
                    va.append(target_mean_loss(va_path))
            if xs:
                ax[0].plot(xs, tr, marker="o", label=name, linewidth=1.2)
                ax[1].plot(xs, va, marker="o", label=name, linewidth=1.2)
        ax[0].set_title("Target-position TRAIN loss")
        ax[0].set_xlabel("step")
        ax[1].set_title("Target-position VAL loss")
        ax[1].set_xlabel("step")
        for a in ax:
            a.legend(fontsize=7)
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "matrix_gap_curves.png", dpi=130)
        print(f"\n[curves] {out_dir / 'matrix_gap_curves.png'}")
    except Exception as exc:  # matplotlib optional
        print(f"[warn] curve plot skipped: {exc}")

    # ---- per-frequency gap plot ----
    if freq_rows:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            arms = sorted({r["arm"] for r in freq_rows})
            fig, ax = plt.subplots(figsize=(9, 6))
            for arm in arms:
                sel = [r for r in freq_rows if r["arm"] == arm]
                sel.sort(key=lambda r: r["frequency"])
                xs = [r["frequency"] for r in sel]
                ys = [r["gap"] for r in sel]
                ax.plot(xs, ys, marker=".", label=arm, linewidth=1.0)
            ax.set_xscale("log")
            ax.set_xlabel("context frequency r (log)")
            ax.set_ylabel("gap (val - train), target position")
            ax.set_title("Per-frequency gap at final probe (6 arms)")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, which="both")
            fig.tight_layout()
            fig.savefig(out_dir / "matrix_freq_gap.png", dpi=130)
            print(f"[freq] {out_dir / 'matrix_freq_gap.png'}")
        except Exception as exc:
            print(f"[warn] freq plot skipped: {exc}")

    print("\nDone. Files written under", out_dir)


if __name__ == "__main__":
    main()
