#!/usr/bin/env python3
"""Build the frequency-vs-gap figure for the 5-gram experiment.

Reads ``allgram_frequency_decomposition.jsonl`` (written by trainer.py) and
produces ``docs/figs/fig_ngram5_freq_gap.json`` with, per probe step:
  - per-bucket train_loss / val_loss / gap
  - bucket label (frequency range)

This is a thin adapter mirroring ``tools/build_exp6_freq_gap.py`` but keyed
on ``branch="fivegram"`` (the upstream tool expects bigram/trigram).

Usage:
  python build_ngram5_freq_gap.py <run_dir>
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "docs" / "figs"


def bucket_label(raw: str) -> str:
    if raw == "0_0":
        return "novel"
    if raw.startswith("ge_"):
        n = int(raw[3:])
        return f"{n}+" if n < 1000 else f"{n//1000}k+"
    lo, hi = raw.split("_")
    if lo == hi:
        return lo
    return f"{lo}-{hi}"


def bucket_order_key(raw: str) -> int:
    if raw == "0_0":
        return -1
    if raw.startswith("ge_"):
        return int(raw[3:]) + 100000
    return int(raw.split("_")[0])


def main(run_dir: Path) -> None:
    decomp_path = run_dir / "allgram_frequency_decomposition.jsonl"
    if not decomp_path.exists():
        raise SystemExit(f"missing {decomp_path}")
    recs = [json.loads(l) for l in decomp_path.read_text().splitlines() if l.strip()]
    # keep only fivegram branch, merged records (those with both train & val)
    recs = [r for r in recs
            if r.get("branch") == "fivegram"
            and r.get("within_bucket_gap") is not None]
    if not recs:
        raise SystemExit("no merged fivegram records found (need probe steps with train+val samples)")

    by_step: dict[int, list[dict]] = defaultdict(list)
    for r in recs:
        by_step[r["step"]].append(r)

    out = {"branch": "fivegram", "steps": {}}
    for step in sorted(by_step):
        rows = sorted(by_step[step], key=lambda r: bucket_order_key(r["bucket"]))
        out["steps"][str(step)] = [
            {
                "bucket": r["bucket"],
                "label": bucket_label(r["bucket"]),
                "train_loss": r["train_loss"],
                "val_loss": r["val_loss"],
                "gap": r["within_bucket_gap"],
                "train_fraction": r.get("train_fraction"),
                "val_fraction": r.get("val_fraction"),
                "exact_global_gap_contribution": r.get("exact_global_gap_contribution"),
            }
            for r in rows
        ]

    FIGS.mkdir(parents=True, exist_ok=True)
    out_path = FIGS / "fig_ngram5_freq_gap.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}")
    print(f"  fivegram: steps={sorted(by_step)}, buckets per step="
          f"{[len(by_step[s]) for s in sorted(by_step)]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: build_ngram5_freq_gap.py <run_dir>")
    main(Path(sys.argv[1]).resolve())
