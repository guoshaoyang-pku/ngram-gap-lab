#!/usr/bin/env python3
"""ngram-gap-lab · row-level loss/gap analysis (pilot).

Recovers per-hash-table-row train/val loss and gap for a run whose final
model was saved with --save_final_model.

Data flow:
  1. read summary.json (config) + final_model.pt (state_dict) + probe_tokens.npz
     (fixed train probe chunks, shape (n_probe, B, T+1), where each chunk row i
      is token_i; input = chunk[:, :-1], target = chunk[:, 1:]).
  2. rebuild NanoGPT(cfg), load weights, run in eval (no_grad).
  3. per-token loss via ngram_freq.compute_per_token_loss on:
       - the fixed train probe (train side)
       - a fixed val probe built from val shards with the same chunking
         semantics as training (first `val_batches` batches of val stream).
  4. map each token to n-gram table rows via table_occupancy.hash_rows_for_branch
     (identical hash to the model), for bigram AND trigram.
  5. aggregate per (branch, layer, hash, row): train loss mean, val loss mean,
     gap = val - train, token counts, distinct contexts, co-occupant count.
  6. write row_level_gap.csv + a pilot figure row_gap_vs_coocc.png

Alignment semantics (verified against train.py + table_occupancy.py):
  - hash_rows_for_branch takes tokens (M, chunk_size=T+1) and returns
    row_id[i] = hash(tokens[i-1], tokens[i])  (prev=tokens[i-1], cur=tokens[i]).
  - Model input = chunk[:, :-1] (T tokens); per-token loss at position i
    predicts chunk[i+1]; the n-gram context for that prediction is
    (chunk[i], chunk[i+1]) = row_id[i+1] (i in 0..T-1).
  - Therefore loss[:, i] <-> rows[:, i+1] for i in 0..T-1  => rows[:, 1:] (T cols)
    pairs with ptl (T cols).  Exact, no off-by-one.

Usage (run from repo root):
  .venv/bin/python tasks/s1_scaling_three_axis/analysis/row_level_gap.py \\
      <run_dir> [--data_dir ...] [--val_batches 4] [--out_prefix ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
CODE_DIR = os.path.join(REPO_ROOT, "tasks", "s1_scaling_three_axis", "code")
sys.path.insert(0, CODE_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import Config, NanoGPT, TokenizedShardDataset, _BASE_BIGRAM_PRIMES, _BASE_TRIGRAM_PRIMES
from ngram_freq import compute_per_token_loss
from table_occupancy import hash_rows_for_branch, expand_bigram_hash_primes


def load_run(run_dir: str):
    with open(os.path.join(run_dir, "summary.json")) as f:
        summary = json.load(f)
    cfg_dict = summary["config"]
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})
    state = torch.load(os.path.join(run_dir, "final_model.pt"),
                       map_location="cpu", weights_only=True)
    return summary, cfg, state


def build_fixed_val_batches(data_dir: str, cfg: Config, n_batches: int,
                            device: torch.device):
    """Same fixed val batches as training: first n_batches of the val stream."""
    val_ds = TokenizedShardDataset(data_dir, cfg.val_shards, cfg.sequence_len,
                                   cfg.device_batch_size, cfg.data_seed)
    val_iter = val_ds.iter_batches(device)
    return [next(val_iter) for _ in range(n_batches)]


def per_token_loss(model, inp, tgt, device):
    inp, tgt = inp.to(device), tgt.to(device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                            dtype=torch.bfloat16):
            ptl = compute_per_token_loss(model, inp, tgt)
    return ptl.cpu().numpy()


def aggregate_branch(model, chunk, ptl, branch, cfg, table_size, primes):
    """Aggregate per (layer0, hash, row) for one side (train or val).

    chunk: (M, T+1) int64 token stream.  ptl: (M, T) per-token loss.
    Returns {row_id: [loss_sum, token_count, set_of_distinct_context_keys]}.
    """
    rows = hash_rows_for_branch(chunk, cfg.vocab_size, table_size, branch, primes)
    li = 0  # layer 0 (pilot aggregates layer 0 only)
    h0, h1 = rows[li][0], rows[li][1]
    out = {}
    for h_idx, h_rows in enumerate(rows[li]):
        tr = h_rows[:, 1:]  # (M, T) aligned with ptl
        flat_rows = tr.ravel()
        flat_loss = ptl.ravel()
        sums = defaultdict(float)
        counts = defaultdict(int)
        for r, l in zip(flat_rows.tolist(), flat_loss.tolist()):
            r = int(r)
            sums[r] += float(l)
            counts[r] += 1
        # distinct context keys per row (from the token stream itself)
        ctx = defaultdict(set)
        prev = np.concatenate([chunk[:, :1], chunk[:, :-1]], axis=1)  # (M, T+1)
        cur = chunk
        if branch == "bigram":
            keys = (prev * cfg.vocab_size + cur)
        else:
            prev2 = np.concatenate([chunk[:, :2], chunk[:, :-2]], axis=1)
            keys = (prev2 * cfg.vocab_size * cfg.vocab_size + prev * cfg.vocab_size + cur)
        # keys[i] = context(chunk[i-1], chunk[i]) = row context at position i
        # we need context for prediction at position i -> keys[:, i+1]
        k_aligned = keys[:, 1:]
        for r, k in zip(flat_rows.tolist(), k_aligned.ravel().tolist()):
            ctx[int(r)].add(int(k))
        out[h_idx] = {
            row_id: (sums[row_id], counts[row_id], ctx[row_id])
            for row_id in counts
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--val_batches", type=int, default=4)
    ap.add_argument("--branches", default="bigram",
                    help="comma-separated: bigram,trigram")
    ap.add_argument("--out_prefix", default="row_level")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary, cfg, state = load_run(args.run_dir)
    data_dir = args.data_dir or cfg.data_dir or os.environ.get("NGLAB_DATA_DIR", "")
    if not data_dir:
        raise SystemExit("--data_dir required (config has no data_dir)")

    print(f"[row] run={summary['run_id']} table_mult={cfg.table_mult} "
          f"branches={args.branches}")
    model = NanoGPT(cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    probe = np.load(os.path.join(args.run_dir, "probe_tokens.npz"))
    if "fixed_train_probe_chunk" in probe:
        probe_chunk = probe["fixed_train_probe_chunk"]  # (n_probe, B, T+1)
        probe_chunk = probe_chunk.reshape(-1, probe_chunk.shape[-1])  # (n_probe*B, T+1)
    else:
        raise SystemExit("probe_tokens.npz missing 'fixed_train_probe_chunk' "
                         "(rerun training with --save_final_model)")
    print(f"[row] fixed train probe chunk {probe_chunk.shape}, "
          f"sha={probe['fixed_train_probe_sha256']}")

    T = probe_chunk.shape[1] - 1
    train_inp = torch.from_numpy(probe_chunk[:, :-1].copy())
    train_tgt = torch.from_numpy(probe_chunk[:, 1:].copy())
    train_ptl = per_token_loss(model, train_inp, train_tgt, device)  # (M, T)

    val_batches = build_fixed_val_batches(data_dir, cfg, args.val_batches, device)
    val_inp_np = np.concatenate([b[0].cpu().numpy() for b in val_batches], axis=0)
    val_tgt_np = np.concatenate([b[1].cpu().numpy() for b in val_batches], axis=0)
    val_ptl = per_token_loss(model, torch.from_numpy(val_inp_np),
                             torch.from_numpy(val_tgt_np), device)  # (B', T)
    # val chunks: we have (inp,tgt) from dataset, rebuild chunk (B', T+1)
    val_chunk = np.concatenate([val_inp_np, val_tgt_np[:, -1:]], axis=1)

    print(f"[row] train probe chunks={probe_chunk.shape} ptl={train_ptl.shape}; "
          f"val chunks={val_chunk.shape} ptl={val_ptl.shape}")

    branches = [b.strip() for b in args.branches.split(",") if b.strip()]
    table_size = cfg.vocab_size * cfg.table_mult
    bigram_primes = expand_bigram_hash_primes(_BASE_BIGRAM_PRIMES, 4)
    trigram_primes = _BASE_TRIGRAM_PRIMES[:3]

    out_rows = []
    for branch in branches:
        primes = bigram_primes if branch == "bigram" else trigram_primes
        train_agg = aggregate_branch(model, probe_chunk, train_ptl, branch,
                                     cfg, table_size, primes)
        val_agg = aggregate_branch(model, val_chunk, val_ptl, branch,
                                   cfg, table_size, primes)
        for h_idx in train_agg:
            for row_id, (s_tr, n_tr, ctx_tr) in train_agg[h_idx].items():
                if row_id not in val_agg[h_idx]:
                    continue
                s_val, n_val, ctx_val = val_agg[h_idx][row_id]
                if n_tr == 0 or n_val == 0:
                    continue
                out_rows.append({
                    "branch": branch,
                    "layer": 0,
                    "hash": h_idx,
                    "row": row_id,
                    "train_tokens": n_tr,
                    "val_tokens": n_val,
                    "distinct_contexts_train": len(ctx_tr),
                    "distinct_contexts_val": len(ctx_val),
                    "train_mean": s_tr / n_tr,
                    "val_mean": s_val / n_val,
                    "gap": s_val / n_val - s_tr / n_tr,
                })

    csv_path = f"{args.out_prefix}.csv"
    fields = ["branch", "layer", "hash", "row", "train_tokens", "val_tokens",
              "distinct_contexts_train", "distinct_contexts_val",
              "train_mean", "val_mean", "gap"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"[row] wrote {len(out_rows)} rows -> {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for idx, branch in enumerate(branches):
        rows = [r for r in out_rows if r["branch"] == branch]
        ax = axes[idx]
        ax.set_title(f"{branch}: row-level gap vs distinct contexts (train probe)")
        for h in sorted({r["hash"] for r in rows}):
            sub = [r for r in rows if r["hash"] == h]
            xs = [r["distinct_contexts_train"] for r in sub]
            ys = [r["gap"] for r in sub]
            ax.scatter(xs, ys, s=8, alpha=0.4, label=f"hash{h}")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("distinct contexts hashed to row (train probe)")
        ax.set_ylabel("row-level gap (val_mean - train_mean)")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig_path = f"{args.out_prefix}_gap_vs_contexts.png"
    fig.savefig(fig_path, dpi=150)
    print(f"[row] wrote {fig_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
