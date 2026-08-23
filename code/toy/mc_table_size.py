#!/usr/bin/env python3
"""MC: finite-row count-table learner — table size vs gap.

Question: when the n-gram table has only R rows (vocab*multiplier) and K
distinct contexts hash into those rows, does the per-bucket gap change because
of (a) parameter count alone (capacity) or (b) low-frequency fluctuation
weighting in row collisions?

Model (matching toy-gap-powerlaw-mechanism.md):
  - Each context c has true distribution P(·|c) over K_eff symbols (probabilistic
    rule).  Train draws r iid samples; the model stores the empirical distribution
    p_hat (Dirichlet alpha smoothing) into the row row(c) = hash(c) % R.
  - Row collisions: contexts sharing a row AVERAGE their empirical distributions
    (the table row is shared).  This is the key finite-capacity effect.
  - Honest val: val draws fresh from P(·|c).
  - gap(r) = val_CE(c) - train_CE(c), per context, then aggregated per bucket.

Two regimes to distinguish:
  * capacity-limited: with R rows and K distinct contexts, collisions squeeze
    rare contexts more (their row is polluted by many other low-freq contexts).
  * fluctuation-weighted: even with NO collision (R >= K, exact rows), gap is
    ~ (K_eff-1)/r per context; changing "table size" does nothing unless the
    number of *distinct low-frequency contexts* (and hence their total weight
    in the aggregate) changes.

The sweep: fix K distinct contexts, vary R (rows) from K/8 .. 8K.  Predict
whether the *aggregate* gap over a Zipf-like context-frequency population
moves with R (params-dominated) or stays flat (fluctuation-dominated), and
whether per-bucket gap(r) changes.

Pure numpy, deterministic, seconds.
"""
from __future__ import annotations

import numpy as np

_GLOBAL_RNG = np.random.default_rng(20260810)

def rng():
    return _GLOBAL_RNG

K_EFF = 8            # symbols per context (participation count)
ALPHA = 0.001        # Dirichlet smoothing
R_VALUES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]   # physical rows
K_CTX = 2048         # distinct contexts
BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Zipf-like population: N_r ~ 1/r^2 over buckets (rank exponent 1)
def zipf_weights(rs):
    w = np.array([1.0 / r ** 2 for r in rs])
    return w / w.sum()

N_BUCKET = zipf_weights(BUCKETS) * K_CTX
N_BUCKET = np.round(N_BUCKET).astype(int)
# adjust to exactly K_CTX
N_BUCKET[0] += K_CTX - N_BUCKET.sum()
assert N_BUCKET.sum() == K_CTX

# per-context true distribution P (K_eff symbols, random)
rng0 = np.random.default_rng(20260810)
P = rng0.dirichlet(np.ones(K_EFF), size=K_CTX)

# assign each context an r
ctx_r = np.concatenate([np.full(n, r) for r, n in zip(BUCKETS, N_BUCKET)])


def simulate(R, rows, seed):
    """Return per-bucket gap and aggregate for a given row count R and fixed
    per-context hash addresses (rows = addresses % R)."""
    rng = np.random.default_rng(seed)
    rows = rows % R
    # build per-row averaged empirical distribution by iterating r draws
    # count matrix: K_CTX x K_EFF of train draws
    train_counts = rng.multinomial(ctx_r, P)
    # row-shared empirical: sum counts per row, renormalize + alpha
    row_count = np.zeros((R, K_EFF))
    np.add.at(row_count, rows, train_counts)
    row_total = row_count.sum(axis=1, keepdims=True)
    row_p = (row_count + ALPHA) / (row_total + ALPHA * K_EFF)
    # per-context p_hat = its row's averaged distribution
    p_hat = row_p[rows]

    # per-context train CE (evaluate on its own empirical samples)
    # E_y~p_hat[-ln p_hat] = H(p_hat) -- but the row is averaged over
    # multiple contexts; use the shared row distribution directly:
    train_ce = -np.sum(p_hat * np.log(np.maximum(p_hat, 1e-30)), axis=1)
    # per-context val CE (honest val against true P)
    val_ce = -np.sum(P * np.log(np.maximum(p_hat, 1e-30)), axis=1)
    gap = val_ce - train_ce

    out = {}
    for r, n in zip(BUCKETS, N_BUCKET):
        m = ctx_r == r
        out[r] = float(gap[m].mean())
    agg = float(np.mean(gap))           # context-uniform aggregate
    # token-weighted aggregate (r-weighted, matching train marginal)
    agg_w = float(np.sum(ctx_r * gap) / np.sum(ctx_r))
    return out, agg, agg_w


print(f"K_CTX={K_CTX}, K_EFF={K_EFF}, buckets={BUCKETS}")
print(f"{'R (rows)':>10} {'agg_ctx':>9} {'agg_tok':>9} " + " ".join(f"r={r:<6}" for r in BUCKETS))
results = {}
# Same context hash layout across R: fixed addresses, mod R, so R is the ONLY change.
_rows_rng = np.random.default_rng(20260810)
rows_fixed = _rows_rng.permutation(K_CTX * 16)[:K_CTX]
for R in R_VALUES:
    per_r, agg, agg_w = simulate(R, rows_fixed, seed=1000 + R)
    results[R] = (per_r, agg, agg_w)
    print(f"R={R:>5} {agg:>9.4f} {agg_w:>9.4f} " + " ".join(f"{per_r[r]:>9.4f}" for r in BUCKETS))

print("\n== relative change vs R=4096 (near-exact rows) ==")
base_per_r, base_agg, base_agg_w = results[4096]
for R in R_VALUES:
    per_r, agg, agg_w = results[R]
    rel_agg = agg / base_agg - 1
    rel_agg_w = agg_w / base_agg_w - 1
    print(f"R={R:>5}: agg_ctx rel={rel_agg:+.2%}  agg_tok rel={rel_agg_w:+.2%}")

# log-log slope of aggregate gap vs R (params-dominated => nonzero)
import numpy.polynomial.polynomial as poly
Rarr = np.array(R_VALUES, dtype=float)
aggs = np.array([results[R][1] for R in R_VALUES])
slope = np.polyfit(np.log(Rarr), np.log(aggs), 1)[0]
print(f"\nlog-log slope of aggregate gap vs R (context-uniform): {slope:.3f}  "
      f"(0 => table-size independent / fluctuation-dominated)")

# regime split: below vs above the key count K_CTX (collision vs collision-free)
for lo, hi, label in [(0, K_CTX, "R < K_CTX (collision regime)"),
                      (K_CTX, None, "R >= K_CTX (collision-free regime)")]:
    mask = [(lo <= R < hi) if hi else (R >= lo) for R in R_VALUES]
    if sum(mask) >= 3:
        s = np.polyfit(np.log(Rarr[mask]), np.log(aggs[mask]), 1)[0]
        print(f"  local slope {label}: {s:.3f}")
aggs_w = np.array([results[R][2] for R in R_VALUES])
slope_w = np.polyfit(np.log(Rarr), np.log(aggs_w), 1)[0]
print(f"log-log slope of aggregate gap vs R (token-weighted):     {slope_w:.3f}")


# ============================================================
# REAL STRUCTURE VERIFICATION (t5_low: K_CTX=32768, vocab=2048)
# load = K_CTX / (vocab * M); M=16 is exactly collision-free.
# Observed GPU result: gap saturates at M>=16 (params give nothing),
# collision regime M<16 raises gap ~ +2.6 nat from M=1..16.
# Static count-table MC predicts a continued rise past M=16 -- the
# real model saturates because sparse tables get sparser updates.
# So: LOW-FREQ FLUCTUATION (collision) DOMINATES, params saturate.
# ============================================================
