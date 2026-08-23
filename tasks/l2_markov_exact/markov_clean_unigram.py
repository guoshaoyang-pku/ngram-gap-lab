#!/usr/bin/env python3
"""toy/markov_clean_unigram.py

Cleanest-possible verification of the exact unigram gap theory on the
l1/l2 second-order Markov chain. Pure numpy + matplotlib, one seeded RNG,
no torch, no shell state, no external data.

Process (vocab size M, alpha = 1-l1-l2 > 0):
    X_{t+1} = X_t      w.p. l1        (repeat current token)
    X_{t+1} = X_{t-1}  w.p. l2        (repeat previous token)
    X_{t+1} ~ Uniform(M) w.p. alpha   (uniform restart)

Models (deliberately minimal, plain count tables):
    unigram: q(c)        = softmax(u),           u in R^M
    bigram : q(c|b)      = softmax(U[b]),        U in R^{M x M}   (B -> C)
Both trained by exact full-batch softmax GD on count sufficient statistics.

Exact theory under test (uniform marginal pi = 1/M):
    r   = P(X_{t+1}=X_t) = (l1 + alpha/M)/(1-l2)
    gam = (1 + l1 + l1*l2 - l2^2) / ((1-l2)(1-l1-l2))   (= 9 for 1/4, 1/2)
    gap_inf = D_KL(pi||phat) + D_KL(phat||pi)  >= 0     (exact identity)
    E[gap_inf]   = gam (M-1) / N
    E[L_train]   = log M - gam (M-1)/(2N)
    E[L_val]     = log M + gam (M-1)/(2N)
    epoch transient (full-batch GD, lr eta):
        q_t = pi + beta_t (phat - pi),  beta_t = 1 - (1 - eta/M)^t
        E[gap(t)]    = beta_t gam (M-1)/N
        E[L_train(t)]= log M - (beta_t - beta_t^2/2) gam (M-1)/N
        E[L_val(t)]  = log M + (beta_t^2/2) gam (M-1)/N
    fluctuation:  std(gap)/mean = sqrt(2/(M-1))   (chi^2_{M-1})
    bigram val floor: H(C|B) = -r log r - (1-r) log((1-r)/(M-1))

Dataloader note: the generator is strictly sequential in time (step t reads
only st[t-1], st[t-2]); a previous vectorized version read uninitialized
future entries and produced out-of-range token ids. We assert the token
range and check the empirical marginal and adjacent-repeat rate against
their closed forms before training.

Outputs: toy/results/markov_clean/{fig1,fig2,fig3}.{png,svg}, results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "toy" / "results" / "markov_clean"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- config ----------------
M = 512
L1, L2 = 0.25, 0.5
ALPHA = 1.0 - L1 - L2
N = 2 ** 21                 # tokens per epoch
SEEDS = (42, 43, 44)
EPOCHS = 5
ETA_SHORT = 0.12            # matches markov_bc_5epoch_fast protocol
ETA_LONG = 2.0              # long-run saturation run
EPOCHS_LONG = 3000
rng = np.random.default_rng(20260811)

LOGM = float(np.log(M))

# ---------------- exact theory ----------------

def gamma_closed(l1: float, l2: float) -> float:
    return (1.0 + l1 + l1 * l2 - l2 ** 2) / ((1.0 - l2) * (1.0 - l1 - l2))

R_ADJ = (L1 + ALPHA / M) / (1.0 - L2)           # P(X_{t+1}=X_t)
GAM = gamma_closed(L1, L2)                      # = 9 exactly
GAP_INF = GAM * (M - 1) / N                     # E[gap] at convergence
HALF = GAM * (M - 1) / (2.0 * N)                # train/val symmetric offset
H_CB = (-R_ADJ * np.log(R_ADJ)
        - (1 - R_ADJ) * np.log((1 - R_ADJ) / (M - 1)))   # bigram val floor

def beta_epoch(e, eta):
    return 1.0 - (1.0 - eta / M) ** e

print(f"M={M} l1={L1} l2={L2} alpha={ALPHA} N={N}")
print(f"theory: r={R_ADJ:.6f}  gamma={GAM:.6f} (exact 9)  "
      f"E[gap_inf]={GAP_INF:.6e}  half={HALF:.6e}  H(C|B)={H_CB:.6f}  "
      f"logM={LOGM:.6f}")

# ---------------- dataloader (sequential, self-validating) ----------------

def simulate(n_steps: int, n_chains: int, seed_rng) -> np.ndarray:
    """S parallel independent chains, strictly sequential in time.
    Step t reads only st[t-1], st[t-2] (already filled). Returns (S, N)."""
    u = seed_rng.random((n_steps, n_chains))
    draws = seed_rng.integers(M, size=(n_steps, n_chains), dtype=np.int32)
    st = np.empty((n_steps, n_chains), dtype=np.int32)
    st[0] = draws[0]
    st[1] = draws[1]
    c12 = L1 + L2
    for t in range(2, n_steps):
        r = u[t]
        st[t] = np.where(r < L1, st[t - 1],
                         np.where(r < c12, st[t - 2], draws[t]))
    return st.T

def validate_chains(name: str, st: np.ndarray) -> None:
    lo, hi = int(st.min()), int(st.max())
    assert lo >= 0 and hi < M, f"{name}: token out of range [{lo},{hi}]"
    marg = np.bincount(st.ravel(), minlength=M) / st.size
    marg_err = float(np.abs(marg - 1.0 / M).max())
    rep = float((st[:, 1:] == st[:, :-1]).mean())
    # tolerances ~3 sigma with Markov inflation gamma=9 already included
    assert marg_err < 1e-3, f"{name}: marginal off by {marg_err}"
    assert abs(rep - R_ADJ) < 3e-3, f"{name}: repeat rate {rep} != r={R_ADJ}"
    print(f"  [{name}] ok: range=[{lo},{hi}] max|marg-1/M|={marg_err:.2e} "
          f"repeat={rep:.5f} (r={R_ADJ:.5f})")

# ---------------- softmax table helpers ----------------

def softmax_rows(U: np.ndarray) -> np.ndarray:
    Z = U - U.max(axis=-1, keepdims=True)
    np.exp(Z, out=Z)
    Z /= Z.sum(axis=-1, keepdims=True)
    return Z

# ---------------- experiment 1: unigram, 5 epochs, fixed vs fresh ----------------
print("\n[1] unigram table, 5 epochs, eta=0.12 (matches markov_bc_5epoch_fast)")
S = len(SEEDS)
train5 = simulate(N, S, rng)                      # fixed train set per seed
val = simulate(N, S, rng)                         # independent val per seed
fresh5 = simulate(N, S * (EPOCHS + 1), rng).reshape(S, EPOCHS + 1, N)  # indep chunks
validate_chains("train5", train5)
validate_chains("val", val)
validate_chains("fresh5", fresh5.reshape(S * (EPOCHS + 1), N))

def run_unigram(replay: str):
    """Full-batch GD u += eta (phat - q). Returns (S, EPOCHS+1, 3): train, val, gap.
    Fresh mode: q_e is evaluated on chunk e (never trained on), then updated on it."""
    out = np.empty((S, EPOCHS + 1, 3))
    for si in range(S):
        u = np.zeros(M)
        ph_fixed = np.bincount(train5[si], minlength=M) / N
        vh = np.bincount(val[si], minlength=M) / N
        for e in range(EPOCHS + 1):
            q = softmax_rows(u)
            if replay == "fixed":
                lt = -float(ph_fixed @ np.log(q))
            else:
                ph_e = np.bincount(fresh5[si, e], minlength=M) / N
                lt = -float(ph_e @ np.log(q))
            lv = -float(vh @ np.log(q))
            out[si, e] = (lt, lv, lv - lt)
            if e == EPOCHS:
                break
            if replay == "fixed":
                u += ETA_SHORT * (ph_fixed - q)
            else:
                u += ETA_SHORT * (np.bincount(fresh5[si, e], minlength=M) / N - q)
    return out

uni_fixed = run_unigram("fixed")
uni_fresh = run_unigram("fresh")

ee = np.arange(EPOCHS + 1)
be = np.array([beta_epoch(e, ETA_SHORT) for e in ee])
th_gap = be * GAP_INF
th_train = LOGM - (be - be ** 2 / 2) * GAP_INF
th_val = LOGM + (be ** 2 / 2) * GAP_INF

print(f"  fixed gap@5ep: mean={uni_fixed[:, -1, 2].mean():.4e} "
      f"std={uni_fixed[:, -1, 2].std():.2e}   theory={th_gap[-1]:.4e}")
print(f"  fresh gap@5ep: mean={uni_fresh[:, -1, 2].mean():+.4e}  (expect ~0)")
print(f"  fixed train@5ep={uni_fixed[:, -1, 0].mean():.7f} theory={th_train[-1]:.7f}")
print(f"  fixed val  @5ep={uni_fixed[:, -1, 1].mean():.7f} theory={th_val[-1]:.7f} "
      f"(logM={LOGM:.7f})")

# ---------------- experiment 2: unigram long-run saturation ----------------
print("\n[2] unigram long-run: eta=2.0, 3000 epochs (fixed replay)")
eval_at = np.unique(np.concatenate([
    np.arange(0, 50), np.geomspace(50, EPOCHS_LONG, 60).astype(int)]))
uni_long = np.empty((S, len(eval_at), 3))
for si in range(S):
    u = np.zeros(M)
    ph = np.bincount(train5[si], minlength=M) / N
    vh = np.bincount(val[si], minlength=M) / N
    q = softmax_rows(u)
    ei = 0
    for e in range(EPOCHS_LONG + 1):
        if e == eval_at[ei]:
            lt = -float(ph @ np.log(q))
            lv = -float(vh @ np.log(q))
            uni_long[si, ei] = (lt, lv, lv - lt)
            ei += 1
            if ei == len(eval_at):
                break
        u += ETA_LONG * (ph - q)
        q = softmax_rows(u)
    # exact converged gap identity for this seed
    gap_exact = float((ph * np.log(ph * M)).sum() + (np.log(1.0 / M / ph) / M).sum())
    print(f"  seed{SEEDS[si]}: gap@final={uni_long[si, ei-1, 2]:.6e}  "
          f"exact D(pi||ph)+D(ph||pi)={gap_exact:.6e}  E[gap_inf]={GAP_INF:.6e}")

be_long = beta_epoch(eval_at, ETA_LONG)
th_gap_long = be_long * GAP_INF

# ---------------- experiment 3: bigram B->C table ----------------
print("\n[3] bigram q(c|b) table, 5 epochs + long-run")
def run_bigram(replay: str, eta: float, epochs: int, eval_grid):
    out = np.empty((S, len(eval_grid), 3))
    for si in range(S):
        U = np.zeros((M, M))
        def counts(seq):
            C = np.zeros((M, M))
            np.add.at(C, (seq[1:-1], seq[2:]), 1.0)   # context b = x_t, target c = x_{t+1}
            return C
        Cf = counts(train5[si])
        Cv = counts(val[si])
        q = softmax_rows(U)
        ei = 0
        for e in range(epochs + 1):
            if e == eval_grid[ei]:
                if replay == "fixed":
                    Ctr = Cf
                else:
                    Ctr = counts(fresh5[si, min(e, EPOCHS)])  # unseen chunk
                rows = Ctr.sum(axis=1, keepdims=True)
                lt = -float((Ctr * np.log(q)).sum() / rows.sum())
                lv = -float((Cv * np.log(q)).sum() / Cv.sum())
                out[si, ei] = (lt, lv, lv - lt)
                ei += 1
                if ei == len(eval_grid):
                    break
            if replay == "fixed":
                rows = Cf.sum(axis=1, keepdims=True)
                U += eta * (Cf / rows - q)
            else:
                Ce = counts(fresh5[si, e])
                rows = Ce.sum(axis=1, keepdims=True)
                U += eta * (Ce / rows - q)
            q = softmax_rows(U)
    return out

grid5 = np.arange(EPOCHS + 1)
bi_fixed = run_bigram("fixed", ETA_SHORT, EPOCHS, grid5)
bi_fresh = run_bigram("fresh", ETA_SHORT, EPOCHS, grid5)
bi_long = run_bigram("fixed", ETA_LONG, EPOCHS_LONG, eval_at)
print(f"  fixed gap@5ep: mean={bi_fixed[:, -1, 2].mean():.4e} "
      f"std={bi_fixed[:, -1, 2].std():.2e}")
print(f"  fresh gap@5ep: mean={bi_fresh[:, -1, 2].mean():+.4e}")
print(f"  long-run: train={bi_long[:, -1, 0].mean():.4f} val={bi_long[:, -1, 1].mean():.4f} "
      f"gap={bi_long[:, -1, 2].mean():.4e}  (H(C|B)={H_CB:.4f})")

# ---------------- experiment 4: gamma via variance ratio ----------------
print("\n[4] gamma check: N*Var(phat_c)/(pi(1-pi)) vs gamma=9")
R4, N4 = 64, 2 ** 17
st4 = simulate(N4, R4, rng)
cnt4 = np.array([np.bincount(st4[s], minlength=M) for s in range(R4)])
ratio = cnt4.var(axis=0, ddof=1) / (N4 * (1.0 / M) * (1 - 1.0 / M))
print(f"  Var ratio over {M} tokens: mean={ratio.mean():.3f} "
      f"(se~{ratio.std()/np.sqrt(M):.3f})  theory gamma={GAM:.3f}")

# ---------------- reconciliation with previous toy runs ----------------
print("\n[5] reconciliation with previous toy results")
prev_path = ROOT / "toy" / "results" / "markov_bc_5epoch_fast" / "summary.json"
recon = {}
if prev_path.exists():
    prev = json.load(open(prev_path))["final"]
    pu = prev["fixed_unigram"]
    print(f"  markov_bc_5epoch_fast fixed_unigram gap = {pu['gap_mean']:.4e} "
      f"+- {pu['gap_std']:.2e}")
    print(f"  this script        fixed_unigram gap = {uni_fixed[:, -1, 2].mean():.4e} "
          f"+- {uni_fixed[:, -1, 2].std():.2e}")
    print(f"  exact theory beta_5*gam(M-1)/N       = {th_gap[-1]:.4e}")
    print(f"  markov_bc_5epoch_fast fresh_unigram gap = {prev['fresh_unigram']['gap_mean']:+.4e} "
          f"(control ~0); this script {uni_fresh[:, -1, 2].mean():+.4e}")
    recon = {"prev_fixed_unigram_gap": pu["gap_mean"],
             "clean_fixed_unigram_gap": float(uni_fixed[:, -1, 2].mean()),
             "theory_gap_5ep": float(th_gap[-1])}

# ---------------- figures ----------------
plt.rcParams.update({"font.size": 9.5})
C_FIX, C_FRESH, C_TH = "#4c72b0", "#c44e52", "#222222"

# fig 1: 5-epoch window (unigram)
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
ax = axes[0]
ax.plot(ee, uni_fixed[:, :, 0].mean(0), "o-", color=C_FIX, lw=1.6, ms=4,
        label="fixed  train")
ax.plot(ee, uni_fixed[:, :, 1].mean(0), "s-", color=C_FRESH, lw=1.6, ms=4,
        label="fixed  val")
ax.plot(ee, uni_fresh[:, :, 1].mean(0), "^:",
        color="#55a868", lw=1.2, ms=4, label="fresh  val")
ax.plot(ee, th_train, "--", color=C_TH, lw=1.2, label="theory train")
ax.plot(ee, th_val, "--", color="0.5", lw=1.2, label="theory val")
ax.set_xlabel("epoch")
ax.set_ylabel("loss (nats)")
ax.set_title(f"(a) unigram losses, 5 epochs (logM={LOGM:.4f})")
ax.legend(fontsize=8, frameon=False)
ax = axes[1]
ax.plot(ee, uni_fixed[:, :, 2].mean(0) * 1e6, "o-", color=C_FIX, lw=1.8, ms=4,
        label="fixed replay (mean of 3 seeds)")
for si in range(S):
    ax.plot(ee, uni_fixed[si, :, 2] * 1e6, "-", color=C_FIX, lw=0.5, alpha=0.35)
ax.plot(ee, uni_fresh[:, :, 2].mean(0) * 1e6, "s-", color=C_FRESH, lw=1.4, ms=4,
        label="fresh replay")
ax.plot(ee, th_gap * 1e6, "--", color=C_TH, lw=1.4,
        label=r"theory $\beta_t\,\gamma(M{-}1)/N$")
ax.axhline(0, color="0.7", lw=0.8)
ax.set_xlabel("epoch")
ax.set_ylabel("gap (1e-6 nats)")
ax.set_title("(b) train-val gap: fixed grows, fresh ~ 0")
ax.legend(fontsize=8, frameon=False)
fig.suptitle("Unigram table on the l1/l2 chain: exact 5-epoch transient", y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT / "fig1_unigram_5epoch.svg")
fig.savefig(OUT / "fig1_unigram_5epoch.png", dpi=150)

# fig 2: long-run saturation
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
ax = axes[0]
m_long = uni_long[:, :, 2].mean(0)
ax.semilogx(eval_at + 1, m_long, "-", color=C_FIX, lw=1.8, label="fixed replay (mean)")
ax.semilogx(eval_at + 1, th_gap_long, "--", color=C_TH, lw=1.4,
            label=r"theory $\beta_t\,\gamma(M{-}1)/N$")
band = th_gap_long * np.sqrt(2.0 / (M - 1))
ax.fill_between(eval_at + 1, th_gap_long - band, th_gap_long + band,
                color="0.4", alpha=0.15, label=r"$\chi^2$ $\pm1\sigma$ band")
ax.axhline(GAP_INF, color=C_FRESH, ls=":", lw=1.2,
           label=rf"plateau $\gamma(M{{-}}1)/N$={GAP_INF:.3e}")
for si in range(S):
    ax.semilogx(eval_at + 1, uni_long[si, :, 2], "-", color=C_FIX, lw=0.4, alpha=0.3)
ax.set_xlabel("epoch (log)")
ax.set_ylabel("gap (nats)")
ax.set_title("(a) gap saturates at gamma(M-1)/N -- no runaway amplification")
ax.legend(fontsize=8, frameon=False)
ax = axes[1]
ax.semilogx(eval_at + 1, uni_long[:, :, 0].mean(0), "-", color=C_FIX, lw=1.8,
            label="train")
ax.semilogx(eval_at + 1, uni_long[:, :, 1].mean(0), "-", color=C_FRESH, lw=1.8,
            label="val")
ax.semilogx(eval_at + 1, LOGM - (be_long - be_long ** 2 / 2) * GAP_INF, "--",
            color=C_TH, lw=1.2, label="theory train")
ax.semilogx(eval_at + 1, LOGM + (be_long ** 2 / 2) * GAP_INF, "--", color="0.5",
            lw=1.2, label="theory val")
ax.axhline(LOGM, color="0.7", lw=0.8, ls=":")
ax.set_xlabel("epoch (log)")
ax.set_ylabel("loss (nats)")
ax.set_title(r"(b) symmetric split $\pm\,\gamma(M{-}1)/2N$ around $\log M$")
ax.legend(fontsize=8, frameon=False)
fig.suptitle("Unigram fixed replay, long run: convergence to the exact plateau", y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT / "fig2_unigram_saturation.svg")
fig.savefig(OUT / "fig2_unigram_saturation.png", dpi=150)

# fig 3: bigram
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
ax = axes[0]
ax.semilogx(eval_at + 1, bi_long[:, :, 0].mean(0), "-", color=C_FIX, lw=1.8,
            label="train (fixed)")
ax.semilogx(eval_at + 1, bi_long[:, :, 1].mean(0), "-", color=C_FRESH, lw=1.8,
            label="val (fixed)")
ax.axhline(H_CB, color=C_TH, ls="--", lw=1.2,
           label=f"H(C|B) closed form = {H_CB:.4f}")
ax.axhline(LOGM, color="0.7", ls=":", lw=1.0, label=f"log M = {LOGM:.4f}")
ax.set_xlabel("epoch (log)")
ax.set_ylabel("loss (nats)")
ax.set_title("(a) bigram B->C learns real structure: log M -> H(C|B)")
ax.legend(fontsize=8, frameon=False)
ax = axes[1]
ax.plot(grid5, bi_fixed[:, :, 2].mean(0) * 1e6, "o-", color=C_FIX, lw=1.8, ms=4,
        label="fixed replay")
for si in range(S):
    ax.plot(grid5, bi_fixed[si, :, 2] * 1e6, "-", color=C_FIX, lw=0.5, alpha=0.35)
ax.plot(grid5, bi_fresh[:, :, 2].mean(0) * 1e6, "s-", color=C_FRESH, lw=1.4, ms=4,
        label="fresh replay")
ax.axhline(0, color="0.7", lw=0.8)
ax.set_xlabel("epoch")
ax.set_ylabel("gap (1e-6 nats)")
ax.set_title("(b) bigram 5-epoch gap: small, fresh ~ 0")
ax.legend(fontsize=8, frameon=False)
fig.suptitle("Bigram table q(c|b): loss reduction without a meaningful gap", y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT / "fig3_bigram.svg")
fig.savefig(OUT / "fig3_bigram.png", dpi=150)

# ---------------- results.json ----------------
results = {
    "config": {"M": M, "lambda1": L1, "lambda2": L2, "N": N,
               "seeds": list(SEEDS), "eta_short": ETA_SHORT,
               "eta_long": ETA_LONG, "epochs_long": EPOCHS_LONG},
    "theory": {"r": R_ADJ, "gamma": GAM, "gap_inf": GAP_INF, "half": HALF,
               "H_CB": H_CB, "gap_5ep": float(th_gap[-1]),
               "train_5ep": float(th_train[-1]), "val_5ep": float(th_val[-1])},
    "unigram_5ep": {
        "fixed_gap_mean": float(uni_fixed[:, -1, 2].mean()),
        "fixed_gap_std": float(uni_fixed[:, -1, 2].std()),
        "fresh_gap_mean": float(uni_fresh[:, -1, 2].mean()),
        "fixed_train": float(uni_fixed[:, -1, 0].mean()),
        "fixed_val": float(uni_fixed[:, -1, 1].mean())},
    "unigram_long": {"gap_final_mean": float(uni_long[:, -1, 2].mean()),
                     "gap_final_std": float(uni_long[:, -1, 2].std())},
    "bigram": {"fixed_gap_5ep": float(bi_fixed[:, -1, 2].mean()),
               "fresh_gap_5ep": float(bi_fresh[:, -1, 2].mean()),
               "long_train": float(bi_long[:, -1, 0].mean()),
               "long_val": float(bi_long[:, -1, 1].mean()),
               "long_gap": float(bi_long[:, -1, 2].mean())},
    "gamma_var_ratio": float(ratio.mean()),
    "reconciliation": recon,
}
json.dump(results, open(OUT / "results.json", "w"), indent=1)
print(f"\nsaved figures + results.json -> {OUT}")
print("DONE")
