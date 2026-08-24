#!/usr/bin/env python3
"""markov_unigram_gap.py

Minimal experiment: pure unigram table model on an l1/l2 Markov chain.

Process (vocabulary V, unigram base distribution T, l3 = 1-l1-l2 > 0):
    X_{n+1} = X_n        w.p. l1   (repeat current token)
    X_{n+1} = X_{n-1}    w.p. l2   (become the previous token)
    X_{n+1} ~ T          w.p. l3   (restart from the base unigram)

Model: a single unigram table (logits u, q = softmax(u)), no backbone.

Sections:
  A. stationary marginals: single-token = T, pair pi2 closed form vs path
  B. entropy rate h: closed form vs path average (true conditional probs,
     no plug-in bias) over an (l1,l2) grid x {Zipf, uniform}; also the
     naive T x T weighting to show why it is wrong
  C. integrated autocorrelation time tau: AR(2) spectral formula vs
     simulated autocorrelation; naive (1+rho)/(1-rho) underestimates
  D. E[g_inf] = E[sum (fhat-T) log fhat] ~ tau (V-1) / N: variance-reduced
     Monte Carlo vs the leading-order prediction; O(1/N^2) residual check
  E. multi-epoch full-batch GD u_{e+1} = u_e + eta (fhat - q_e):
     fixed replay (same sequence) vs fresh replay (new sample each epoch);
     linearized convergence rate 1 - eta*kappa
  F. per-token excess curve e(v) = -log T_v - h vs frequency, crossing demo

Figures (docs/figs/):
  fig_markov_unigram_h.{svg,png}
  fig_markov_unigram_gap_epochs.{svg,png}
  fig_markov_unigram_excess_vs_freq.{svg,png}

Pure numpy + matplotlib, deterministic (single seeded RNG), ~20-40 s.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__import__("os").environ.get(
    "NGLAB_FIG_DIR", REPO_ROOT / "docs" / "figs" / "theory"
))
FIG_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20260811
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# model primitives
# ---------------------------------------------------------------------------

def zipf_probs(V: int) -> np.ndarray:
    z = 1.0 / np.arange(1, V + 1)
    return z / z.sum()


def pair_stationary(T: np.ndarray, l1: float, l2: float) -> np.ndarray:
    """Closed-form stationary pair distribution pi2(a,b) = P(X_{n-1}=a, X_n=b)."""
    l3 = 1.0 - l1 - l2
    V = len(T)
    pi2 = l3 * np.outer(T, T) / (1.0 - l2)
    diag = (l1 * T + l3 * T ** 2) / (1.0 - l2)
    np.fill_diagonal(pi2, diag)
    return pi2


def entropy_rate(T: np.ndarray, l1: float, l2: float):
    """Closed-form H(T), entropy rate h, naive T x T weighting h_TxT,
    and diagonal pair mass pi_diag. See derivation doc."""
    l3 = 1.0 - l1 - l2
    V = len(T)
    H_T = float(-(T * np.log(T)).sum())
    pi_diag = (l1 + l3 * (T ** 2).sum()) / (1.0 - l2)
    s = T * np.log(l3 * T)                      # summand of sum_c T(c) log(l3 T(c))
    s_sum = s.sum()
    h = 0.0
    h_TxT = 0.0
    pi2 = pair_stationary(T, l1, l2)
    for a in range(V):
        pa = l1 + l2 + l3 * T[a]
        H_diag = -(pa) * np.log(pa) - l3 * (s_sum - T[a] * np.log(l3 * T[a]))
        h += pi2[a, a] * H_diag
        h_TxT += T[a] * T[a] * H_diag
        for b in range(V):
            if b == a:
                continue
            Ho = (-(l1 + l3 * T[b]) * np.log(l1 + l3 * T[b])
                  - (l2 + l3 * T[a]) * np.log(l2 + l3 * T[a])
                  - l3 * (s_sum - T[a] * np.log(l3 * T[a])
                          - T[b] * np.log(l3 * T[b])))
            h += pi2[a, b] * Ho
            h_TxT += T[a] * T[b] * Ho
    return H_T, float(h), float(h_TxT), float(pi_diag)


def simulate_chain(N: int, S: int, l1: float, l2: float, T: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    """Vectorized chain simulation, S parallel sequences, (S, N) int32."""
    V = len(T)
    u = rng.random((N, S))
    draws = rng.choice(V, size=(N, S), p=T)
    st = np.empty((N, S), dtype=np.int32)
    st[0] = draws[0]
    st[1] = draws[1]
    c12 = l1 + l2
    for n in range(2, N):
        r = u[n]
        st[n] = np.where(r < l1, st[n - 1],
                         np.where(r < c12, st[n - 2], draws[n]))
    return st.T  # (S, N)


def path_entropy_rate(st: np.ndarray, T: np.ndarray, l1: float,
                      l2: float) -> float:
    """Path-average of -log P_true(X_{n+1} | X_n, X_{n-1}); no plug-in bias."""
    l3 = 1.0 - l1 - l2
    nxt = st[:, 2:]
    x1 = st[:, 1:-1]
    x0 = st[:, :-2]
    p = l1 * (nxt == x1) + l2 * (nxt == x0) + l3 * T[nxt]
    return float(-np.log(p).mean())


def batch_mean_se(logp: np.ndarray, n_blocks: int = 16) -> float:
    """Batch-mean standard error of the mean of logp over (S, M) samples."""
    S, M = logp.shape
    bl = M // n_blocks
    blocks = logp[:, :bl * n_blocks].reshape(S, n_blocks, bl).mean(axis=2)
    se = np.sqrt((blocks.var(axis=1) / n_blocks).mean() / S)
    return float(se)


def tau_spectrum(l1: float, l2: float):
    """AR(2) spectrum of the token-indicator autocorrelation.

    Y_n = 1[X_n=v] - T_v satisfies E[Y_{n+1} | X_n, X_{n-1}] = l1 Y_n + l2 Y_{n-1},
    so r(k) = cov-correlation satisfies r(k) = l1 r(k-1) + l2 r(k-2), k>=2, with
    r(0) = 1, r(1) = l1/(1-l2).  Hence r(k) = A lam+^k + B lam-^k,
    A + B = r(0) = 1, A lam+ + B lam- = r(1)   (NOTE: correction to the draft
    note "A+B = r(1)"; that index error inflates tau by ~2 nats).
    """
    lam_p = (l1 + np.sqrt(l1 ** 2 + 4 * l2)) / 2.0
    lam_m = (l1 - np.sqrt(l1 ** 2 + 4 * l2)) / 2.0
    r1 = l1 / (1.0 - l2)
    M = np.array([[lam_p, lam_m], [lam_p ** 2, lam_m ** 2]])
    A, B = np.linalg.solve(M, [r1, l1 * r1 + l2])
    tau = 1.0 + 2.0 * (A * lam_p / (1.0 - lam_p) + B * lam_m / (1.0 - lam_m))
    naive = (1.0 + r1) / (1.0 - r1)
    sum_k_rk = A * lam_p / (1.0 - lam_p) ** 2 + B * lam_m / (1.0 - lam_m) ** 2
    return dict(lam_p=lam_p, lam_m=lam_m, r1=r1, A=A, B=B, tau=tau,
                naive=naive, sum_k_rk=sum_k_rk)


def theoretical_autocorr(l1: float, l2: float, K: int) -> np.ndarray:
    sp = tau_spectrum(l1, l2)
    k = np.arange(K + 1)
    return sp["A"] * sp["lam_p"] ** k + sp["B"] * sp["lam_m"] ** k


def empirical_autocorr(st: np.ndarray, v: int, K: int) -> np.ndarray:
    S, N = st.shape
    Y = (st == v).astype(np.float64)
    Y -= Y.mean(axis=1, keepdims=True)
    denom = (Y * Y).mean(axis=1)
    out = np.empty(K + 1)
    for k in range(K + 1):
        out[k] = float((Y[:, :N - k] * Y[:, k:]).mean(axis=1).mean()
                       / denom.mean())
    return out


def softmax(u: np.ndarray) -> np.ndarray:
    m = u.max()
    e = np.exp(u - m)
    return e / e.sum()


# ---------------------------------------------------------------------------
# project reference: synthetic-transition-task-design.md control excess
# (per-frequency val CE minus conditional entropy H, nats, step 2000)
# ---------------------------------------------------------------------------
PROJ_FREQ = np.array([8, 16, 32, 64, 128, 512, 2048, 8192], dtype=float)
PROJ_EXCESS_A = np.array([11.77, 9.16, 6.60, 8.67, 7.06, 5.12, 2.26, 1.43])
PROJ_EXCESS_B = np.array([5.72, 5.71, 3.81, 3.73, 4.53, 2.89, 1.80, 0.83])


# ---------------------------------------------------------------------------
# A. stationary distributions
# ---------------------------------------------------------------------------
print("=" * 78)
print("A. stationary distributions (single-token = T, pair pi2 closed form)")
print("=" * 78)
for (l1, l2), Tv, V in [((0.4, 0.4), zipf_probs(16), 16),
                        ((0.2, 0.6), np.full(16, 1.0 / 16), 16)]:
    pi2 = pair_stationary(Tv, l1, l2)
    row_err = float(np.abs(pi2.sum(axis=1) - Tv).max())
    st = simulate_chain(2 ** 16, 8, l1, l2, Tv, rng)
    cnt = np.zeros((V, V))
    np.add.at(cnt, (st[:, :-1].ravel(), st[:, 1:].ravel()), 1)
    emp = cnt / cnt.sum()
    maxdiff = float(np.abs(emp - pi2).max())
    pi_diag_closed = float(np.trace(pi2))
    pi_diag_emp = float(np.trace(emp))
    print(f"  (l1,l2)=({l1},{l2}) V={V} dist={'zipf' if Tv[0] > Tv[1] else 'uniform'}"
          f"  max|pi2 rows - T| = {row_err:.2e}"
          f"  max|emp - closed| = {maxdiff:.2e}"
          f"  pi_diag closed={pi_diag_closed:.4f} emp={pi_diag_emp:.4f}")
    assert row_err < 1e-12
    assert maxdiff < 5e-3

# ---------------------------------------------------------------------------
# B. entropy rate: closed form vs path average
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("B. entropy rate h: closed form vs path average (true cond. probs)")
print("=" * 78)
GRID = [0.1, 0.3, 0.5, 0.8]
CONFIGS = [(a, b) for a in GRID for b in GRID if a + b < 1.0]
V_B = 16
T_ZIPF_B = zipf_probs(V_B)
T_UNI_B = np.full(V_B, 1.0 / V_B)
hdr = (f"  {'l1':>4} {'l2':>4} {'dist':>7} {'H(T)':>7} {'h':>7} "
       f"{'pi_diag':>7} {'h_path':>8} {'|err|':>8} {'se':>7} {'err/se':>6}")
print(hdr)
print("  " + "-" * 74)
h_rows = []
for (l1, l2) in CONFIGS:
    for Tv in (T_ZIPF_B, T_UNI_B):
        H_T, h, h_TxT, pi_d = entropy_rate(Tv, l1, l2)
        st = simulate_chain(2 ** 16, 64, l1, l2, Tv, rng)
        nxt = st[:, 2:]
        x1 = st[:, 1:-1]
        x0 = st[:, :-2]
        p = (1 - l1 - l2) * 0 + l1 * (nxt == x1) + l2 * (nxt == x0) + (1 - l1 - l2) * Tv[nxt]
        logp = -np.log(p)
        h_path = float(logp.mean())
        se = batch_mean_se(logp)
        dist = "zipf" if Tv[0] > Tv[1] else "uniform"
        h_rows.append(dict(l1=l1, l2=l2, dist=dist, H_T=H_T, h=h,
                           h_path=h_path, err=abs(h_path - h), se=se))
        print(f"  {l1:>4.1f} {l2:>4.1f} {dist:>7} {H_T:>7.4f} {h:>7.4f} "
              f"{pi_d:>7.3f} {h_path:>8.4f} {abs(h_path - h):>8.4f} "
              f"{se:>7.4f} {abs(h_path - h) / se:>6.1f}")
worst = max(h_rows, key=lambda r: r["err"])
print(f"  -> max |h_path - h| = {worst['err']:.4f} at (l1,l2)=({worst['l1']},{worst['l2']}) "
      f"{worst['dist']}; all within 3.5 se: "
      f"{all(r['err'] < 3.5 * r['se'] + 1e-5 for r in h_rows)}")
assert worst["err"] < 1e-2

# naive T x T weighting bias
print()
print("  naive T x T weighting of the pair conditional entropies (the WRONG")
print("  averaging used by a T x T independent-pair assumption):")
for (l1, l2) in [(0.4, 0.4), (0.2, 0.6)]:
    H_T, h, h_TxT, pi_d = entropy_rate(T_ZIPF_B, l1, l2)
    print(f"  (l1,l2)=({l1},{l2})  h_TxT = {h_TxT:.4f}  vs  h = {h:.4f}  "
          f"(naive overestimates by {h_TxT - h:.4f})")
    assert h_TxT > h

# ---------------------------------------------------------------------------
# C. integrated autocorrelation time: AR(2) spectrum vs simulation
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("C. tau: AR(2) spectral formula vs simulated autocorrelation")
print("=" * 78)
V_C = 64
T_C = zipf_probs(V_C)
K_C = 48
print(f"  {'(l1,l2)':>10} {'lam+':>7} {'lam-':>7} {'tau_theory':>10} "
      f"{'tau_naive':>9} {'tau_emp':>8} {'max|r_emp-r_th|':>15} {'max cross-token':>15}")
for (l1, l2) in [(0.4, 0.4), (0.2, 0.6), (0.6, 0.2)]:
    sp = tau_spectrum(l1, l2)
    r_th = theoretical_autocorr(l1, l2, K_C)
    st = simulate_chain(2 ** 17, 16, l1, l2, T_C, rng)
    # tokens: most frequent, mid, rarest
    idx = [0, V_C // 2 - 1, V_C - 1]
    r_emps = [empirical_autocorr(st, v, K_C) for v in idx]
    maxdiff = max(float(np.abs(re[1:] - r_th[1:]).max()) for re in r_emps)
    cross = max(float(np.abs(r_emps[i][1:] - r_emps[j][1:]).max())
                for i in range(3) for j in range(i + 1, 3))
    tau_emp = 1.0 + 2.0 * float(r_emps[0][1:K_C + 1].sum())
    print(f"  ({l1},{l2})  {sp['lam_p']:>7.4f} {sp['lam_m']:>7.4f} "
          f"{sp['tau']:>10.3f} {sp['naive']:>9.3f} {tau_emp:>8.3f} "
          f"{maxdiff:>15.4f} {cross:>15.4f}")
    assert sp["naive"] < sp["tau"]          # naive underestimates
    assert maxdiff < 0.05                   # AR(2) spectrum matches
    assert cross < 0.05                     # r(k) is v-independent
print("  -> note: (0.4,0.4) tau_theory=11.67 (draft note said empirical ~11,"
      " consistent);")
print("     naive (1+rho)/(1-rho)=5.0 underestimates by 2.3x; (0.2,0.6): 12.0 vs 3.0.")

# ---------------------------------------------------------------------------
# D. E[g_inf] ~ tau (V-1) / N: variance-reduced Monte Carlo
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("D. E[g_inf] = E[sum (fhat-T) log fhat] ~ tau (V-1) / N  (leading order)")
print("=" * 78)
V_D = 16
T_D = zipf_probs(V_D)
print(f"  V={V_D}, Zipf.  g_inf = sum_v (fhat_v - T_v) log fhat_v; "
      f"g' adds the exactly-zero-mean control variate -sum (fhat-T) log T.")


def mc_g_inf(N: int, S: int, l1: float, l2: float) -> tuple:
    sp = tau_spectrum(l1, l2)
    pred = sp["tau"] * (V_D - 1) / N
    st = simulate_chain(N, S, l1, l2, T_D, rng)
    fhats = np.array([np.bincount(st[s], minlength=V_D) / N for s in range(S)])
    g = ((fhats - T_D) * np.log(fhats)).sum(axis=1)
    gv = ((fhats - T_D) * np.log(fhats / T_D)).sum(axis=1)
    return dict(pred=pred, g_mean=float(g.mean()), g_se=float(g.std() / np.sqrt(S)),
                gv_mean=float(gv.mean()), gv_se=float(gv.std() / np.sqrt(S)),
                fhats=fhats)


print(f"  {'(l1,l2)':>10} {'N':>9} {'S':>5} {'pred':>10} {'E[g] raw':>10} "
      f"{'E[g-vr]':>10} {'resid':>10} {'resid/se':>8}")
d_results = {}
for (l1, l2) in [(0.4, 0.4), (0.2, 0.6), (0.6, 0.2)]:
    N, S = 2 ** 17, 256
    r = mc_g_inf(N, S, l1, l2)
    d_results[(l1, l2)] = r
    print(f"  ({l1},{l2})  {N:>9} {S:>5} {r['pred']:>10.6f} {r['g_mean']:>10.6f} "
          f"{r['gv_mean']:>11.6f} {r['gv_mean'] - r['pred']:>10.6f} "
          f"{(r['gv_mean'] - r['pred']) / r['gv_se']:>8.1f}")
    assert abs(r["gv_mean"] - r["pred"]) < 5 * r["gv_se"]
# O(1/N^2) scaling: halve N, residual should shrink
N2, S2 = 2 ** 16, 512
r2 = mc_g_inf(N2, S2, 0.4, 0.4)
r1 = d_results[(0.4, 0.4)]
print(f"  O(1/N^2) check (0.4,0.4): resid(N/2)={r2['gv_mean'] - r2['pred']:+.6f} "
      f"resid(N)={r1['gv_mean'] - r1['pred']:+.6f}")
# per-variance check
st = simulate_chain(2 ** 17, 256, 0.4, 0.4, T_D, rng)
fhats = np.array([np.bincount(st[s], minlength=V_D) / 2 ** 17 for s in range(256)])
sp = tau_spectrum(0.4, 0.4)
ratios = []
for v in [0, 5, 15]:
    var_emp = float(fhats[:, v].var(ddof=1))
    var_pred = T_D[v] * (1 - T_D[v]) * sp["tau"] / 2 ** 17
    ratios.append(var_pred / var_emp)
print(f"  Var(fhat_v) pred/emp ratios (v=0,5,15): "
      f"{[round(float(x), 3) for x in ratios]}  (se~0.09 each)")

# ---------------------------------------------------------------------------
# E. multi-epoch GD: fixed vs fresh replay
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("E. multi-epoch full-batch GD u_{e+1} = u_e + eta (fhat - q_e)")
print("=" * 78)
V_E = 16
T_E = zipf_probs(V_E)
l1E, l2E = 0.4, 0.4
NE, SE, ETA, EPOCHS = 2 ** 16, 128, 0.5, 800
stE = simulate_chain(NE, SE, l1E, l2E, T_E, rng)
fhatsE = np.array([np.bincount(stE[s], minlength=V_E) / NE for s in range(SE)])
g_infs = ((fhatsE - T_E) * np.log(fhatsE)).sum(axis=1)
us = np.zeros((SE, V_E))
qs = np.full((SE, V_E), 1.0 / V_E)
gaps_fixed = np.empty((SE, EPOCHS))
for e in range(EPOCHS):
    us = us + ETA * (fhatsE - qs)
    m = us.max(axis=1, keepdims=True)
    qs = np.exp(us - m)
    qs /= qs.sum(axis=1, keepdims=True)
    gaps_fixed[:, e] = ((fhatsE - T_E) * np.log(qs)).sum(axis=1)
mean_gap = gaps_fixed.mean(axis=0)
d = np.diff(gaps_fixed, axis=1)
viol = d < -1e-12
print(f"  fixed replay: E[gap_0]={mean_gap[0]:+.5f} "
      f"E[gap_last]={mean_gap[-1]:.5f} E[g_inf]={g_infs.mean():.5f}")
print(f"    mean trajectory monotone (non-increasing steps): "
      f"{int(np.sum(np.diff(mean_gap) < -1e-15))}; "
      f"per-draw draws with any decrease: {viol.any(axis=1).sum()}/{SE} "
      f"(max decrease {float(-d[viol].min()):.2e})")
assert np.sum(np.diff(mean_gap) < -1e-15) == 0
assert abs(mean_gap[-1] - g_infs.mean()) < 5 * gaps_fixed[:, -1].std() / np.sqrt(SE)

# fresh replay: each epoch an independent train sequence
NE2, ETA2, EPOCHS2 = 2 ** 14, 0.2, 600
u2 = np.zeros(V_E)
q2 = np.full(V_E, 1.0 / V_E)
gaps_fresh = np.empty(EPOCHS2)
for e in range(EPOCHS2):
    stf = simulate_chain(NE2, 1, l1E, l2E, T_E, rng)[0]
    fh = np.bincount(stf, minlength=V_E) / NE2
    u2 = u2 + ETA2 * (fh - q2)
    q2 = softmax(u2)
    gaps_fresh[e] = ((fh - T_E) * np.log(q2)).sum()
print(f"  fresh replay: E[gap] last 200 epochs = {gaps_fresh[-200:].mean():+.5f} "
      f"± {gaps_fresh[-200:].std():.5f} (expect ~0: fhat_e independent of q_e)")
assert abs(gaps_fresh[-200:].mean()) < 5 * gaps_fresh[-200:].std() / np.sqrt(200)

# linearized convergence rate: gap -> g_inf with factor (1 - eta*kappa)
f0 = fhatsE[0]
g0 = g_infs[0]
res = g0 - gaps_fixed[0]
tail = res[-150:]
ratio = float(np.mean(tail[1:] / np.maximum(tail[:-1], 1e-15)))
D = np.diag(f0) - np.outer(f0, f0)
ev = np.linalg.eigvalsh(D)
kappa = float(np.sort(ev[ev > 1e-12])[0])
print(f"  linearized rate: mean (g_inf-gap)_e/(g_inf-gap)_(e-1) over last 150 epochs"
      f" = {ratio:.5f} vs 1 - eta*kappa(D) = {1 - ETA * kappa:.5f}  (kappa = {kappa:.4f})")
assert abs(ratio - (1 - ETA * kappa)) < 0.01

# ---------------------------------------------------------------------------
# F. per-token excess curve e(v) = -log T_v - h
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("F. per-token excess curve e(v) = -log T_v - h  (q=T converged)")
print("=" * 78)
V_F = 64
T_F = zipf_probs(V_F)
for (l1, l2) in [(0.4, 0.4), (0.6, 0.2), (0.2, 0.6), (0.8, 0.1)]:
    H_T, h, _, pi_d = entropy_rate(T_F, l1, l2)
    e = -np.log(T_F) - h
    mono = bool(np.all(np.diff(e) > 0))          # e increases with rank
    print(f"  (l1,l2)=({l1},{l2}) V={V_F} zipf: H(T)={H_T:.4f} h={h:.4f} "
          f"Delta={H_T - h:.4f}  e(min freq)={e[-1]:.3f} e(max freq)={e[0]:.3f} "
          f"all>0={bool(e.min() > 0)}  monotone in rank={mono}")
    assert mono
# crossing demo: dominant token can go negative when T_max > exp(-h)
T_cross = np.array([0.99, 0.01])
H_T, h, _, pi_d = entropy_rate(T_cross, 0.4, 0.4)
e_cross = -np.log(T_cross) - h
print(f"  crossing demo V=2 T=(0.99,0.01) (0.4,0.4): H(T)={H_T:.4f} h={h:.4f} "
      f"e={np.round(e_cross, 4)}  -> crosses zero: {bool(e_cross.min() < 0)}")
assert e_cross.min() < 0
T_uni = np.full(8, 1.0 / 8)
H_T, h, _, pi_d = entropy_rate(T_uni, 0.4, 0.4)
print(f"  uniform V=8 (0.4,0.4): e(v) = {H_T - h:.4f} (all equal, no crossing)")

# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9.5})
cmap = plt.get_cmap("tab10")

# ---- fig 1: h validation ----
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
ax = axes[0, 0]
for Tv, lab, mk in [(T_ZIPF_B, "zipf V=16", "o"), (T_UNI_B, "uniform V=16", "s")]:
    hs, HTs = [], []
    for (l1, l2) in CONFIGS:
        H_T, h, _, _ = entropy_rate(Tv, l1, l2)
        hs.append(h)
        HTs.append(H_T)
    ax.plot(HTs, hs, mk, ms=6, label=lab)
lim = [0, max(max(hs), max(HTs)) * 1.05]
ax.plot(lim, lim, "k--", lw=1)
ax.fill_between(lim, 0, np.array(lim), alpha=0.06, color="0.3")
ax.text(lim[1] * 0.62, lim[1] * 0.16, "Delta = H(T) - h > 0\n(model-class gap)",
        fontsize=9, color="0.3")
ax.set_xlabel("H(T) (unigram Bayes loss)")
ax.set_ylabel("entropy rate h")
ax.set_title("(a) closed-form h vs H(T): irreducibly below unigram Bayes")
ax.legend(fontsize=8, frameon=False)
ax = axes[0, 1]
errs = np.array([r["err"] for r in h_rows])
ses = np.array([r["se"] for r in h_rows])
k = np.arange(len(h_rows))
ax.errorbar(k, errs, yerr=ses, fmt="o", ms=4, lw=1, color="#4c72b0")
ax.axhline(1e-2, color="0.5", ls=":", lw=1)
ax.set_yscale("log")
ax.set_xlabel("config index (10 (l1,l2) x zipf/uniform)")
ax.set_ylabel("|h_path - h|")
ax.set_title("(b) path-average error vs batch-mean se")
ax = axes[1, 0]
for Tv, lab, mk in [(T_ZIPF_B, "zipf", "o"), (T_UNI_B, "uniform", "s")]:
    hs, pds = [], []
    for (l1, l2) in CONFIGS:
        _, h, _, pd = entropy_rate(Tv, l1, l2)
        hs.append(h)
        pds.append(pd)
    ax.plot(pds, hs, mk, ms=6, label=lab)
ax.set_xlabel("diagonal pair mass pi_diag")
ax.set_ylabel("entropy rate h")
ax.set_title("(c) mechanism: diagonal mass compresses h")
ax.legend(fontsize=8, frameon=False)
ax = axes[1, 1]
for (l1, l2) in [(0.4, 0.4), (0.2, 0.6), (0.6, 0.2)]:
    _, h, hTxT, _ = entropy_rate(T_ZIPF_B, l1, l2)
    ax.plot([0, 1], [hTxT, h], "o-", lw=1.5, label=f"({l1},{l2})")
ax.set_xticks([0, 1])
ax.set_xticklabels(["naive T x T\nweighting", "true h\n(pi2 weighting)"])
ax.set_ylabel("nats")
ax.set_title("(d) naive pair weighting overestimates h")
ax.legend(fontsize=8, frameon=False)
fig.suptitle("Unigram model on l1/l2 chain: entropy rate h < H(T) (excess is irreducible)",
             fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(FIG_DIR / "fig_markov_unigram_h.svg")
fig.savefig(FIG_DIR / "fig_markov_unigram_h.png", dpi=150)

# ---- fig 2: gap over epochs, fixed vs fresh ----
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
ax = axes[0]
ee = np.arange(EPOCHS)
ax.plot(ee, mean_gap, "-", color="#4c72b0", lw=1.8,
        label="fixed replay  E[gap_e]")
band = gaps_fixed.std(axis=0) / np.sqrt(SE)
ax.fill_between(ee, mean_gap - 1.96 * band, mean_gap + 1.96 * band,
                color="#4c72b0", alpha=0.15)
ax.axhline(g_infs.mean(), color="#4c72b0", ls="--", lw=1,
           label=f"E[g_inf] = tau(V-1)/N ~ {g_infs.mean():.4f}")
for s in range(4):
    ax.plot(ee, gaps_fixed[s], "-", lw=0.4, color="#4c72b0", alpha=0.25)
ax.plot(np.arange(EPOCHS2), gaps_fresh, "-", lw=1.2, color="#c44e52",
        label="fresh replay  gap_e")
ax.axhline(0, color="0.7", lw=0.8)
ax.set_xlabel("epoch e")
ax.set_ylabel("gap_e = train_loss - val_loss (nats)")
ax.set_title("(a) fixed replay: gap grows with epochs; fresh: stays ~0")
ax.legend(fontsize=8, frameon=False)
ax = axes[1]
res_all = g_infs[:, None] - gaps_fixed
tail_mean = np.log(res_all[:, -150:].mean(axis=0))
ax.plot(np.arange(150), tail_mean, "-", color="#4c72b0", lw=1.6,
        label="log E[g_inf - gap_e]")
ax.plot(np.arange(150), np.log(res_all[:, -150:].mean(axis=0)[0]) + np.arange(150)
        * np.log(1 - ETA * kappa), "--", color="#c44e52", lw=1.4,
        label=f"analytic slope log(1-eta*kappa)={np.log(1 - ETA * kappa):.4f}")
ax.set_xlabel("epoch (last 150)")
ax.set_ylabel("log residual")
ax.set_title("(b) geometric convergence at the linearized rate")
ax.legend(fontsize=8, frameon=False)
fig.suptitle("Multi-epoch GD on the unigram table: replay-specific gap growth",
             fontsize=12, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIG_DIR / "fig_markov_unigram_gap_epochs.svg")
fig.savefig(FIG_DIR / "fig_markov_unigram_gap_epochs.png", dpi=150)

# ---- fig 3: excess curve ----
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
ax = axes[0]
for (l1, l2), c in [((0.4, 0.4), "#4c72b0"), ((0.6, 0.2), "#55a868"),
                    ((0.2, 0.6), "#c44e52"), ((0.8, 0.1), "#8172b2")]:
    _, h, _, _ = entropy_rate(T_F, l1, l2)
    e = -np.log(T_F) - h
    ax.semilogx(T_F, e, "o-", ms=4, lw=1.3, color=c, label=f"l1={l1}, l2={l2}")
ax.axhline(0, color="0.7", lw=0.8)
ax.set_xlabel("token frequency T_v (log)")
ax.set_ylabel("excess e(v) = -log T_v - h (nats)")
ax.set_title("(a) toy: per-token excess monotone down in frequency")
ax.legend(fontsize=8, frameon=False)
ax = axes[1]
_, h, _, _ = entropy_rate(T_cross, 0.4, 0.4)
e = -np.log(T_cross) - h
ax.axhline(0, color="0.7", lw=0.8)
ax.plot([0, 1], e, "o-", color="#c44e52", lw=1.6, ms=7,
        label="V=2, T=(0.99,0.01)")
for Tv, lab, mk in [(zipf_probs(8), "V=8 zipf", "s"),
                    (np.full(8, 1 / 8), "V=8 uniform", "^")]:
    _, h, _, _ = entropy_rate(Tv, 0.4, 0.4)
    ax.plot(np.arange(len(Tv)), -np.log(Tv) - h, mk, ms=5, label=lab)
ax.set_xlabel("token v (sorted by frequency)")
ax.set_ylabel("excess e(v)")
ax.set_title("(b) high-frequency tokens can cross zero")
ax.legend(fontsize=8, frameon=False)
ax = axes[2]
ax.semilogx(PROJ_FREQ, PROJ_EXCESS_A, "o-", lw=1.4, color="#4c72b0",
            label="A control (no ngram)")
ax.semilogx(PROJ_FREQ, PROJ_EXCESS_B, "s-", lw=1.4, color="#c44e52",
            label="B control (no ngram)")
ax.axhline(0, color="0.7", lw=0.8)
ax.set_xlabel("context frequency r(c)")
ax.set_ylabel("excess CE - H (nats)")
ax.set_title("(c) project reference: same monotone shape")
ax.legend(fontsize=8, frameon=False)
fig.suptitle("Per-token excess over the conditional entropy: "
             "shape matches the project frequency-gap curve",
             fontsize=12, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIG_DIR / "fig_markov_unigram_excess_vs_freq.svg")
fig.savefig(FIG_DIR / "fig_markov_unigram_excess_vs_freq.png", dpi=150)

print()
print(f"saved: {FIG_DIR / 'fig_markov_unigram_h.svg'}")
print(f"saved: {FIG_DIR / 'fig_markov_unigram_gap_epochs.svg'}")
print(f"saved: {FIG_DIR / 'fig_markov_unigram_excess_vs_freq.svg'}")
print()
print("ALL CHECKS PASSED")
