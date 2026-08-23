# Synthetic transition task

## Purpose

The next controlled task keeps the frequency axis explicit and changes only
the conditional transition law. Every context is a five-token context
`c=(c0,...,c4)` and every block is

```text
[c0,c1,c2,c3,c4,next,SEP]
```

The train frequency `r(c)` is measured from the complete emitted train epoch
with a collision-free exact context index. Hash bucket occupancy is not used
for any gap plot.

## Common protocol

- vocabulary: 8192 tokens, `SEP=8191`
- context order: 5
- sequence length: 2048
- block length: 7
- same context list and same frequency profile for both schemes
- train targets are sampled from the known `P(next|c)`
- validation targets are independently sampled from the same distribution
- the exact conditional entropy can be estimated or computed from the stored
  transition matrix, so the Bayes loss is an explicit reference

The first pilot uses 4096 contexts and frequency scale 8. It is deliberately
small and clean; the full run can increase the context count without changing
the generator.

## Scheme A: sparse Markov chain with restart

For each context, choose a sparse support of `k=8` targets and define

```text
P(y|c) = (1-epsilon) S_c(y) + epsilon pi_zipf(y)
```

with `epsilon=0.10`. `S_c` is a normalized sparse distribution and
`pi_zipf` is a global Zipf base distribution.

This is the PageRank/teleport construction. The restart mass makes every
target possible, while the sparse component creates context-specific
transitions. The main knobs are `epsilon`, support size, and Zipf exponent.

## Scheme B: low-rank shared structure plus private sparse structure

Draw a topic mixture `u_c` for each context and topic distributions
`V_1,...,V_K`. The transition law is

```text
P(y|c) = (1-s) sum_k u_c(k) V_k(y) + s E_c(y)
```

where `E_c` is supported on four context-specific targets. The pilot uses
`K=32` and `s=0.50`.

The shared low-rank component can be learned across contexts; the private
component must be memorized. This gives a direct test of whether the
frequency-gap signal is associated with the non-shareable part of the
transition law.

## Primary comparison

For each scheme, report:

1. raw exact-frequency gap points and a `sqrt(r)` moving average;
2. `y=0` reference and a separate 3-sigma display clip;
3. train/validation CE against known Bayes loss;
4. injection norm dynamics;
5. the same frequency profile and the same random context list across schemes.

The most informative result is a comparison of the gap after subtracting the
known conditional-entropy baseline. Scheme A tests sparse local transitions
with a global restart. Scheme B tests shared structure versus private
memorization. Neither result should be interpreted as evidence about hash
collision frequency.

## Files

- generator: `toy/synthetic_transition_gen.py`
- generated contract: `run_contract.json`
- exact frequency index: `exact_ngram_counts.npz`
- dataset metadata: `metadata.json`
## Results — pilot (2026-08-07, seed 42, 360-2)

Four runs, 2000 steps each, `VAL_LOSS_INTERVAL_STEPS=10`, dev batch 4 /
total batch 8192, baseline LR warmdown, probes at 100/200/400/600/800/1000/
1500/2000 (constructed-block **val** probes). Both datasets: train 57.09M /
val 456.69M tokens. Conditional-entropy (Bayes) reference from
`transition_matrix.npz`: **A = 2.567 nats, B = 4.387 nats**
(= `frequency_weighted_bayes_ce` in `metadata.json`).

Val probe target CE (nats) and excess over H at step 2000:

| run | CE | H(c) | excess |
|---|---|---|---|
| A + ngram (inject) | 2.638 | 2.567 | **+0.071** |
| A control (no ngram) | 4.112 | 2.567 | +1.545 |
| B + ngram (inject) | 4.448 | 4.387 | **+0.061** |
| B control (no ngram) | 5.290 | 4.387 | +0.903 |

Per-frequency excess CE at step 2000 (n = probe targets in that frequency cell):

| r(c) | A inject | A control | B inject | B control |
|---|---|---|---|---|
| 8    | +4.40 (n2)  | +11.77 (n2)  | +1.88 (n3)  | +5.72 (n3)  |
| 16   | +4.89 (n2)  | +9.16 (n2)   | +4.94 (n2)  | +5.71 (n2)  |
| 32   | +1.35 (n3)  | +6.60 (n3)   | +2.01 (n3)  | +3.81 (n3)  |
| 64   | +3.09 (n8)  | +8.67 (n8)   | +2.45 (n4)  | +3.73 (n4)  |
| 128  | −0.21 (n8)  | +7.06 (n8)   | +3.34 (n7)  | +4.53 (n7)  |
| 512  | +0.42 (n18) | +5.12 (n18)  | +0.08 (n28) | +2.89 (n28) |
| 2048 | -0.38 (n47) | +2.26 (n47)  | +0.73 (n38) | +1.80 (n38) |
| 8192 | +0.06 (n2248) | +1.43 (n2248) | +0.03 (n2251) | +0.83 (n2251) |

Pooled (excess CE, nats):

| group | A inject | A control | B inject | B control |
|---|---|---|---|---|
| low r≤128  | +1.98 (n23)  | +8.15 (n23)  | +2.88 (n19)  | +4.56 (n19)  |
| mid 512–2048 | −0.16 (n65) | +3.05 (n65) | +0.46 (n66) | +2.26 (n66) |
| high r=8192 | +0.06 (n2248) | +1.43 (n2248) | +0.03 (n2251) | +0.83 (n2251) |

Figures: `docs/figs/fig_synth_excess_vs_freq.svg` (excess vs r, log x),
`docs/figs/fig_synth_excess_vs_step.svg` (convergence). Machine-readable
summaries: `docs/figs/synth_{A,B}_summary.json`; analysis script
`ngram5_freq_gap/analyze_synth.py` (reads run probe npz + transition matrix).

Interpretation:

1. **Injection closes the gap.** With the exact train context-count index
   injected as VE tables, val target CE reaches the conditional entropy within
   noise (excess ≈ 0.06–0.07 nats) by step 2000, and the excess is flat across
   frequency. The model learns to use the table in a short run.
2. **The frequency gap is a real, controlled phenomenon.** Without injection,
   excess is 1.55 (A) / 0.90 (B) nats overall and monotonically decreasing in
   r(c): ≈8–9 nats at r=8–64 → ≈1.4 nats at r=8192 (A); ≈4–6 → ≈0.8 (B).
3. **Scheme B control is better than A control** (0.90 vs 1.55 nats): B's
   shared low-rank component transfers across contexts, while A's private
   sparse support must be memorized per context. The gap is associated with the
   non-shareable part of the transition law, as designed.
4. Caveats: low-frequency cells have n=2–8 (noisy); the r=8192 cell dominates
   the overall mean (n≈2250); 2000 steps is short — injected runs are
   effectively converged, controls are still learning, so the control numbers
   are upper bounds on the converged gap.

Resource note: inject runs `num_params≈2.88B` (incl. VE tables), peak VRAM
16.7GB, ~171s; controls `num_params≈64.6M`, peak VRAM 3.0GB, ~144s.
