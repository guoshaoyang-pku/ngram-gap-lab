# Theory-Zipf toy: frequency diagnostics

This self-contained folder contains the completed reproducibility rerun
`theory_zipf_iid_ngram_on_freq_s42_fixed_rerun_20260905` (seed 42, 1000 steps),
including exact context-frequency measurements. Its files replace the previous
visualization inputs in this directory.

## Source artifacts

- `train_log.jsonl`: online train loss, fixed validation loss, and global gap.
- `table_norm.jsonl`: bigram/trigram table RMS values.
- `freq_bin_loss.jsonl`: standard frequency-bucket statistics.
- `exact_freq_loss.jsonl`: exact train hit-count statistics and shared-context gaps.
- `summary.json`: recorded configuration, metric semantics, final endpoint, and index hash.
- `train.log`: raw console output.

The global metric is

`gap = fixed validation loss - current-batch online training loss`.

The rerun endpoint is train loss `3.2352876663`, validation loss
`4.4797916412`, and online gap `+1.2445039749` at step 1000.

The frequency diagnostic is the shared-context quantity

`gap(f) = validation mean loss - training mean loss`

for contexts with exact train hit count `f`. Novel contexts (`f=0`) are not
assigned a gap because they have no train-side counterpart.

## Reproduce figures

From this directory:

```bash
python3 plot_theory_zipf.py
```

The script generates:

- `loss_gap_curve.png`
- `gap_table_rms_curve.png`
- `gap_vs_frequency_bigram.png`
- `gap_vs_frequency_trigram.png`

The frequency figures use the final recorded checkpoint, retain positive-gap
points with at least five shared contexts, and show geometric-bin means plus a
weighted log-log fit. The fit is descriptive for this single seed and endpoint;
it is not a claim of a universal power law.
