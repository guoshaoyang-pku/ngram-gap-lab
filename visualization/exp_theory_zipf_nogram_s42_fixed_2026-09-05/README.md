# Theory-Zipf toy: no-ngram control

This folder contains the completed no-ngram control run
`theory_zipf_iid_nogram_s42_fixed_20260905` (seed 42, 1000 steps).  The data
and measurements were copied from the corresponding fixed run on `h200`.

## Source artifacts

- `train_log.jsonl`: online train loss, fixed validation loss, and global gap.
- `table_norm.jsonl`: one record per logged step; it has no table-RMS fields
  because all n-gram tables are disabled.
- `freq_bin_loss.jsonl`: frequency-bucket token counts and mean losses.
- `exact_freq_loss.jsonl`: exact train hit-count statistics and shared-context
  gaps for the diagnostic branches.
- `summary.json`: recorded configuration and endpoint metrics.
- `train.log`: raw server console output.

The global metric is

$$\mathrm{gap}=\mathrm{fixed\ validation\ loss}-\mathrm{current\ batch\ online\ training\ loss}.$$

At step 1000, the recorded train loss is `4.0080485344`, validation loss is
`3.9960638285`, and online gap is `-0.0119847059`.  The model has
`64,568,832` parameters; the summary records `enable_unigram_ve = false`,
`enable_bigram_ve = false`, `enable_trigram_ve = false`, and both clean-table
sizes equal to zero.

## Reproduce figures

From this directory:

```bash
python3 plot_theory_zipf.py
```

The script generates:

- `loss_gap_curve.png` — train/validation loss and global gap;
- `gap_table_rms_curve.png` — global gap only, explicitly noting that no
  n-gram table RMS exists in this control;
- `gap_vs_frequency_bigram.png` and `gap_vs_frequency_trigram.png` — final
  shared-context gap against exact train hit count, with positive-gap points,
  geometric-bin means, and a descriptive weighted log-log fit.

Novel contexts (`f=0`) are not assigned a standard gap because there is no
train-side counterpart.  The frequency fit is descriptive for this single
seed and endpoint, not a universal-law claim.
