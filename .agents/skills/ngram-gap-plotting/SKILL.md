---
name: ngram-gap-plotting
description: Create, update, or review evidence-traceable plots for ngram-gap-lab runs. Use for loss/gap, frequency, table-occupancy, or scaling figures and before embedding a research plot in the report or blog; do not use to manufacture a figure from prose or historical numbers.
---

# Ngram Gap Plotting

Read `agents.md` §0 P4/P7 and `docs/plot_scripts/README.md` in full before
modifying a plotting script or generated figure. Read the target run's
`summary.json` and the relevant JSONL inputs before interpreting it.

## Select valid evidence

Use only the run namespace and `_fixed` artifact status allowed by
`agents.md`. Confirm the run's config, final step, seed, table architecture,
and measurement semantics match the requested comparison. Keep historical,
deprecated, incomplete, and proxy artifacts visibly separated from canonical
ones.

The primary series is:

```text
online gap = fixed validation loss - online current-training-batch loss
```

Do not substitute `fixed_train_loss.jsonl` for it. Do not treat `novel`
frequency contexts as having a train/validation gap: they have no train-token
loss counterpart.

## Produce a traceable figure

- Read numerical inputs from JSONL or another recorded artifact; never hand
  enter result numbers into HTML/SVG.
- Put source code in `docs/plot_scripts/` and generated files in the relevant
  `docs/figs/<line>/` directory. Keep a script and its generated figure
  together.
- Use a clear caption or adjacent text that names run IDs, step/endpoint,
  seed count, metric definition, and any changed experimental coordinate.
- Use loss/gap curves only for dense-monitoring runs. Label final-only/sparse
  output as an endpoint result rather than a curve.
- For frequency plots, retain token fractions separately from mean losses and
  contributions; use the prescribed log-axis filtering and exclude undefined
  or non-positive values from log-gap displays.

## Verify before handoff

Run the generator from the repository root, inspect its output, and check
paths and HTML/SVG references. Verify that legends/axes match the data, no
generated figure silently falls back to an unverified source, and `git diff
--check` passes. For blog embeds, check that iframes are fully visible without
internal scrolling. Do not alter the authoritative blog or push it without
separate authorization.
