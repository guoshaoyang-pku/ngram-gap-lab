# Documentation artifacts

`frequency-gap-by-hit-count.html` is the only HTML report in this directory. It
combines the frozen historical figures with all nine RMSProp Stage 1 conditions
and all thirteen new Stage 2A conditions, plus all four Stage 3R order-control
runs in a strict 2×2 matrix: epoch-1 original/random order × later-epoch
shuffle/no-shuffle. Each row is a checkpoint fork with data seed 42 and an
independent order seed 101. All current stages include complete interactive
online-loss and frequency-bucket curve explorers. Stage 2A also
contains the epoch-2 online gap versus LR curve and two fixed-probe
before/after gap-contribution versus LR/frequency curves. Stage 3R adds a
four-condition switchable, unthinned online-loss comparison and a per-condition
global online-gap curve. Its condition explorer can show or hide replay-edge
dense samples; frequency legends are placed outside the data region.
The cumulative exact-frequency masking chapter validates and plots all 49
formal thresholds (`none`, 47 numeric cutoffs, and `all`) as three epoch-end
online-gap curves on a continuous numeric x-axis with log1p/linear views. A
same-setting `none` bridge run is validated separately and
reported as an environment audit rather than included as a 50th sweep point.
The chapter can switch between three absolute epoch-end curves and two
epoch-over-epoch-1 increment curves; an integer upper-bound input (default 210)
filters the displayed measured thresholds without converting the x-axis back
to categorical spacing.

Regenerate it from the repository root after syncing `data/runs/`:

```powershell
python docs/generate_report.py `
  --runs-root data/runs `
  --historical-data docs/data/historical-figures.json `
  --out docs/frequency-gap-by-hit-count.html
```

The command validates every required run and writes the report atomically. The
historical interactive data is frozen in `docs/data/historical-figures.json`;
publication SVGs can be regenerated with `docs/plot_scripts/gen_all_figures.py`.
Cluster launchers intentionally produce run data only and do not create reports.
