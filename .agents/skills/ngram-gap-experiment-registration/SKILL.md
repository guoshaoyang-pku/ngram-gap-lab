---
name: ngram-gap-experiment-registration
description: Register, launch-plan, backfill, or hand off one reproducible ngram-gap-lab experiment. Use when assigning an experiment to a collaborator, creating a run_id, or turning a completed run into an evidence-traceable record; do not use for an unscoped research brainstorm.
---

# Ngram Gap Experiment Registration

Read `agents.md` §0–1, `docs/experiment-lines.md`, and the relevant portion of
`docs/experiment-log.md` before writing. Check `docs/claims-ledger.md` before
describing any result as a conclusion.

## Register before compute

One experiment has one `run_id`, one result directory, one registry row, and
one detailed section. Add a `planned` row and section before reserving a GPU.
The section must identify:

- scientific question and falsifiable expected comparison;
- owner, date, target GPU/cluster, run ID, seed(s), and planned endpoint;
- complete baseline identity plus the one changed coordinate; and
- exact command, data/frequency-index paths, expected artifact directory, and
  acceptance checks.

For authoritative main-line runs, use
`data/runs_fixed/<run_id>_fixed/`. Scaling runs belong in the explicitly named
`data/runs_scaling/` namespace. Never overwrite an existing completed run.

## Handover packet

Give a collaborator a small, executable packet rather than an informal setting
description:

1. link to `agents.md` §1 and the relevant experiment-log section;
2. command with all baseline and changed flags explicit, including clean-table
   capacities and `--table_betas 0.0,0.99`;
3. data/train/validation/frequency-index paths and proof that train and val do
   not overlap;
4. `git rev-parse HEAD` plus hashes of `code/train.py`, `code/ngram_freq.py`,
   and the launcher before multi-machine work;
5. output directory, required JSONL files, final step/metric, and stop rule.

Do not ask a collaborator to interpret an old `runs/` result, an un-suffixed
directory, or a deprecated setting as a new baseline.

## Run lifecycle

Update the registry through `planned → running → done` (or `stalled`). At
completion, verify `summary.json`, `train_log.jsonl`, the requested diagnostic
files, the recorded config, final step, seed, and output path. Backfill the
observed train loss, fixed validation loss, and online gap with their step.
Record failures honestly; a missing artifact is not a negative result.

Do not push, change branches/remotes, transfer more than 1 GB, or delete prior
artifacts without the required user authorization. Make a separate run when
code or measurement semantics change.

## Ready-for-plot handoff

Hand a completed run to `$ngram-gap-plotting` only with the run ID, artifact
path, intended comparison, metric definition, seed count, and the evidence
status that the figure may support.
