# Cluster train.py patch guide (S5)

The 5-gram frequency-gap experiment needs **3 small changes** to the cluster's
`/data3/guoshaoyang/ngram-gap-exp/train.py` so the existing n-gram injection
pipeline can (a) load the 5-gram block dataset and (b) decompose the gap by
5-gram frequency.  The local `ngram5_freq_gap/trainer.py` already implements
the full pipeline independently (for CPU smoke tests), so on the cluster you
can either run `ngram5_freq_gap/trainer.py` directly (no train.py patch
needed) OR patch train.py to run the experiment through the canonical cluster
entry point.

## Option A: run via ngram5_freq_gap/trainer.py (no train.py patch)

`run_on_cluster.sh` does this by default.  The trainer imports `model.py`
which dynamically loads `NanoGPTOriginal` from the cluster `train.py`, so the
n-gram injection tables are live.  The trainer uses `lib.make_dataloader(
data_mode="ngram5_blocks")` (already patched in the synced `lib.py`) to load
the block dataset, and `FivegramIndex` (local to trainer.py) for the frequency
decomposition.  **No train.py patch required.**

This is the recommended path for the first runs.

## Option B: patch train.py to add fivegram branch + ngram5 data entry

If you want the experiment to run through the canonical cluster `train.py`
(to reuse its full observable/theory/probe instrumentation), apply these 3
changes:

### Change 1: add `ngram5_blocks` to the data-mode allowlist

In `lib.py` (already synced), `TRAIN_DATA_MODES` already includes
`"ngram5_blocks"` and `_ngram5_block_dataloader` is implemented.  No change
needed in train.py for data loading — `make_dataloader(..., data_mode=
"ngram5_blocks")` dispatches correctly when `NGRAM5_DATA_DIR` is set.

### Change 2: add `fivegram` branch to GlobalNgramFrequencyIndex

In `train.py`, class `GlobalNgramFrequencyIndex` (around line 3778 in the
7-30 snapshot):

```python
class GlobalNgramFrequencyIndex:
    branches = ("bigram", "trigram", "fivegram")  # <-- add "fivegram"
```

And extend `_keys` / `lookup` / `bucket_labels` to handle the fivegram
branch.  The fivegram key is the **bucket id** (hash5 % bucket_count), not a
base-V encoding (8192^5 overflows int64).  The lookup must compute
`hash5(c0..c4) % bucket_count` at query time using the same polynomial hash
as `ngram5_freq_gap/data_gen.py` (Mersenne prime 2^61-1).

Reference implementation: `ngram5_freq_gap/trainer.py` `_bucket_id_tensor` +
`FivegramIndex.lookup_frequency`.

### Change 3: load fivegram_counts.npz from NGRAM5_DATA_DIR

When `NGRAM5_DATA_DIR` is set, point `GLOBAL_FREQUENCY_DIR` at it so the
`GlobalNgramFrequencyIndex` finds `fivegram_counts.npz` + `metadata.json`.
Around the `GLOBAL_FREQUENCY_DIR` assignment (line ~3699):

```python
GLOBAL_FREQUENCY_DIR = Path(
    os.environ.get("NGRAM5_DATA_DIR", "").strip() or
    os.environ.get("NGRAM_GLOBAL_FREQUENCY_DIR", "").strip() or
    str(RUN_ARTIFACT_DIR / "ngram_global_frequency")
)
```

And in the decomposition call sites (`_frequency_bucket_values` ~line 7088,
`_frequency_mask_coverage_values` ~line 7124), extend the
`for branch in ("bigram", "trigram"):` loops to include `"fivegram"` and
build `previous3`/`previous4` for the 5-token context lookup.

### Change 4 (optional): skip tokenizer when data_mode=ngram5_blocks

If train.py hard-requires a tokenizer at init, guard the
`Tokenizer.from_directory()` call with:

```python
if TRAIN_DATA_MODE != "ngram5_blocks":
    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
else:
    tokenizer = None
    vocab_size = json.loads(Path(os.environ["NGRAM5_DATA_DIR"]).joinpath(
        "meta.json").read_text())["vocab"]
```

## Verifying the patch

After patching, run a 10-step sanity:

```bash
NGRAM5_DATA_DIR=/data3/guoshaoyang/ngram-gap-exp/ngram5_data/alpha0.0 \
TRAIN_DATA_MODE=ngram5_blocks MAX_TRAINING_STEPS=10 \
python -u train.py
```

The run should complete without errors and produce
`allgram_frequency_decomposition.jsonl` with `branch=fivegram` records.
