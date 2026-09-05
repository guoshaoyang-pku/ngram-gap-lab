#!/usr/bin/env python3
"""Generate the theory-first iid finite-Zipf toy with main-line shard geometry.

All scientific and storage parameters are versioned in this file.  The script
takes no command-line arguments.  Run it from any directory with

    python3 tasks/l7_theory_zipf/toy_theory_zipf_20260903.py

The only intentional data-distribution change relative to the repository's
main-line natural-language experiment is the iid finite-Zipf token law.  The
vocabulary, model-facing dimensions, sequence length, device-batch geometry,
train shard geometry, and validation shard geometry are aligned with the
original 1x experiment.  The complete non-overlapping validation pool is
materialized as shards 2..10 and 6542.  code/train.py's default fixed
validation evaluator still reads only four batches from the beginning of that
pool; the four-batch setting is an evaluation setting, not a reason to shorten
the validation files.

The generator deliberately has no n-gram-specific data mechanism.  It samples

    x_1, x_2, ..., x_N  iid~ P(x),
    P(x_(r)) = r**(-alpha) / sum_j j**(-alpha).

The model, not this script, later discovers bigrams and trigrams by sliding
over each packed 2049-token loader chunk.  Train and validation use the same
marginal with independent random streams, so P(y | c) = P(y) for every finite
context c.  Different train/validation shards mean independent samples; the
same token IDs can and should occur in both sets under the shared support.

Outputs in OUTPUT_DIR:
  shard_00001.bin                         complete train shard
  shard_00002.bin ... shard_00010.bin     complete validation shards
  shard_06542.bin                          validation tail shard (284 batches)
  marginal_oracle.npz                     theory curves
  metadata.json, meta.json                provenance and loader contract
  run_contract.json                        scientific invariants

The script never writes a transition matrix, context labels, blocks, padding,
a separator token, or context-specific continuation mappings.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Versioned experiment setting.  Change values here, not on a server command.
# ---------------------------------------------------------------------------

GENERATOR_DATE = "20260903"
SETTING_REVISION_DATE = "20260904"
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "inputs"
    / "theory_zipf_iid_mainline_aligned_20260904"
)

# Main-line model/data geometry from agents.md section 1.
VOCAB_SIZE = 8192
SUPPORT_SIZE = 8192
MODEL_N_LAYER = 8
MODEL_N_HEAD = 6
MODEL_N_EMBD = 768
SEQUENCE_LEN = 2048
DEVICE_BATCH_SIZE = 72

# Toy-specific distribution.  This is the scientific variable, not a hidden
# bigram/trigram construction.
ZIPF_ALPHA = 4.0 / 3.0

# Main-line fixed-replay seed contract.
TRAIN_SEED = 42
VALIDATION_SEED_BASE = TRAIN_SEED + 1_000_003

# The original 1x train shard has 337 device batches.  Each loader sample
# needs sequence_len + 1 raw tokens to make an input/target pair.  The full
# non-overlapping validation source pool has nine complete shards (2..10) and
# a 284-batch tail shard (6542).  train.py captures only four fixed validation
# batches at startup; that is an evaluation-cache setting, not the amount of
# validation data written here.
TOKENS_PER_LOADER_CHUNK = SEQUENCE_LEN + 1
TRAIN_DEVICE_BATCHES_PER_EPOCH = 337
# Number of batches consumed by code/train.py's default fixed validation eval.
FIXED_VALIDATION_DEVICE_BATCHES = 4
TOKENS_PER_DEVICE_BATCH = DEVICE_BATCH_SIZE * SEQUENCE_LEN
VAL_INTERVAL_STEPS = 10
FULL_SHARD_TOKENS = (
    TRAIN_DEVICE_BATCHES_PER_EPOCH
    * DEVICE_BATCH_SIZE
    * TOKENS_PER_LOADER_CHUNK
)
# Loss counts exclude the final target-shaping token in every loader chunk.
TRAIN_LOSS_TOKENS = (
    TRAIN_DEVICE_BATCHES_PER_EPOCH * DEVICE_BATCH_SIZE * SEQUENCE_LEN
)
ORIGINAL_VALIDATION_TAIL_DEVICE_BATCHES = 284
ORIGINAL_VALIDATION_POOL_SHARDS = tuple(range(2, 11)) + (6542,)
ORIGINAL_VALIDATION_FULL_SHARD_IDS = tuple(range(2, 11))
ORIGINAL_VALIDATION_FULL_SHARD_COUNT = len(ORIGINAL_VALIDATION_FULL_SHARD_IDS)
ORIGINAL_VALIDATION_TAIL_SHARD_TOKENS = (
    ORIGINAL_VALIDATION_TAIL_DEVICE_BATCHES
    * DEVICE_BATCH_SIZE
    * TOKENS_PER_LOADER_CHUNK
)

TRAIN_SHARD_SPECS = ((1, FULL_SHARD_TOKENS),)
VALIDATION_SHARD_SPECS = tuple(
    (shard_id, FULL_SHARD_TOKENS)
    for shard_id in ORIGINAL_VALIDATION_FULL_SHARD_IDS
) + ((6542, ORIGINAL_VALIDATION_TAIL_SHARD_TOKENS),)

TRAIN_TOKENS = sum(token_count for _, token_count in TRAIN_SHARD_SPECS)
VALIDATION_TOKENS = sum(
    token_count for _, token_count in VALIDATION_SHARD_SPECS
)
VALIDATION_TO_TRAIN_RATIO = VALIDATION_TOKENS / TRAIN_TOKENS
TRAIN_TO_VALIDATION_RATIO = TRAIN_TOKENS / VALIDATION_TOKENS
VALIDATION_LOSS_TOKENS = sum(
    (token_count // TOKENS_PER_LOADER_CHUNK) * SEQUENCE_LEN
    for _, token_count in VALIDATION_SHARD_SPECS
)
VALIDATION_DEVICE_BATCHES_FULL_POOL = sum(
    token_count // TOKENS_PER_LOADER_CHUNK
    for _, token_count in VALIDATION_SHARD_SPECS
) // DEVICE_BATCH_SIZE
FIXED_VALIDATION_RAW_TOKENS = (
    FIXED_VALIDATION_DEVICE_BATCHES
    * DEVICE_BATCH_SIZE
    * TOKENS_PER_LOADER_CHUNK
)
FIXED_VALIDATION_LOSS_TOKENS = (
    FIXED_VALIDATION_DEVICE_BATCHES * DEVICE_BATCH_SIZE * SEQUENCE_LEN
)
VALIDATION_TO_TRAIN_LOSS_RATIO = VALIDATION_LOSS_TOKENS / TRAIN_LOSS_TOKENS
TRAIN_TO_VALIDATION_LOSS_RATIO = TRAIN_LOSS_TOKENS / VALIDATION_LOSS_TOKENS
TRAIN_RAW_TO_VALIDATION_LOSS_RATIO = TRAIN_TOKENS / VALIDATION_LOSS_TOKENS

# Complete validation pool, now materialized rather than provenance-only.
ORIGINAL_VALIDATION_POOL_TOKENS = VALIDATION_TOKENS
ORIGINAL_VALIDATION_POOL_TO_TRAIN_RATIO = (
    ORIGINAL_VALIDATION_POOL_TOKENS / FULL_SHARD_TOKENS
)

# Operational settings are also versioned here so the launch command stays
# argument-free.  OVERWRITE remains false to protect existing generated data.
CHUNK_TOKENS = 1_000_000
OVERWRITE = False
ORACLE_FREQUENCIES = tuple(2**k for k in range(21))
UINT16_MAX = np.iinfo(np.uint16).max


def shard_name(shard_id: int) -> str:
    """Return the filename expected by code/train.py for one shard ID."""
    return f"shard_{shard_id:05d}.bin"


def sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Return the SHA-256 digest of a file without loading it all in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_setting() -> None:
    """Fail before writing if the versioned setting violates its contract."""
    if VOCAB_SIZE < 2 or VOCAB_SIZE > UINT16_MAX + 1:
        raise ValueError("VOCAB_SIZE must be in [2, 65536] for uint16 files")
    if SUPPORT_SIZE < 2 or SUPPORT_SIZE > VOCAB_SIZE:
        raise ValueError("SUPPORT_SIZE must satisfy 2 <= SUPPORT_SIZE <= VOCAB_SIZE")
    if not math.isfinite(ZIPF_ALPHA) or ZIPF_ALPHA <= 1.0:
        raise ValueError("ZIPF_ALPHA must be finite and greater than 1")
    if SEQUENCE_LEN <= 0 or DEVICE_BATCH_SIZE <= 0:
        raise ValueError("SEQUENCE_LEN and DEVICE_BATCH_SIZE must be positive")
    if TRAIN_DEVICE_BATCHES_PER_EPOCH <= 0 or FIXED_VALIDATION_DEVICE_BATCHES <= 0:
        raise ValueError("train and validation device-batch counts must be positive")
    if CHUNK_TOKENS <= 0:
        raise ValueError("CHUNK_TOKENS must be positive")
    if [shard_id for shard_id, _ in TRAIN_SHARD_SPECS] != [1]:
        raise ValueError("main-line alignment requires train shard 1")
    expected_val_ids = list(ORIGINAL_VALIDATION_FULL_SHARD_IDS) + [6542]
    actual_val_ids = [shard_id for shard_id, _ in VALIDATION_SHARD_SPECS]
    if actual_val_ids != expected_val_ids:
        raise ValueError("validation must materialize shards 2..10 and 6542")
    for shard_id, token_count in VALIDATION_SHARD_SPECS:
        expected_tokens = (
            ORIGINAL_VALIDATION_TAIL_SHARD_TOKENS
            if shard_id == 6542
            else FULL_SHARD_TOKENS
        )
        if token_count != expected_tokens:
            raise ValueError(
                f"validation shard {shard_id} must contain {expected_tokens} raw tokens"
            )
    all_specs = TRAIN_SHARD_SPECS + VALIDATION_SHARD_SPECS
    if any(token_count <= 0 for _, token_count in all_specs):
        raise ValueError("every shard must contain a positive number of tokens")
    if any(token_count % TOKENS_PER_LOADER_CHUNK for _, token_count in all_specs):
        raise ValueError("every shard must align to sequence_len + 1")


def zipf_probabilities() -> np.ndarray:
    """Return the normalized finite-Zipf marginal over token IDs 0..K-1."""
    ranks = np.arange(1, SUPPORT_SIZE + 1, dtype=np.float64)
    weights = ranks ** (-ZIPF_ALPHA)
    probabilities = weights / weights.sum()
    return probabilities.astype(np.float64, copy=False)


def sample_zipf(
    rng: np.random.Generator,
    cdf: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    """Draw ordinary token IDs by inverse-CDF sampling."""
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if sample_count == 0:
        return np.empty(0, dtype=np.uint16)
    uniforms = rng.random(sample_count)
    token_ids = np.searchsorted(cdf, uniforms, side="right")
    return token_ids.astype(np.uint16, copy=False)


def write_iid_shard(
    path: Path,
    rng: np.random.Generator,
    cdf: np.ndarray,
    token_count: int,
) -> int:
    """Write exactly token_count iid samples as little-endian uint16."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("wb") as handle:
        while written < token_count:
            amount = min(CHUNK_TOKENS, token_count - written)
            tokens = sample_zipf(rng, cdf, amount)
            handle.write(tokens.astype("<u2", copy=False).tobytes())
            written += amount
    return written


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return beta from y approximately proportional to x**(-beta)."""
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    mask = (
        np.isfinite(x_array)
        & np.isfinite(y_array)
        & (x_array > 0)
        & (y_array > 0)
    )
    if int(mask.sum()) < 3:
        return float("nan")
    slope, _intercept = np.polyfit(
        np.log(x_array[mask]), np.log(y_array[mask]), 1
    )
    return float(-slope)


def oracle_curves(
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the frequency grid, P0(f), and expected singleton mass."""
    frequencies = np.asarray(ORACLE_FREQUENCIES, dtype=np.int64)
    missing_mass = np.empty(len(frequencies), dtype=np.float64)
    singleton_mass = np.empty(len(frequencies), dtype=np.float64)
    for index, frequency in enumerate(frequencies):
        unseen = np.power(1.0 - probabilities, int(frequency))
        missing_mass[index] = float(np.sum(probabilities * unseen))
        singleton = np.power(1.0 - probabilities, int(frequency) - 1)
        singleton_mass[index] = float(np.sum(probabilities * singleton))
    return frequencies, missing_mass, singleton_mass


def write_json(path: Path, payload: dict) -> None:
    """Write stable, human-readable JSON with a trailing newline."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def shard_record(shard_id: int, token_count: int, role: str, seed: int) -> dict:
    """Build one metadata record for a generated train or validation shard."""
    chunks = token_count // TOKENS_PER_LOADER_CHUNK
    return {
        "id": int(shard_id),
        "file": shard_name(shard_id),
        "role": role,
        "seed": int(seed),
        "tokens": int(token_count),
        "raw_tokens": int(token_count),
        "loss_tokens": int(chunks * SEQUENCE_LEN),
        "bytes": int(token_count * np.dtype("<u2").itemsize),
        "loader_chunks": int(chunks),
        "device_batches": int(chunks // DEVICE_BATCH_SIZE),
    }


def generate() -> dict:
    """Generate aligned train/validation shards, oracle arrays, and metadata."""
    validate_setting()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_specs = TRAIN_SHARD_SPECS + VALIDATION_SHARD_SPECS
    generated_names = [shard_name(shard_id) for shard_id, _ in all_specs]
    generated_names += [
        "marginal_oracle.npz",
        "metadata.json",
        "meta.json",
        "run_contract.json",
    ]
    existing = [
        OUTPUT_DIR / name for name in generated_names if (OUTPUT_DIR / name).exists()
    ]
    if existing and not OVERWRITE:
        raise FileExistsError(
            "refusing to overwrite existing generated files; change OUTPUT_DIR "
            "or set OVERWRITE=True in the script: "
            + ", ".join(str(path) for path in existing)
        )
    if OVERWRITE:
        for path in existing:
            path.unlink()

    probabilities = zipf_probabilities()
    cdf = np.cumsum(probabilities, dtype=np.float64)
    cdf[-1] = 1.0

    shard_records: list[dict] = []
    train_rng = np.random.default_rng(TRAIN_SEED)
    for shard_id, token_count in TRAIN_SHARD_SPECS:
        path = OUTPUT_DIR / shard_name(shard_id)
        written = write_iid_shard(path, train_rng, cdf, token_count)
        shard_records.append(shard_record(shard_id, written, "train", TRAIN_SEED))

    for offset, (shard_id, token_count) in enumerate(VALIDATION_SHARD_SPECS):
        seed = VALIDATION_SEED_BASE + offset
        path = OUTPUT_DIR / shard_name(shard_id)
        validation_rng = np.random.default_rng(seed)
        written = write_iid_shard(path, validation_rng, cdf, token_count)
        shard_records.append(shard_record(shard_id, written, "validation", seed))

    frequencies, missing_mass, singleton_mass = oracle_curves(probabilities)
    asymptotic_beta = 1.0 - 1.0 / ZIPF_ALPHA
    finite_beta = loglog_slope(frequencies, missing_mass)
    entropy_nats = float(-np.sum(probabilities * np.log(probabilities)))
    np.savez(
        OUTPUT_DIR / "marginal_oracle.npz",
        frequencies=frequencies,
        p0_missing_mass=missing_mass,
        expected_singleton_mass=singleton_mass,
        probabilities=probabilities,
        entropy_nats=np.asarray(entropy_nats),
        zipf_alpha=np.asarray(ZIPF_ALPHA),
        asymptotic_beta=np.asarray(asymptotic_beta),
        finite_oracle_beta=np.asarray(finite_beta),
    )

    hashes = {
        record["file"]: sha256(OUTPUT_DIR / record["file"])
        for record in shard_records
    }
    metadata = {
        "schema_version": 3,
        "experiment": "theory_iid_zipf_mainline_aligned",
        "generator": Path(__file__).name,
        "generator_date": GENERATOR_DATE,
        "setting_revision_date": SETTING_REVISION_DATE,
        "output_dir": str(OUTPUT_DIR),
        "data_format": "flat_little_endian_uint16_token_stream",
        "toy_distribution": {
            "kind": "iid_finite_zipf",
            "support_size": SUPPORT_SIZE,
            "zipf_alpha": ZIPF_ALPHA,
            "train_validation_same_marginal": True,
            "train_validation_independent_rng": True,
        },
        "model_alignment": {
            "n_layer": MODEL_N_LAYER,
            "n_head": MODEL_N_HEAD,
            "n_embd": MODEL_N_EMBD,
            "vocab_size": VOCAB_SIZE,
            "sequence_len": SEQUENCE_LEN,
            "device_batch_size": DEVICE_BATCH_SIZE,
            "token_file_scalar_dtype": "uint16",
            "note": "the file stores token IDs; 768 is the model embedding width",
        },
        "split_alignment": {
            "train_shards": [shard_id for shard_id, _ in TRAIN_SHARD_SPECS],
            "validation_shards": [
                shard_id for shard_id, _ in VALIDATION_SHARD_SPECS
            ],
            "train_tokens": TRAIN_TOKENS,
            "validation_tokens": VALIDATION_TOKENS,
            "train_raw_tokens": TRAIN_TOKENS,
            "validation_raw_tokens": VALIDATION_TOKENS,
            "original_validation_pool_tokens": ORIGINAL_VALIDATION_POOL_TOKENS,
            "original_validation_pool_to_train_ratio": ORIGINAL_VALIDATION_POOL_TO_TRAIN_RATIO,
            "train_loss_tokens_per_epoch": TRAIN_LOSS_TOKENS,
            "validation_loss_tokens_full_pool": VALIDATION_LOSS_TOKENS,
            "fixed_validation_raw_tokens": FIXED_VALIDATION_RAW_TOKENS,
            "fixed_validation_loss_tokens_per_eval": FIXED_VALIDATION_LOSS_TOKENS,
            "validation_interval_steps": VAL_INTERVAL_STEPS,
            "train_raw_to_full_validation_loss_token_ratio": TRAIN_RAW_TO_VALIDATION_LOSS_RATIO,
            "validation_device_batches_full_pool": VALIDATION_DEVICE_BATCHES_FULL_POOL,
            "fixed_validation_device_batches": FIXED_VALIDATION_DEVICE_BATCHES,
            "original_validation_pool_shards": list(ORIGINAL_VALIDATION_POOL_SHARDS),
            "materialized_validation_is_minimal_fixed_val_subset": False,
            "materialized_validation_shards": [
                shard_id for shard_id, _ in VALIDATION_SHARD_SPECS
            ],
            "validation_to_train_token_ratio": VALIDATION_TO_TRAIN_RATIO,
            "train_to_validation_token_ratio": TRAIN_TO_VALIDATION_RATIO,
            "validation_to_train_loss_token_ratio": VALIDATION_TO_TRAIN_LOSS_RATIO,
            "train_to_validation_loss_token_ratio": TRAIN_TO_VALIDATION_LOSS_RATIO,
            "train_to_validation_ratio_text": f"1:{VALIDATION_TO_TRAIN_RATIO:.10f}",
            "validation_to_train_ratio_text": f"{VALIDATION_TO_TRAIN_RATIO:.10f}:1",
            "source": (
                "agents.md section 1; code/train.py val_batches=4; "
                "docs/experiment-log.md sections 8 and 19"
            ),
        },
        "shards": shard_records,
        "oracle": {
            "file": "marginal_oracle.npz",
            "frequencies": frequencies.tolist(),
            "asymptotic_beta": asymptotic_beta,
            "finite_oracle_beta": finite_beta,
            "entropy_nats": entropy_nats,
            "p0_definition": "sum_y p(y)*(1-p(y))**f",
            "singleton_definition": "sum_y p(y)*(1-p(y))**(f-1)",
        },
        "context_frequency_source": "derive_from_raw train shard 1",
        "contains_explicit_contexts": False,
        "contains_explicit_bigrams": False,
        "contains_explicit_trigrams": False,
        "contains_blocks": False,
        "contains_sep": False,
        "contains_padding": False,
        "contains_transition_matrix": False,
        "contains_context_specific_continuation_map": False,
        "sha256": hashes,
    }
    write_json(OUTPUT_DIR / "metadata.json", metadata)
    write_json(OUTPUT_DIR / "meta.json", metadata)

    contract = {
        "contract_version": 3,
        "experiment": metadata["experiment"],
        "generator": metadata["generator"],
        "versioned_setting": {
            "vocab_size": VOCAB_SIZE,
            "support_size": SUPPORT_SIZE,
            "model_n_layer": MODEL_N_LAYER,
            "model_n_head": MODEL_N_HEAD,
            "model_n_embd": MODEL_N_EMBD,
            "sequence_len": SEQUENCE_LEN,
            "device_batch_size": DEVICE_BATCH_SIZE,
            "train_device_batches_per_epoch": TRAIN_DEVICE_BATCHES_PER_EPOCH,
            "validation_device_batches_full_pool": VALIDATION_DEVICE_BATCHES_FULL_POOL,
            "fixed_validation_device_batches": FIXED_VALIDATION_DEVICE_BATCHES,
            "validation_interval_steps": VAL_INTERVAL_STEPS,
            "total_batch_tokens": TOKENS_PER_DEVICE_BATCH,
            "data_mode": "fixed",
            "zipf_alpha": ZIPF_ALPHA,
            "train_seed": TRAIN_SEED,
            "validation_seed_base": VALIDATION_SEED_BASE,
            "train_tokens": TRAIN_TOKENS,
            "validation_tokens": VALIDATION_TOKENS,
            "validation_to_train_token_ratio": VALIDATION_TO_TRAIN_RATIO,
            "train_to_validation_token_ratio": TRAIN_TO_VALIDATION_RATIO,
            "train_loss_tokens_per_epoch": TRAIN_LOSS_TOKENS,
            "validation_loss_tokens_full_pool": VALIDATION_LOSS_TOKENS,
            "fixed_validation_raw_tokens": FIXED_VALIDATION_RAW_TOKENS,
            "fixed_validation_loss_tokens_per_eval": FIXED_VALIDATION_LOSS_TOKENS,
            "validation_to_train_loss_token_ratio": VALIDATION_TO_TRAIN_LOSS_RATIO,
            "train_to_validation_loss_token_ratio": TRAIN_TO_VALIDATION_LOSS_RATIO,
            "train_raw_to_full_validation_loss_token_ratio": TRAIN_RAW_TO_VALIDATION_LOSS_RATIO,
            "original_validation_pool_tokens": ORIGINAL_VALIDATION_POOL_TOKENS,
            "original_validation_pool_to_train_ratio": ORIGINAL_VALIDATION_POOL_TO_TRAIN_RATIO,
        },
        "scientific_invariants": {
            "raw_stream_only": True,
            "token_process": "x_t iid~ finite_Zipf(alpha)",
            "same_train_validation_marginal": True,
            "independent_train_validation_rng": True,
            "train_validation_shards_nonoverlapping": True,
            "token_id_values_may_overlap_between_splits": True,
            "no_generator_defined_ngram_structure": True,
            "context_frequency_is_measured_after_generation": True,
            "contexts_derived_within_each_loader_chunk": True,
            "cross_chunk_contexts_not_counted": True,
            "same_shards_for_ngram_off_and_bigram_trigram_on": True,
            "model_architecture_unchanged": True,
            "mainline_vocab_sequence_embedding_aligned": True,
            "mainline_train_shard_geometry_aligned": True,
            "mainline_fixed_val_eval_geometry_aligned": True,
            "original_validation_pool_recorded_not_materialized": False,
            "complete_validation_pool_materialized": True,
        },
        "model_use": {
            "data_dir": str(OUTPUT_DIR),
            "train_shards": "1",
            "validation_shards": ",".join(
                str(sid) for sid in ORIGINAL_VALIDATION_POOL_SHARDS
            ),
            "materialized_validation_shards": ",".join(
                str(sid) for sid in ORIGINAL_VALIDATION_POOL_SHARDS
            ),
            "fixed_eval_validation_shards": "2",
            "validation_batches": FIXED_VALIDATION_DEVICE_BATCHES,
            "fixed_eval_source_prefix": {
                "shard": 2,
                "raw_tokens": FIXED_VALIDATION_RAW_TOKENS,
                "loss_tokens": FIXED_VALIDATION_LOSS_TOKENS,
                "batches": FIXED_VALIDATION_DEVICE_BATCHES,
            },
            "fixed_eval_reads_prefix_only": True,
            "original_validation_pool_shards": ",".join(
                str(sid) for sid in ORIGINAL_VALIDATION_POOL_SHARDS
            ),
            "original_validation_pool_tokens": ORIGINAL_VALIDATION_POOL_TOKENS,
            "original_validation_pool_to_train_ratio": ORIGINAL_VALIDATION_POOL_TO_TRAIN_RATIO,
            "frequency_index_source": "build only from shard_00001.bin",
            "note": (
                "The complete validation source pool is materialized; train.py "
                "fixed-val evaluation consumes only its first four batches by default."
            ),
        },
        "data_sha256": hashes,
    }
    write_json(OUTPUT_DIR / "run_contract.json", contract)
    return metadata


def main() -> None:
    """Run the fixed, versioned generation setting without CLI parameters."""
    try:
        metadata = generate()
    except (FileExistsError, ValueError, OSError) as exc:
        raise SystemExit(f"[toy] generation failed: {exc}") from exc

    split = metadata["split_alignment"]
    print(f"[toy] output_dir={OUTPUT_DIR}")
    print(
        f"[toy] train_tokens={split['train_tokens']:,} "
        f"validation_tokens={split['validation_tokens']:,}"
    )
    print(
        "[toy] full validation-pool:train raw-token ratio="
        f"{split['validation_to_train_token_ratio']:.10f}:1"
    )
    print(
        "[toy] fixed-val evaluated loss-token ratio (train:fixed-val)="
        f"{TRAIN_LOSS_TOKENS / FIXED_VALIDATION_LOSS_TOKENS:.8f}:1 "
        f"({FIXED_VALIDATION_DEVICE_BATCHES} batches)"
    )
    print(
        f"[toy] vocab={VOCAB_SIZE} n_embd={MODEL_N_EMBD} "
        f"sequence_len={SEQUENCE_LEN} alpha={ZIPF_ALPHA:.12g}"
    )
    print("[toy] raw iid shards only; context frequencies are measured downstream")


if __name__ == "__main__":
    main()
