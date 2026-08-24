#!/usr/bin/env python3
"""Trainer for the 5-gram frequency-controlled dataset.

Loads the block-structured token stream produced by ``data_gen.py`` via
``lib.make_dataloader(data_mode="ngram5_blocks")`` (the same loader the cluster
``train.py`` uses) and runs a standard nanoGPT training loop.  At configurable
probe steps it evaluates train/val loss *at the target position of each block*
(position 5, the ``next`` token) and decomposes the loss by the natural
5-gram bucket frequency ``r(b)`` so we can plot gap-vs-frequency.

The model is imported from ``model.py`` which resolves to the repository's
minimal ``NanoGPT`` (with live n-gram injection tables).  The cluster launcher
syncs the same ``code/train.py`` source beside this package.

The frequency decomposition reads ``fivegram_counts.npz`` (written by
data_gen) and looks up each target position's 5-gram bucket via hash5 %
bucket_count, matching the key scheme in the npz.

Env vars (all optional; defaults shown):
  NGRAM5_DATA_DIR       (required — path to data_gen output)
  NGRAM5_PROBE_STEPS    100,200,300,400,500,750,1000,2000
  NGRAM5_PROBE_FREQUENCY_MODE  exact_context  (or bucket_diagnostic)
  NGRAM5_BUCKET_EDGES   0,1,2,3,4,5,6,11,21,51,101,201,501,1001,5001
  MAX_TRAINING_STEPS    2000
  DEVICE_BATCH_SIZE     72
  MAX_SEQ_LEN            2048
  SEED                  42
  LEARNING_RATE         9e-4  (overridden by NANOGPT_ADAM_LR if set)
  WARMUP_RATIO          0.0
  ADAM_WARMDOWN_RATIO   0.65
  WARMDOWN_RATIO        0.95  (used for the late n-gram beta2 ramp)
  FINAL_LR_FRAC         0.05
  WEIGHT_DECAY          0.1
  VAL_LOSS_INTERVAL_STEPS 10
  VAL_LOSS_BATCHES      4
  TORCH_COMPILE         1
  REMOTE_RESULT_DIR     (optional) mirror jsonl here
  NGRAM5_CPU_SMOKE      0  (set to 1 to allow CPU + skip compile, for tests)
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

# Load the experiment's patched runtime lib under a private module name.
# ``model.py`` loads the cluster model definitions, whose imports can populate
# ``sys.modules['lib']`` with the unpatched repository-root module.  A plain
# ``from lib import ...`` would then silently bypass ngram5_blocks.
_runtime_spec = importlib.util.spec_from_file_location(
    "_ngram5_runtime_lib", _HERE / "lib.py"
)
if _runtime_spec is None or _runtime_spec.loader is None:
    raise RuntimeError("cannot load ngram5_freq_gap/lib.py")
_runtime_lib = importlib.util.module_from_spec(_runtime_spec)
_runtime_spec.loader.exec_module(_runtime_lib)
make_dataloader = _runtime_lib.make_dataloader

from model import GPT, GPTConfig, MODEL_PROVENANCE  # noqa: E402
from hash_utils import hash_bucket_tensor  # noqa: E402


def _encode_exact_context_tensor(context_tokens, vocab: int) -> torch.Tensor:
    key = torch.zeros_like(context_tokens[0], dtype=torch.int64)
    for token in context_tokens:
        key = key * vocab + token.to(torch.int64)
    return key


# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------

def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


DATA_DIR = Path(os.environ.get("NGRAM5_DATA_DIR", "")).resolve()
if not DATA_DIR:
    raise RuntimeError("NGRAM5_DATA_DIR is required (path to data_gen output)")
REMOTE_RESULT_DIR = (
    Path(os.environ["REMOTE_RESULT_DIR"]).resolve()
    if os.environ.get("REMOTE_RESULT_DIR")
    else None
)
CPU_SMOKE = env_bool("NGRAM5_CPU_SMOKE", False)

SEED = int(os.environ.get("SEED", "42"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "2048"))
MAX_TRAINING_STEPS = int(os.environ.get("MAX_TRAINING_STEPS", "2000"))
DEVICE_BATCH_SIZE = int(os.environ.get("DEVICE_BATCH_SIZE", "72"))
TOTAL_BATCH_SIZE = int(os.environ.get("TOTAL_BATCH_SIZE",
    str(DEVICE_BATCH_SIZE * MAX_SEQ_LEN)))

# LR: prefer NANOGPT_ADAM_LR (cluster convention) if set, else LEARNING_RATE.
LEARNING_RATE = float(os.environ.get("NANOGPT_ADAM_LR",
    os.environ.get("LEARNING_RATE", "0.004")))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.1"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.0"))
ADAM_WARMDOWN_RATIO = float(os.environ.get("ADAM_WARMDOWN_RATIO", "0.65"))
MUON_WARMDOWN_RATIO = float(os.environ.get("WARMDOWN_RATIO", "0.95"))
FINAL_LR_FRAC = float(os.environ.get("FINAL_LR_FRAC", "0.05"))
ADAM_BETAS = (0.8, 0.95)
DEMON_FINAL_BETA1 = 0.55
NGRAM_VE_BETAS = (0.0, 0.999)
NGRAM_VE_BETA2_WARMDOWN = 0.9999

VAL_LOSS_INTERVAL_STEPS = int(os.environ.get("VAL_LOSS_INTERVAL_STEPS", "10"))
VAL_LOSS_BATCHES = int(os.environ.get("VAL_LOSS_BATCHES", "4"))
PROBE_BATCHES = max(1, int(os.environ.get("NGRAM5_PROBE_BATCHES", "2")))
PROBE_FREQUENCY_MODE = os.environ.get(
    "NGRAM5_PROBE_FREQUENCY_MODE", "bucket"
).strip().lower()
if PROBE_FREQUENCY_MODE == "exact":
    PROBE_FREQUENCY_MODE = "exact_context"
if PROBE_FREQUENCY_MODE == "bucket":
    PROBE_FREQUENCY_MODE = "bucket_diagnostic"
if PROBE_FREQUENCY_MODE not in {"bucket_diagnostic", "exact_context"}:
    raise ValueError(
        "NGRAM5_PROBE_FREQUENCY_MODE must be exact_context or bucket_diagnostic"
    )
PROBE_DETAIL_FORMAT = "npz_v1"
PROBE_DETAIL_SCOPE = "fixed_probe_batches"
TRACE_ALL_BATCHES = env_bool("NGRAM5_TRACE_ALL_BATCHES", False)
TRACE_COMPRESSION = env_bool("NGRAM5_TRACE_COMPRESSION", True)
TRACE_ROOT_OVERRIDE = os.environ.get("NGRAM5_TRACE_DIR", "").strip()
TORCH_COMPILE = (
    env_bool("TORCH_COMPILE", True)
    and not CPU_SMOKE
    and not TRACE_ALL_BATCHES
)

_probe_steps_env = os.environ.get("NGRAM5_PROBE_STEPS",
    "100,200,300,400,500,750,1000,2000").replace(";", ",")
PROBE_STEPS = {int(v) for v in _probe_steps_env.split(",") if v.strip().isdigit()}

_edges_env = os.environ.get("NGRAM5_BUCKET_EDGES",
    "0,1,2,3,4,5,6,11,21,51,101,201,501,1001,5001").replace(";", ",")
BUCKET_EDGES = tuple(int(v) for v in _edges_env.split(",") if v.strip().isdigit())
assert BUCKET_EDGES and BUCKET_EDGES[0] == 0, "edges must start with 0"

# ---------------------------------------------------------------------------
# Distributed (DDP) + checkpoint control
# ---------------------------------------------------------------------------
# Standard torchrun env (RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT),
# with NGRAM5_* fallbacks for manual launches and CPU smoke tests.
DDP_WORLD_SIZE = int(os.environ.get(
    "WORLD_SIZE", os.environ.get("NGRAM5_WORLD_SIZE", "1")))
DDP_RANK = int(os.environ.get("RANK", os.environ.get("NGRAM5_RANK", "0")))
DDP_LOCAL_RANK = int(os.environ.get(
    "LOCAL_RANK", os.environ.get("NGRAM5_LOCAL_RANK", "0")))
DDP_MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
DDP_MASTER_PORT = os.environ.get("MASTER_PORT", "29500")

CKPT_DIR = os.environ.get("NGRAM5_CKPT_DIR", "").strip()
CKPT_INTERVAL_STEPS = int(os.environ.get("NGRAM5_CKPT_INTERVAL_STEPS", "1000"))
_ckpt_steps_env = os.environ.get(
    "NGRAM5_CKPT_STEPS", "").replace(";", ",")
CKPT_STEPS = {int(v) for v in _ckpt_steps_env.split(",") if v.strip().isdigit()}
# Keep only the newest N model-only (no-optimizer) checkpoints on disk; full
# (optimizer-carrying) checkpoints are always retained.
CKPT_KEEP_MODEL_ONLY = int(os.environ.get("NGRAM5_CKPT_KEEP_MODEL_ONLY", "2"))
INIT_CKPT = os.environ.get("NGRAM5_INIT_CKPT", "").strip()


# ---------------------------------------------------------------------------
# Data loading (via the private patched runtime lib above)
# ---------------------------------------------------------------------------
# BLOCK_LEN and TARGET_OFFSET are set at runtime from the dataset's meta.json
# (order-dependent: block = [c0..c_{order-1}, next, SEP], so
# BLOCK_LEN = order + 2, TARGET_OFFSET = order).
BLOCK_LEN = 7   # default (order=5); overwritten in main()
TARGET_OFFSET = 5


def _bucket_name(idx: int, edges: tuple[int, ...]) -> str:
    """Map a 1-based bucketize result to the label format used by build_exp6."""
    n = len(edges)
    if idx <= 0:
        return "0_0"
    if idx >= n:
        lo = edges[idx - 1]
        return f"ge_{lo}"
    lo = edges[idx - 1]
    hi = edges[idx] - 1
    if lo == hi:
        return f"{lo}_{hi}"
    return f"{lo}_{hi}"


# ---------------------------------------------------------------------------
# n-gram frequency index (mirrors cluster GlobalNgramFrequencyIndex)
# ---------------------------------------------------------------------------

class NgramIndex:
    """Legacy hash-bucket occupancy index for diagnostics only.

    This class must never be used for the reported frequency-gap probe.
    ``ExactNgramIndex`` is the only valid source for ``r``.
    """

    def __init__(self, data_dir: Path, device: torch.device, order: int):
        self.device = device
        self.order = order
        self.context_lookup = None
        meta = json.loads((data_dir / "metadata.json").read_text())
        self.vocab_size = int(meta.get("vocab_size", meta.get("vocab")))
        self.bucket_count = int(meta["bucket_count"])
        import numpy as np
        with np.load(data_dir / "fivegram_counts.npz") as z:
            self.keys = torch.from_numpy(z["keys"].astype(np.int64, copy=False)).to(device)
            self.counts = torch.from_numpy(z["counts"].astype(np.int64, copy=False)).to(device)
        if self.keys.numel() == 0:
            raise ValueError("fivegram_counts.npz is empty")
        if not bool(torch.all(self.keys[1:] > self.keys[:-1])):
            raise ValueError("fivegram_counts keys must be sorted ascending")

    def lookup_frequency(self, *context_tokens) -> torch.Tensor:
        """Return natural frequency r(b) for each n-gram context in the batch.

        Args: ``order`` context token tensors (e.g. 3 for trigram, 5 for 5-gram).
        """
        if len(context_tokens) != self.order:
            raise ValueError(f"expected {self.order} context tensors, got {len(context_tokens)}")
        bucket_ids = hash_bucket_tensor(*context_tokens, bucket_count=self.bucket_count)
        flat = bucket_ids.reshape(-1)
        locs = torch.searchsorted(self.keys, flat)
        valid = locs < self.keys.numel()
        safe = locs.clamp_max(self.keys.numel() - 1)
        valid = valid & (self.keys[safe] == flat)
        freqs = torch.zeros_like(flat, dtype=torch.long)
        freqs[valid] = self.counts[safe[valid]]
        return freqs.view_as(bucket_ids)


class ExactNgramIndex:
    """Exact train-epoch context frequency index.

    The frequency key is the base-v encoded context tuple.  It is independent
    of the model's hash-bucket lookup and is built from train only.
    """

    def __init__(self, data_dir: Path, device: torch.device, order: int):
        self.device = device
        self.order = order
        self.context_lookup = None
        metadata_path = data_dir / "metadata.json"
        meta = json.loads(metadata_path.read_text())
        self.vocab_size = int(meta["vocab_size"])
        self.frequency_definition = meta.get(
            "frequency_definition", "unknown"
        )
        if self.frequency_definition != "exact_train_epoch_context_count":
            raise ValueError(
                f"frequency index is not exact train-only: {self.frequency_definition}"
            )
        self.frequency_source_split = meta.get("frequency_source_split", "train")
        if self.frequency_source_split != "train":
            raise ValueError(
                "frequency index must be sourced from the complete train epoch"
            )
        import numpy as np
        with np.load(data_dir / "exact_ngram_counts.npz") as z:
            if "contexts" in z:
                contexts = z["contexts"].astype(np.int32, copy=False)
                counts = z["counts"].astype(np.int64, copy=False)
                if contexts.ndim != 2 or contexts.shape[1] != order:
                    raise ValueError("exact context matrix has the wrong shape")
                self.context_lookup = {
                    tuple(int(token) for token in context): int(count)
                    for context, count in zip(contexts, counts)
                }
                self.keys = None
                self.counts = torch.from_numpy(counts).to(device)
            else:
                self.keys = torch.from_numpy(z["keys"].astype(np.int64, copy=False)).to(device)
                self.counts = torch.from_numpy(z["counts"].astype(np.int64, copy=False)).to(device)
        if self.context_lookup is not None:
            if not self.context_lookup:
                raise ValueError("exact_ngram_counts.npz is empty")
            return
        if self.keys.numel() == 0:
            raise ValueError("exact_ngram_counts.npz is empty")
        if not bool(torch.all(self.keys[1:] > self.keys[:-1])):
            raise ValueError("exact n-gram keys must be sorted ascending")

    def lookup_frequency(self, *context_tokens) -> torch.Tensor:
        if len(context_tokens) != self.order:
            raise ValueError(f"expected {self.order} context tensors, got {len(context_tokens)}")
        if self.context_lookup is not None:
            flat = [
                token.detach().cpu().reshape(-1).tolist()
                for token in context_tokens
            ]
            values = [
                self.context_lookup.get(tuple(row), 0)
                for row in zip(*flat)
            ]
            return torch.tensor(
                values,
                dtype=torch.long,
                device=context_tokens[0].device,
            ).view_as(context_tokens[0])
        keys = _encode_exact_context_tensor(context_tokens, self.vocab_size).reshape(-1)
        locs = torch.searchsorted(self.keys, keys)
        valid = locs < self.keys.numel()
        safe = locs.clamp_max(self.keys.numel() - 1)
        valid = valid & (self.keys[safe] == keys)
        freqs = torch.zeros_like(keys, dtype=torch.long)
        freqs[valid] = self.counts[safe[valid]]
        return freqs.view_as(context_tokens[0])


# ---------------------------------------------------------------------------
# LR schedule (step-indexed, same shape as vanilla_control / cluster baseline)
# ---------------------------------------------------------------------------

def lr_multiplier(step: int, max_steps: int, warmup_ratio: float,
                  warmdown_ratio: float, final_lr_fraction: float) -> float:
    progress = min(step / max_steps, 1.0)
    if progress < warmup_ratio:
        return progress / warmup_ratio if warmup_ratio > 0 else 1.0
    warmdown_start = 1.0 - warmdown_ratio
    if progress < warmdown_start:
        return 1.0
    cooldown = max(0.0, (1.0 - progress) / warmdown_ratio)
    return cooldown + (1.0 - cooldown) * final_lr_fraction


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

_ARTIFACTS_ENABLED = True


def _set_artifacts_enabled(enabled: bool) -> None:
    global _ARTIFACTS_ENABLED
    _ARTIFACTS_ENABLED = enabled


def write_json(path: Path, payload: Any) -> None:
    if not _ARTIFACTS_ENABLED:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(paths: list[Path], payload: Any) -> None:
    if not _ARTIFACTS_ENABLED:
        return
    line = json.dumps(payload, sort_keys=True) + "\n"
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def write_batch_trace(
    trace_root: Path,
    manifest_paths: tuple[Path, ...],
    *,
    split: str,
    step: int,
    batch_index: int,
    loader_epoch: int,
    targets: torch.Tensor,
    token_losses: torch.Tensor,
    injection_stats: dict[str, dict[str, Any]],
    batch_hash: str,
    compression: bool,
) -> dict[str, Any]:
    import numpy as np

    target_cpu = targets.detach().to(dtype=torch.int32, device="cpu")
    loss_cpu = token_losses.detach().to(dtype=torch.float32, device="cpu")
    valid_cpu = target_cpu.ne(-1)
    valid_losses = loss_cpu[valid_cpu]
    if valid_losses.numel():
        loss_mean = float(valid_losses.mean())
        loss_min = float(valid_losses.min())
        loss_max = float(valid_losses.max())
    else:
        loss_mean = loss_min = loss_max = 0.0

    payload = {
        "targets": target_cpu.numpy(),
        "token_loss": loss_cpu.numpy(),
        "valid_mask": valid_cpu.numpy(),
        "bigram_sequence_rms": np.asarray(
            injection_stats.get("bigram_rms", {}).get("sequence_rms", []),
            dtype=np.float32,
        ),
        "trigram_sequence_rms": np.asarray(
            injection_stats.get("trigram_rms", {}).get("sequence_rms", []),
            dtype=np.float32,
        ),
        "total_sequence_rms": np.asarray(
            injection_stats.get("total_rms", {}).get("sequence_rms", []),
            dtype=np.float32,
        ),
        "fourgram_sequence_rms": np.asarray(
            injection_stats.get("fourgram_rms", {}).get("sequence_rms", []),
            dtype=np.float32,
        ),
    }
    split_dir = trace_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".npz" if compression else ".npz"
    path = split_dir / f"step_{step:06d}_batch_{batch_index:04d}{suffix}"
    temporary = path.with_name(path.name + ".tmp.npz")
    if compression:
        np.savez_compressed(temporary, **payload)
    else:
        np.savez(temporary, **payload)
    temporary.replace(path)

    valid_count = int(valid_cpu.sum())
    record = {
        "schema_version": 1,
        "format": "batch_trace_npz_v1",
        "split": split,
        "step": step,
        "batch_index": batch_index,
        "loader_epoch": int(loader_epoch),
        "path": str(path.relative_to(trace_root)),
        "bytes": path.stat().st_size,
        "batch_hash": batch_hash,
        "shape": list(target_cpu.shape),
        "loss_dtype": "float32",
        "target_dtype": "int32",
        "valid_token_count": valid_count,
        "loss_mean": loss_mean,
        "loss_min": loss_min,
        "loss_max": loss_max,
        "bigram_batch_rms": float(
            injection_stats.get("bigram_rms", {}).get("batch_rms", 0.0)
        ),
        "trigram_batch_rms": float(
            injection_stats.get("trigram_rms", {}).get("batch_rms", 0.0)
        ),
        "total_batch_rms": float(
            injection_stats.get("total_rms", {}).get("batch_rms", 0.0)
        ),
        "fourgram_batch_rms": float(
            injection_stats.get("fourgram_rms", {}).get("batch_rms", 0.0)
        ),
        "norm_definition": "sqrt(mean(hidden_state.float() ** 2))",
    }
    line = json.dumps(record, sort_keys=True) + "\n"
    for manifest_path in manifest_paths:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    return record


def tensor_pair_sha256(inputs: torch.Tensor, targets: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in (inputs, targets):
        cpu = tensor.detach().contiguous().cpu()
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def batch_to_device(batch, device: torch.device):
    """Move the tensor portion of a loader batch to the training device."""
    inputs, targets, epoch, *rest = batch
    return (
        inputs.to(device=device, non_blocking=True),
        targets.to(device=device, non_blocking=True),
        epoch,
        *rest,
    )


def _write_probe_details(
    detail_dirs: tuple[Path, ...],
    detail_manifest_paths: tuple[Path, ...],
    *,
    step: int,
    split: str,
    inputs: list,
    targets: list,
    losses: list,
    contexts: list,
    target_tokens: list,
    target_losses: list,
    frequencies: list,
    order: int,
    block_len: int,
    target_offset: int,
) -> None:
    if not detail_dirs or not inputs:
        return
    import numpy as np

    payload = {
        "inputs": np.concatenate(inputs, axis=0),
        "targets": np.concatenate(targets, axis=0),
        "token_losses": np.concatenate(losses, axis=0),
        "contexts": np.concatenate(contexts, axis=0),
        "target_tokens": np.concatenate(target_tokens, axis=0),
        "target_losses": np.concatenate(target_losses, axis=0),
        "frequencies": np.concatenate(frequencies, axis=0),
        "target_positions": np.flatnonzero(
            ((np.arange(targets[0].shape[1]) + 1) % block_len) == target_offset
        ).astype("int32"),
    }
    filename = f"step_{step:05d}_{split}.npz"
    manifests = []
    for detail_dir in detail_dirs:
        detail_dir.mkdir(parents=True, exist_ok=True)
        path = detail_dir / filename
        np.savez_compressed(path, **payload)
        manifests.append({
            "step": step,
            "split": split,
            "path": str(path),
            "bytes": path.stat().st_size,
            "rows": int(payload["inputs"].shape[0]),
            "sequence_length": int(payload["inputs"].shape[1]),
            "target_count": int(payload["contexts"].shape[0] * payload["contexts"].shape[1]),
            "order": order,
            "block_len": block_len,
            "target_offset": target_offset,
            "format": PROBE_DETAIL_FORMAT,
        })
    for manifest_path, manifest in zip(detail_manifest_paths, manifests):
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Frequency-bucketed gap probe
# ---------------------------------------------------------------------------

@torch.no_grad()
def frequency_gap_probe(model, batches,
                        bucket_edges: tuple[int, ...], index: NgramIndex,
                        step: int, epoch: int, *, is_train: bool,
                        device: torch.device, order: int,
                        frequency_mode: str = "exact_context",
                        detail_dirs: tuple[Path, ...] = (),
                        detail_manifest_paths: tuple[Path, ...] = ()) -> list[dict]:
    """Evaluate per-frequency-bucket train or val loss at target positions.

    For each batch we:
      1. Forward the model to get per-token CE.
      2. Build a target-position mask: True at positions where targets[t] is
         the ``next`` token of a block.  Block layout is
         [c0..c_{order-1}, next, SEP]; targets = row[1:] shifts by 1, so
         the ``next`` at block-offset ``order`` becomes targets-offset
         ``order-1``.  We compute ``(t+1) % BLOCK_LEN == TARGET_OFFSET``.
      3. For each target position, extract the ``order``-token context
         (inputs[t-order+1 .. t]) and look up its natural frequency r(b).
      4. Bucketize r(b) into the frequency edges and accumulate loss/count.
    """
    edges_t = torch.tensor(bucket_edges, device=device, dtype=torch.long)
    n_edge = len(bucket_edges)
    sum_loss = torch.zeros(n_edge + 1, device=device, dtype=torch.float64)
    counts = torch.zeros(n_edge + 1, device=device, dtype=torch.float64)
    exact_stats: dict[int, list[float]] = {}
    detail_inputs = []
    detail_targets = []
    detail_losses = []
    detail_contexts = []
    detail_target_tokens = []
    detail_target_losses = []
    detail_frequencies = []

    model.eval()
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None
    for source_batch in batches:
        batch = batch_to_device(source_batch, device)
        inputs, targets = batch[0], batch[1]
        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype) \
                if autocast_dtype is not None else torch.amp.autocast(device_type="cpu", enabled=False):
            logits = model(inputs)
        ce = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)

        T = targets.size(1)
        pos = torch.arange(T, device=device)
        target_mask = ((pos + 1) % BLOCK_LEN == TARGET_OFFSET)
        target_mask = target_mask & (pos >= order - 1)
        row_stride = ((T + 1) // BLOCK_LEN) * BLOCK_LEN
        target_mask = target_mask & ((pos + 1) < row_stride)

        tgt_positions = pos[target_mask]
        if tgt_positions.numel() == 0:
            continue
        # Extract order context tokens: inputs[t-order+1, ..., t]
        ctx_tokens = [inputs[:, tgt_positions - order + 1 + k].reshape(-1)
                      for k in range(order)]
        freqs = index.lookup_frequency(*ctx_tokens).view(
            inputs.size(0), -1)
        bucket_idx = torch.bucketize(freqs, edges_t, right=True)
        losses = ce[:, tgt_positions]
        if detail_dirs:
            context = torch.stack(ctx_tokens, dim=-1).view(
                inputs.size(0), -1, order
            )
            detail_inputs.append(inputs.detach().cpu().numpy().astype("int32"))
            detail_targets.append(targets.detach().cpu().numpy().astype("int32"))
            detail_losses.append(ce.detach().cpu().numpy().astype("float32"))
            detail_contexts.append(context.detach().cpu().numpy().astype("int32"))
            detail_target_tokens.append(
                targets[:, tgt_positions].detach().cpu().numpy().astype("int32")
            )
            detail_target_losses.append(
                losses.detach().cpu().numpy().astype("float32")
            )
            detail_frequencies.append(
                freqs.detach().cpu().numpy().astype("int64")
            )
        if frequency_mode == "exact_context":
            for frequency in torch.unique(freqs).tolist():
                sel = freqs == frequency
                stats = exact_stats.setdefault(int(frequency), [0.0, 0.0])
                stats[0] += float(sel.sum().item())
                stats[1] += float(losses[sel].sum().item())
        else:
            for bi in range(1, n_edge + 1):
                sel = (bucket_idx == bi)
                counts[bi] += sel.sum().double()
                sum_loss[bi] += losses[sel].sum().double()

    records = []
    role = "train" if is_train else "val"
    if frequency_mode == "exact_context":
        total_valid = sum(stats[0] for stats in exact_stats.values())
        for frequency in sorted(exact_stats):
            count, loss_sum = exact_stats[frequency]
            if count == 0:
                continue
            loss = loss_sum / count
            frac = count / total_valid if total_valid > 0 else 0.0
            records.append({
                "measurement": "allgram_fixed_frequency_decomposition",
                "schema_version": 2,
                "branch": "exact_context",
                "bucket": None,
                "frequency": frequency,
                "frequency_definition": "exact_train_epoch_context_count",
                "frequency_source_split": "train",
                "frequency_key_type": "exact_context",
                "probe_set": role,
                "step": step,
                "epoch": epoch,
                "train_loss": loss if is_train else None,
                "val_loss": loss if not is_train else None,
                "train_fraction": frac if is_train else None,
                "val_fraction": frac if not is_train else None,
                "within_bucket_gap": None,
                "support_mode": "paired",
                "composition_component": None,
                "conditional_loss_component": None,
                "exact_global_gap_contribution": None,
                "shaoyang_val_fraction_gap_contribution": None,
                "reconstruction_error": None,
            })
        _write_probe_details(
            detail_dirs,
            detail_manifest_paths,
            step=step,
            split=role,
            inputs=detail_inputs,
            targets=detail_targets,
            losses=detail_losses,
            contexts=detail_contexts,
            target_tokens=detail_target_tokens,
            target_losses=detail_target_losses,
            frequencies=detail_frequencies,
            order=order,
            block_len=BLOCK_LEN,
            target_offset=TARGET_OFFSET,
        )
        model.train()
        return records

    total_valid = counts[1:].sum().item()
    for bi in range(1, n_edge + 1):
        c = counts[bi].item()
        if c == 0:
            continue
        loss = (sum_loss[bi] / c).item()
        frac = c / total_valid if total_valid > 0 else 0.0
        records.append({
            "measurement": "allgram_fixed_frequency_decomposition",
            "schema_version": 2,
            "branch": "fivegram",
            "bucket": _bucket_name(bi, bucket_edges),
            "frequency": None,
            "probe_set": role,
            "step": step,
            "epoch": epoch,
            "train_loss": loss if is_train else None,
            "val_loss": loss if not is_train else None,
            "train_fraction": frac if is_train else None,
            "val_fraction": frac if not is_train else None,
            "within_bucket_gap": None,
            "support_mode": "paired",
            "composition_component": None,
            "conditional_loss_component": None,
            "exact_global_gap_contribution": None,
            "shaoyang_val_fraction_gap_contribution": None,
            "reconstruction_error": None,
        })
    _write_probe_details(
        detail_dirs,
        detail_manifest_paths,
        step=step,
        split=role,
        inputs=detail_inputs,
        targets=detail_targets,
        losses=detail_losses,
        contexts=detail_contexts,
        target_tokens=detail_target_tokens,
        target_losses=detail_target_losses,
        frequencies=detail_frequencies,
        order=order,
        block_len=BLOCK_LEN,
        target_offset=TARGET_OFFSET,
    )
    model.train()
    return records


def merge_train_val_gap(train_recs: list[dict], val_recs: list[dict]) -> list[dict]:
    def record_key(record: dict) -> tuple[str, object]:
        if record.get("branch") == "exact_context":
            return ("frequency", record.get("frequency"))
        return ("bucket", record.get("bucket"))

    val_by_key = {record_key(record): record for record in val_recs}
    merged = []
    for tr in train_recs:
        va = val_by_key.get(record_key(tr))
        if va is None:
            merged.append(tr)
            continue
        train_loss = tr["train_loss"]
        val_loss = va["val_loss"]
        gap = (val_loss - train_loss) if (train_loss is not None and val_loss is not None) else None
        va_frac = va.get("val_fraction") or 0.0
        contrib = (gap * va_frac) if gap is not None else None
        merged.append({
            **tr,
            "val_loss": val_loss,
            "val_fraction": va_frac,
            "within_bucket_gap": gap,
            "exact_global_gap_contribution": contrib,
            "shaoyang_val_fraction_gap_contribution": contrib,
        })
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CPU_SMOKE and not torch.cuda.is_available():
        raise RuntimeError("train.py requires CUDA; set NGRAM5_CPU_SMOKE=1 for CPU tests")

    # ---- distributed init ----
    ddp = DDP_WORLD_SIZE > 1 and not CPU_SMOKE
    is_main = (not ddp) or (DDP_RANK == 0)
    _set_artifacts_enabled(is_main)
    if ddp:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://{DDP_MASTER_ADDR}:{DDP_MASTER_PORT}",
            rank=DDP_RANK,
            world_size=DDP_WORLD_SIZE,
        )
        torch.cuda.set_device(DDP_LOCAL_RANK)
        print(f"[ddp] rank={DDP_RANK}/{DDP_WORLD_SIZE} "
              f"local_rank={DDP_LOCAL_RANK} master={DDP_MASTER_ADDR}:{DDP_MASTER_PORT}",
              flush=True)

    device = torch.device(
        f"cuda:{DDP_LOCAL_RANK}" if ddp
        else ("cuda" if not CPU_SMOKE else "cpu")
    )
    torch.manual_seed(SEED + DDP_RANK)
    if not CPU_SMOKE:
        torch.cuda.manual_seed(SEED + DDP_RANK)
        torch.set_float32_matmul_precision("high")

    # ---- resume state ----
    start_step = 0
    resumed = False
    if INIT_CKPT:
        if not Path(INIT_CKPT).is_file():
            raise FileNotFoundError(f"NGRAM5_INIT_CKPT not found: {INIT_CKPT}")

    # ---- load dataset metadata ----
    meta = json.loads((DATA_DIR / "meta.json").read_text())
    vocab = int(meta["vocab"])
    sep_token = int(meta["sep_token"])
    order = int(meta["order"])
    if meta.get("frequency_definition") != "exact_train_epoch_context_count":
        raise ValueError(
            "refusing to run: frequency definition is not the exact full-train "
            f"context count ({meta.get('frequency_definition')!r})"
        )
    if meta.get("frequency_source_split") != "train":
        raise ValueError(
            "refusing to run: frequency source must be the train split"
        )
    if meta.get("frequency_key_type") != "exact_context":
        raise ValueError(
            "refusing to run: frequency key must be collision-free exact_context"
        )
    if meta.get("hash_bucket_occupancy_diagnostic", False):
        raise ValueError(
            "refusing to run: hash bucket occupancy cannot define the gap frequency"
        )
    if not (DATA_DIR / "exact_ngram_counts.npz").is_file():
        raise FileNotFoundError(
            f"missing collision-free frequency index: {DATA_DIR / 'exact_ngram_counts.npz'}"
        )
    # Set order-dependent globals for the probe.
    global BLOCK_LEN, TARGET_OFFSET
    BLOCK_LEN = order + 2
    TARGET_OFFSET = order
    print(f"[trainer] dataset: {DATA_DIR}  order={order}  vocab={vocab}  sep={sep_token}", flush=True)
    print(f"[trainer] train_tokens={meta['train_tokens']:,}  "
          f"val_tokens={meta['val_tokens']:,}  "
          f"nonempty_buckets={meta['n_nonempty_buckets']:,}", flush=True)

    # ---- exact train-only frequency index (rank 0 only; probes run on rank 0) ----
    freq_index = None
    if is_main:
        freq_index = ExactNgramIndex(DATA_DIR, device, order=order)
        n_exact = (freq_index.keys.numel() if freq_index.keys is not None
                   else freq_index.counts.numel())
        print(f"[trainer] exact ngram index: {n_exact:,} contexts loaded "
              f"(source=train full epoch, order={order})", flush=True)

    # ---- data loaders (via lib.make_dataloader) ----
    os.environ["NGRAM5_DATA_DIR"] = str(DATA_DIR)
    os.environ["NGRAM5_RANK"] = str(DDP_RANK)
    os.environ["NGRAM5_WORLD_SIZE"] = str(DDP_WORLD_SIZE)
    train_loader = make_dataloader(
        tokenizer=None, B=DEVICE_BATCH_SIZE, T=MAX_SEQ_LEN, split="train",
        data_mode="ngram5_blocks", data_seed=SEED, return_metadata=False)
    val_loader = make_dataloader(
        tokenizer=None, B=DEVICE_BATCH_SIZE, T=MAX_SEQ_LEN, split="val",
        data_mode="ngram5_blocks", data_seed=SEED + 1, return_metadata=False)

    # ---- model ----
    # GPTConfig field names differ between vanilla and NanoGPTOriginal; we
    # build the config from env to match the cluster's build_model_config.
    n_embd = int(os.environ.get("MODEL_DIM", "768"))
    n_head_dim = int(os.environ.get("MODEL_HEAD_DIM", "128"))
    n_head = n_embd // n_head_dim if n_head_dim else 6
    config_kwargs = dict(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab,
        n_layer=int(os.environ.get("MODEL_DEPTH", "8")),
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd,
        window_pattern=os.environ.get("WINDOW_PATTERN", "LLLL"),
        dropout=0.0,
        bias=True,
        enable_unigram_ve=env_bool("ENABLE_UNIGRAM_VE", False),
        enable_bigram_ve=env_bool("ENABLE_BIGRAM_VE", True),
        enable_trigram_ve=env_bool("ENABLE_TRIGRAM_VE", True),
        enable_fourgram_ve=env_bool("ENABLE_FOURGRAM_VE", False),
        current_ngram_injection_impl=os.environ.get(
            "CURRENT_NGRAM_INJECTION_IMPL", "none"),
        enable_nanogpt_ngram_ve=env_bool("NANOGPT_ENABLE_NGRAM_VE", True),
        nanogpt_attention_impl=os.environ.get("NANOGPT_ATTENTION_IMPL", "fused"),
        nanogpt_ngram_injection_impl=os.environ.get(
            "NANOGPT_NGRAM_INJECTION_IMPL", "nanogpt"),
        nanogpt_ngram_injection_position=os.environ.get(
            "NANOGPT_NGRAM_INJECTION_POSITION", "input"),
        position_encoding=os.environ.get("POSITION_ENCODING", "learned_abs"),
        attention_norm=os.environ.get("CURRENT_ATTENTION_NORM", "none"),
        normalization=os.environ.get("CURRENT_NORMALIZATION", "layernorm"),
        enable_head_gate=os.environ.get("CURRENT_HEAD_GATE", "none") == "enabled",
        residual_path=os.environ.get("CURRENT_RESIDUAL_PATH", "plain"),
        enable_layer_pool=os.environ.get("CURRENT_LAYER_POOL", "none") == "enabled",
        mlp_activation=os.environ.get("CURRENT_MLP", "gelu"),
        tie_embeddings=os.environ.get("CURRENT_EMBEDDING_TYING", "tied") == "tied",
        embedding_init=os.environ.get("CURRENT_EMBEDDING_INIT", "nanogpt_like"),
        block_init=os.environ.get("CURRENT_BLOCK_INIT", "nanogpt_style"),
        logit_softcap=os.environ.get("CURRENT_LOGIT_SOFTCAP", "none"),
        linear_bias=os.environ.get("CURRENT_LINEAR_BIAS", "none"),
    )

    # Filter to fields the config class accepts
    import inspect
    sig = inspect.signature(GPTConfig.__init__)
    accepted = set(sig.parameters) - {"self"}
    filtered = {k: v for k, v in config_kwargs.items() if k in accepted}
    dropped = {k: v for k, v in config_kwargs.items() if k not in accepted}
    if dropped:
        print(f"[trainer] dropped unsupported config fields: {list(dropped)}", flush=True)

    config = GPTConfig(**filtered)
    if CPU_SMOKE:
        model = GPT(config).to(device)
        if hasattr(model, "init_weights"):
            model.init_weights()
    else:
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=device)
        model.init_weights()
        model.to(dtype=torch.bfloat16)
        if hasattr(model, "tie_weights"):
            model.tie_weights()
    param_counts = (model.num_scaling_params() if hasattr(model, "num_scaling_params")
                    else {"total": sum(p.numel() for p in model.parameters())})
    print(f"[trainer] model: {type(model).__name__}  params={param_counts.get('total', '?'):,}",
          flush=True)
    print(f"[trainer] model provenance: {MODEL_PROVENANCE.get('source_description', '?')}",
          flush=True)

    if hasattr(model, "setup_optimizer"):
        optimizer = model.setup_optimizer(
            unembedding_lr=0.004,
            embedding_lr=0.6,
            scalar_lr=0.8,
            adam_betas=ADAM_BETAS,
            matrix_lr=0.04,
            weight_decay=WEIGHT_DECAY,
            ngram_ve_betas=NGRAM_VE_BETAS,
            ngram_ve_lr_scale=1.0,
            nanogpt_adam_lr=LEARNING_RATE,
            nanogpt_ngram_optimizer=os.environ.get(
                "NANOGPT_NGRAM_OPTIMIZER", "mixed"),
            nanogpt_matrix_optimizer=os.environ.get(
                "NANOGPT_MATRIX_OPTIMIZER", "adamw"),
            nanogpt_optimizer_grouping=os.environ.get(
                "NANOGPT_OPTIMIZER_GROUPING", "nanogpt"),
            current_matrix_optimizer="adamw",
            current_matrix_adam_lr=LEARNING_RATE,
            current_optimizer_grouping="nanogpt_style",
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS,
            weight_decay=WEIGHT_DECAY, fused=not CPU_SMOKE,
        )
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

    optimizer_groups = getattr(optimizer, "param_groups", [])
    ngram_groups = [g for g in optimizer_groups if g.get("is_ngram_ve", False)]
    adam_groups = [g for g in optimizer_groups if not g.get("is_ngram_ve", False)]
    adam_demon_groups = [
        g for g in adam_groups if g.get("demon_beta1", False) and "betas" in g
    ]
    for group in optimizer_groups:
        group.setdefault("initial_lr", group["lr"])

    # ---- resume: load model + optimizer weights ----
    if INIT_CKPT:
        ckpt = torch.load(INIT_CKPT, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            print(f"[resume] state_dict mismatch: missing={missing} "
                  f"unexpected={unexpected}", flush=True)
        if ckpt.get("has_optimizer") and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            print("[resume] checkpoint has no optimizer state; "
                  "optimizer re-initialized from scratch", flush=True)
        start_step = int(ckpt.get("step", 0))
        resumed = True
        print(f"[resume] loaded {INIT_CKPT} at step {start_step}", flush=True)

    # ---- DDP wrap (after optimizer creation so setup_optimizer is untouched) ----
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[DDP_LOCAL_RANK],
            find_unused_parameters=True,
        )

    if TORCH_COMPILE and not CPU_SMOKE:
        model = torch.compile(model, dynamic=False, fullgraph=True,
                              mode=os.environ.get("TORCH_COMPILE_MODE", "max-autotune"))

    # ---- fixed train/val batches for comparable periodic probes ----
    fixed_train_probe_batches = []
    for _ in range(PROBE_BATCHES):
        fixed_train_probe_batches.append(
            batch_to_device(next(train_loader), device)
        )
    fixed_train_probe_hashes = [
        tensor_pair_sha256(x, y) for x, y, _ in fixed_train_probe_batches
    ]

    fixed_val_batches = []
    for _ in range(max(VAL_LOSS_BATCHES, PROBE_BATCHES)):
        vx, vy, vep = batch_to_device(next(val_loader), device)
        fixed_val_batches.append((vx, vy, vep))
    fixed_val_hashes = [
        tensor_pair_sha256(vx, vy)
        for vx, vy, _ in fixed_val_batches[:VAL_LOSS_BATCHES]
    ]
    fixed_val_probe_batches = fixed_val_batches[:PROBE_BATCHES]
    fixed_val_probe_hashes = [
        tensor_pair_sha256(vx, vy) for vx, vy, _ in fixed_val_probe_batches
    ]
    print(f"[trainer] fixed train probe hashes: {fixed_train_probe_hashes}", flush=True)
    print(f"[trainer] fixed val hashes: {fixed_val_hashes}", flush=True)
    print(f"[trainer] fixed val probe hashes: {fixed_val_probe_hashes}", flush=True)

    # ---- artifacts ----
    train_loss_paths: list[Path] = []
    val_loss_paths: list[Path] = []
    decomp_paths: list[Path] = []
    detail_dirs: tuple[Path, ...] = ()
    detail_manifest_paths: tuple[Path, ...] = ()
    trace_root = Path(TRACE_ROOT_OVERRIDE) if TRACE_ROOT_OVERRIDE else _HERE / "run_artifacts"
    trace_manifest_paths: tuple[Path, ...] = ()
    artifact_dir = _HERE / "run_artifacts"
    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    if is_main:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        train_loss_paths = [artifact_dir / f"training_loss_{run_stamp}.jsonl"]
        val_loss_paths = [artifact_dir / f"validation_loss_{run_stamp}.jsonl"]
        decomp_paths = [artifact_dir / f"allgram_frequency_decomposition_{run_stamp}.jsonl"]
        detail_dirs = (artifact_dir / f"probe_details_{run_stamp}",)
        detail_manifest_paths = (
            artifact_dir / f"probe_details_{run_stamp}.jsonl",
        )
        trace_root = Path(TRACE_ROOT_OVERRIDE) if TRACE_ROOT_OVERRIDE else artifact_dir / f"batch_trace_{run_stamp}"
        trace_manifest_paths = (
            trace_root / "trace_manifest.jsonl",
        )
        if REMOTE_RESULT_DIR is not None:
            REMOTE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
            train_loss_paths.append(REMOTE_RESULT_DIR / "training_loss.jsonl")
            val_loss_paths.append(REMOTE_RESULT_DIR / "validation_loss.jsonl")
            decomp_paths.append(REMOTE_RESULT_DIR / "allgram_frequency_decomposition.jsonl")
            detail_dirs = detail_dirs + (REMOTE_RESULT_DIR / "probe_details",)
            detail_manifest_paths = detail_manifest_paths + (
                REMOTE_RESULT_DIR / "probe_details.jsonl",
            )
            if not TRACE_ROOT_OVERRIDE:
                trace_root = REMOTE_RESULT_DIR / "batch_trace"
            trace_manifest_paths = trace_manifest_paths + (
                REMOTE_RESULT_DIR / "trace_manifest.jsonl",
            )
        for p in train_loss_paths + val_loss_paths + decomp_paths + list(detail_manifest_paths):
            p.write_text("", encoding="utf-8")
        if TRACE_ALL_BATCHES:
            trace_root.mkdir(parents=True, exist_ok=True)
            for p in trace_manifest_paths:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("", encoding="utf-8")

    run_contract = {
        "schema_version": 1,
        "control": "ngram5_freq_gap",
        "model_provenance": MODEL_PROVENANCE,
        "parameter_counts": param_counts,
        "optimizer": {
            "class": type(optimizer).__name__,
            "learning_rate": LEARNING_RATE,
            "adam_betas": list(ADAM_BETAS),
            "ngram_ve_betas": list(NGRAM_VE_BETAS),
            "weight_decay": WEIGHT_DECAY,
            "ngram_optimizer": os.environ.get("NANOGPT_NGRAM_OPTIMIZER", "mixed"),
            "matrix_optimizer": os.environ.get("NANOGPT_MATRIX_OPTIMIZER", "adamw"),
            "grouping": os.environ.get("NANOGPT_OPTIMIZER_GROUPING", "nanogpt"),
        },
        "schedule": {
            "warmup_ratio": WARMUP_RATIO,
            "adam_warmdown_ratio": ADAM_WARMDOWN_RATIO,
            "ngram_lr_schedule": "constant",
            "final_lr_fraction": FINAL_LR_FRAC,
        },
        "experiment": {
            "seed": SEED,
            "max_training_steps": MAX_TRAINING_STEPS,
            "model_class": type(model).__name__,
            "trunk": os.environ.get("NGRAM5_TRUNK", "transformer"),
            "sequence_len": MAX_SEQ_LEN,
            "device_batch_size": DEVICE_BATCH_SIZE,
            "total_batch_tokens": TOTAL_BATCH_SIZE,
            "validation_interval_steps": VAL_LOSS_INTERVAL_STEPS,
            "validation_batches": VAL_LOSS_BATCHES,
            "validation_batch_hashes": fixed_val_hashes,
            "frequency_probe_batches": PROBE_BATCHES,
            "train_probe_batch_hashes": fixed_train_probe_hashes,
            "val_probe_batch_hashes": fixed_val_probe_hashes,
            "probe_steps": sorted(PROBE_STEPS),
            "probe_frequency_mode": PROBE_FREQUENCY_MODE,
            "frequency_definition": "exact_train_epoch_context_count",
            "frequency_source_split": "train",
            "frequency_key_type": "exact_context",
            "frequency_index_scope": "complete upstream train epoch before "
                                      "controlled block resampling",
            "hash_bucket_occupancy_used_for_gap": False,
            "probe_detail_format": PROBE_DETAIL_FORMAT,
            "probe_detail_scope": PROBE_DETAIL_SCOPE,
            "probe_detail_files": "probe_details/{step}_{split}.npz",
            "trace_all_batches": TRACE_ALL_BATCHES,
            "trace_compression": TRACE_COMPRESSION,
            "trace_root": str(trace_root),
            "trace_manifest": "trace_manifest.jsonl",
            "trace_schema": "batch_trace_npz_v1",
            "trace_validation_mode": "fixed_validation_batches_every_step",
            "injection_norm_definition": "sqrt(mean(hidden_state.float() ** 2))",
            "injection_norm_components": ["bigram", "trigram", "total"],
            "bucket_edges": list(BUCKET_EDGES),
            "torch_compile": TORCH_COMPILE,
            "cpu_smoke": CPU_SMOKE,
        },
        "dataset": meta,
    }
    write_json(artifact_dir / f"run_contract_{run_stamp}.json", run_contract)
    if REMOTE_RESULT_DIR is not None:
        write_json(REMOTE_RESULT_DIR / "run_contract.json", run_contract)

    autocast_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if not CPU_SMOKE
                    else torch.amp.autocast(device_type="cpu", enabled=False))

    # The fixed train probe batches are also the first training batches, so
    # freezing the probe set does not skip or consume extra training data.
    training_stream = itertools.chain(fixed_train_probe_batches, train_loader)
    x, y, epoch = batch_to_device(next(training_stream), device)

    smooth_train_loss = 0.0
    total_training_time = 0.0
    t_start = time.time()

    saved_model_ckpts: list = []

    for step in range(start_step, MAX_TRAINING_STEPS):
        completed = step + 1
        if not CPU_SMOKE:
            torch.cuda.synchronize()
        update_start = time.time()
        with autocast_ctx:
            if TRACE_ALL_BATCHES:
                train_token_losses, train_injection_stats = model(
                    x,
                    y,
                    return_token_losses=True,
                    return_injection_stats=True,
                )
                train_valid = y.ne(-1)
                loss = (
                    train_token_losses[train_valid].mean()
                    if train_valid.any()
                    else train_token_losses.sum() * 0.0
                )
            else:
                loss = model(x, y)
        train_loss_val = loss.detach()
        if ddp:
            torch.distributed.all_reduce(
                train_loss_val, op=torch.distributed.ReduceOp.AVG
            )
        if TRACE_ALL_BATCHES:
            write_batch_trace(
                trace_root,
                trace_manifest_paths,
                split="train",
                step=completed,
                batch_index=completed - 1,
                loader_epoch=epoch,
                targets=y,
                token_losses=train_token_losses,
                injection_stats=train_injection_stats,
                batch_hash=tensor_pair_sha256(x, y),
                compression=TRACE_COMPRESSION,
            )
        loss.backward()
        x, y, epoch = batch_to_device(next(training_stream), device)

        progress = min(step / MAX_TRAINING_STEPS, 1.0)
        lrm = lr_multiplier(step, MAX_TRAINING_STEPS, WARMUP_RATIO,
                            ADAM_WARMDOWN_RATIO, FINAL_LR_FRAC)
        current_lr = LEARNING_RATE * lrm
        if optimizer_groups:
            for group in adam_groups:
                group["lr"] = group["initial_lr"] * lrm
            adam_warmdown_start = 1.0 - ADAM_WARMDOWN_RATIO
            if progress >= adam_warmdown_start:
                adam_frac = (progress - adam_warmdown_start) / ADAM_WARMDOWN_RATIO
                beta1 = ADAM_BETAS[0] + (DEMON_FINAL_BETA1 - ADAM_BETAS[0]) * adam_frac
                for group in adam_demon_groups:
                    group["betas"] = (beta1, group["betas"][1])
            muon_frac = max(
                0.0,
                (progress - (1.0 - MUON_WARMDOWN_RATIO)) / MUON_WARMDOWN_RATIO,
            )
            late_frac = max(0.0, (muon_frac - 0.7) / 0.3)
            if late_frac > 0.0:
                ve_beta2 = NGRAM_VE_BETAS[1] + late_frac * (
                    NGRAM_VE_BETA2_WARMDOWN - NGRAM_VE_BETAS[1]
                )
                for group in ngram_groups:
                    group["beta2"] = ve_beta2
            optimizer.step()
        else:
            optimizer.step(lr_mult=lrm)
        optimizer.zero_grad(set_to_none=True)

        train_loss_f = float(train_loss_val.item())
        if not math.isfinite(train_loss_f) or train_loss_f > 100:
            raise RuntimeError(f"non-finite/exploding training loss: {train_loss_f}")

        if not CPU_SMOKE:
            torch.cuda.synchronize()
        dt = time.time() - update_start
        if step > 10:
            total_training_time += dt
        smooth_train_loss = 0.9 * smooth_train_loss + 0.1 * train_loss_f
        debiased = smooth_train_loss / (1.0 - 0.9 ** completed)
        append_jsonl(train_loss_paths, {
            "step": completed,
            "measurement_phase": "pre_update_loss_post_update_log",
            "updates_completed": completed,
            "train_loss": train_loss_f,
            "smoothed_train_loss": debiased,
            "learning_rate": current_lr,
            "lr_multiplier": lrm,
            "loader_epoch": epoch,
            "total_batch_tokens": TOTAL_BATCH_SIZE,
            "update_seconds": dt,
        })

        last_val_loss = None
        do_validation = (
            (TRACE_ALL_BATCHES or completed % VAL_LOSS_INTERVAL_STEPS == 0)
            and is_main
            and (not resumed or completed > start_step)
        )
        if do_validation:
            model.eval()
            with torch.no_grad():
                validation_loss_sum = 0.0
                validation_token_count = 0
                validation_records = []
                for validation_batch_index, (vx, vy, vepoch) in enumerate(fixed_val_batches):
                    with autocast_ctx:
                        if TRACE_ALL_BATCHES:
                            val_token_losses, val_injection_stats = model(
                                vx,
                                vy,
                                return_token_losses=True,
                                return_injection_stats=True,
                            )
                        else:
                            val_loss = model(vx, vy)
                            val_token_losses = None
                        if TRACE_ALL_BATCHES:
                            valid = vy.ne(-1)
                            validation_loss_sum += float(
                                val_token_losses[valid].sum().item()
                            )
                            validation_token_count += int(valid.sum().item())
                            validation_records.append(
                                write_batch_trace(
                                    trace_root,
                                    trace_manifest_paths,
                                    split="val",
                                    step=completed,
                                    batch_index=validation_batch_index,
                                    loader_epoch=vepoch,
                                    targets=vy,
                                    token_losses=val_token_losses,
                                    injection_stats=val_injection_stats,
                                    batch_hash=tensor_pair_sha256(vx, vy),
                                    compression=TRACE_COMPRESSION,
                                )
                            )
                        else:
                            validation_loss_sum += float(val_loss.item())
                            validation_token_count += 1
            model.train()
            if TRACE_ALL_BATCHES:
                last_val_loss = (
                    validation_loss_sum / validation_token_count
                    if validation_token_count
                    else 0.0
                )
            else:
                last_val_loss = validation_loss_sum / VAL_LOSS_BATCHES
            append_jsonl(val_loss_paths, {
                "step": completed,
                "val_loss": last_val_loss,
                "val_epoch": int(fixed_val_batches[0][2]),
                "val_batch_hashes": fixed_val_hashes,
                "validation_batch_count": len(fixed_val_batches),
                "valid_token_count": validation_token_count,
            })
            print(f"\n[val_loss] step {completed:05d} | loss: {last_val_loss:.6f} | epoch: {epoch}",
                  flush=True)

        # ---- frequency-bucketed gap probe (rank 0 only) ----
        if (completed in PROBE_STEPS and is_main
                and (not resumed or completed > start_step)):
            tr_recs = frequency_gap_probe(
                model, fixed_train_probe_batches,
                bucket_edges=BUCKET_EDGES, index=freq_index,
                step=completed, epoch=epoch, is_train=True, device=device,
                order=order, frequency_mode=PROBE_FREQUENCY_MODE,
                detail_dirs=detail_dirs,
                detail_manifest_paths=detail_manifest_paths)
            va_recs = frequency_gap_probe(
                model, fixed_val_probe_batches,
                bucket_edges=BUCKET_EDGES, index=freq_index,
                step=completed, epoch=epoch, is_train=False, device=device,
                order=order, frequency_mode=PROBE_FREQUENCY_MODE,
                detail_dirs=detail_dirs,
                detail_manifest_paths=detail_manifest_paths)
            merged = merge_train_val_gap(tr_recs, va_recs)
            for rec in merged:
                append_jsonl(decomp_paths, rec)
            print(f"[probe] step {completed}: wrote {len(merged)} bucket records",
                  flush=True)

        # ---- checkpoint ----
        if is_main and CKPT_DIR and completed > start_step:
            ckpt_dir = Path(CKPT_DIR)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            do_full = (
                completed in CKPT_STEPS
                or completed == MAX_TRAINING_STEPS
            )
            if do_full or completed % CKPT_INTERVAL_STEPS == 0:
                save_model = model.module if ddp else model
                ckpt_path = ckpt_dir / f"step_{completed:08d}.pt"
                payload = {
                    "model": save_model.state_dict(),
                    "step": completed,
                    "epoch": epoch,
                    "train_loss": train_loss_f,
                    "val_loss": last_val_loss,
                }
                if do_full:
                    payload["optimizer"] = optimizer.state_dict()
                    payload["has_optimizer"] = True
                else:
                    payload["has_optimizer"] = False
                    saved_model_ckpts.append(ckpt_path)
                    while len(saved_model_ckpts) > CKPT_KEEP_MODEL_ONLY:
                        stale = saved_model_ckpts.pop(0)
                        try:
                            stale.unlink()
                        except FileNotFoundError:
                            pass
                torch.save(payload, ckpt_path)
                print(f"\n[ckpt] step {completed}: saved {ckpt_path} "
                      f"(optimizer={'yes' if do_full else 'no'})", flush=True)

        val_txt = f" | val_loss: {last_val_loss:.6f}" if last_val_loss is not None else ""
        print(f"\rstep {completed:05d}/{MAX_TRAINING_STEPS} | loss: {debiased:.6f}"
              f"{val_txt} | lr: {current_lr:.3e} | dt: {dt*1000:.0f}ms | epoch: {epoch}",
              end="", flush=True)

        if step == 0:
            gc.collect()
            if not CPU_SMOKE:
                gc.freeze()
                gc.disable()

    if is_main:
        print()
    # final probe if not already done (rank 0 only)
    if (is_main and MAX_TRAINING_STEPS not in PROBE_STEPS
            and (not resumed or MAX_TRAINING_STEPS > start_step)):
        tr_recs = frequency_gap_probe(
            model, fixed_train_probe_batches,
            bucket_edges=BUCKET_EDGES, index=freq_index,
            step=MAX_TRAINING_STEPS, epoch=epoch, is_train=True, device=device,
            order=order, frequency_mode=PROBE_FREQUENCY_MODE,
            detail_dirs=detail_dirs,
            detail_manifest_paths=detail_manifest_paths)
        va_recs = frequency_gap_probe(
            model, fixed_val_probe_batches,
            bucket_edges=BUCKET_EDGES, index=freq_index,
            step=MAX_TRAINING_STEPS, epoch=epoch, is_train=False, device=device,
            order=order, frequency_mode=PROBE_FREQUENCY_MODE,
            detail_dirs=detail_dirs,
            detail_manifest_paths=detail_manifest_paths)
        merged = merge_train_val_gap(tr_recs, va_recs)
        for rec in merged:
            append_jsonl(decomp_paths, rec)

    if is_main:
        summary = {
            "schema_version": 1,
            "control": "ngram5_freq_gap",
            "status": "complete",
            "num_steps": MAX_TRAINING_STEPS,
            "total_tokens": MAX_TRAINING_STEPS * TOTAL_BATCH_SIZE,
            "training_seconds": total_training_time,
            "total_seconds": time.time() - t_start,
            "num_params": param_counts.get("total"),
            "run_contract": run_contract,
        }
        if not CPU_SMOKE:
            summary["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        write_json(artifact_dir / f"summary_{run_stamp}.json", summary)
        if REMOTE_RESULT_DIR is not None:
            write_json(REMOTE_RESULT_DIR / "summary.json", summary)
        print("---")
        print(f"num_steps:    {MAX_TRAINING_STEPS}")
        print(f"num_params_M: {param_counts.get('total', 0) / 1e6:.3f}")
        print(f"total_tokens_M: {MAX_TRAINING_STEPS * TOTAL_BATCH_SIZE / 1e6:.3f}")

    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
