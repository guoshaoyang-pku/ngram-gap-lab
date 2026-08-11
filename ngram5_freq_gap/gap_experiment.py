"""Pure experiment controls shared by training code and CPU-only tests."""

from __future__ import annotations

import hashlib
import json
import random


LR_SCHEDULE_MODES = {"baseline", "constant", "shifted"}
REPLAY_ORDER_MODES = {"original", "reverse", "cyclic_offset", "seeded_permutation"}


def epoch_indices(num_documents: int, seed: int, epoch: int) -> list[int]:
    if num_documents <= 0:
        raise ValueError("num_documents must be positive")
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    indices = list(range(num_documents))
    random.Random(seed + epoch - 1).shuffle(indices)
    return indices


def epoch_reshuffle_indices(num_documents: int, seed: int, epoch: int) -> list[int]:
    """Keep the first pass canonical and reshuffle every repeated pass."""
    if num_documents <= 0:
        raise ValueError("num_documents must be positive")
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if epoch == 1:
        return list(range(num_documents))
    return epoch_indices(num_documents, seed, epoch)


def batch_fixed_shuffle_indices(num_batches: int, seed: int) -> list[int]:
    """A single packed-batch permutation reused for every epoch."""
    return epoch_indices(num_batches, seed, 1)


def batch_epoch_reshuffle_indices(num_batches: int, seed: int, epoch: int) -> list[int]:
    """A fresh packed-batch permutation for every epoch, including epoch 1."""
    return epoch_indices(num_batches, seed, epoch)


def replacement_indices(num_documents: int, batch_size: int, rng: random.Random) -> list[int]:
    if num_documents <= 0 or batch_size <= 0:
        raise ValueError("num_documents and batch_size must be positive")
    return [rng.randrange(num_documents) for _ in range(batch_size)]


def shuffle_buffer_stream(num_documents: int, buffer_size: int, rng: random.Random):
    """Yield a continuous low-variance random stream without epoch resets."""
    if num_documents <= 0 or buffer_size <= 0:
        raise ValueError("num_documents and buffer_size must be positive")
    active_size = min(buffer_size, num_documents)
    buffer = list(range(active_size))
    next_index = active_size % num_documents
    while True:
        slot = rng.randrange(active_size)
        selected = buffer[slot]
        buffer[slot] = next_index
        next_index = (next_index + 1) % num_documents
        yield selected


def interleaved_replay_offsets(new_steps: int, replay_steps: int) -> list[int]:
    """Select the most recent packed batches to replay after each new-data block."""
    if new_steps <= 0:
        raise ValueError("new_steps must be positive")
    if not 0 < replay_steps <= new_steps:
        raise ValueError("replay_steps must be in [1, new_steps]")
    return list(range(new_steps - replay_steps, new_steps))


def ordered_replay_offsets(
    num_cached_batches: int,
    replay_steps: int,
    *,
    order: str = "original",
    cyclic_offset: int = 0,
    seed: int = 0,
) -> dict[str, object]:
    """Return an audited ordering of the cached batches selected for replay.

    The selected set is identical to the existing replay contract: the most
    recent ``min(replay_steps, num_cached_batches)`` cached batches.  Only the
    traversal order changes.  The returned SHA-256 covers the exact ordered
    source offsets, so a run can record what was actually replayed without
    hashing or copying batch tensors.
    """
    if num_cached_batches < 0:
        raise ValueError("num_cached_batches must be non-negative")
    if replay_steps <= 0:
        raise ValueError("replay_steps must be positive")
    normalized_order = order.strip().lower()
    if normalized_order not in REPLAY_ORDER_MODES:
        raise ValueError(
            f"order must be one of {sorted(REPLAY_ORDER_MODES)}; got {order!r}"
        )

    actual_replay_steps = min(replay_steps, num_cached_batches)
    source_offsets = list(
        range(num_cached_batches - actual_replay_steps, num_cached_batches)
    )
    effective_offset = 0
    effective_seed = None
    if normalized_order == "reverse":
        source_offsets.reverse()
    elif normalized_order == "cyclic_offset" and source_offsets:
        effective_offset = cyclic_offset % len(source_offsets)
        source_offsets = (
            source_offsets[effective_offset:] + source_offsets[:effective_offset]
        )
    elif normalized_order == "seeded_permutation":
        effective_seed = int(seed)
        random.Random(effective_seed).shuffle(source_offsets)

    encoded_order = json.dumps(source_offsets, separators=(",", ":")).encode("ascii")
    return {
        "mode": normalized_order,
        "num_cached_batches": int(num_cached_batches),
        "replay_steps_requested": int(replay_steps),
        "actual_replay_steps": int(actual_replay_steps),
        "source_offsets": source_offsets,
        "source_order_sha256": hashlib.sha256(encoded_order).hexdigest(),
        "cyclic_offset_configured": int(cyclic_offset),
        "cyclic_offset_effective": int(effective_offset),
        "seed_configured": int(seed),
        "seed_effective": effective_seed,
    }


def replay_order_annotations(
    manifest: dict[str, object],
    replay_offset: int,
    source_offset: int,
) -> dict[str, object]:
    """Build per-batch metadata and verify it matches the audited order."""
    source_offsets = manifest.get("source_offsets")
    if not isinstance(source_offsets, list):
        raise ValueError("replay-order manifest is missing source_offsets")
    if not 0 <= replay_offset < len(source_offsets):
        raise ValueError("replay_offset is outside the replay-order manifest")
    if int(source_offsets[replay_offset]) != int(source_offset):
        raise ValueError("source_offset does not match the replay-order manifest")
    return {
        "interleaved_replay_offset": int(replay_offset),
        "interleaved_source_offset": int(source_offset),
        "interleaved_replay_order": manifest["mode"],
        "interleaved_replay_order_hash": manifest["source_order_sha256"],
        "interleaved_replay_order_size": manifest["actual_replay_steps"],
        "interleaved_replay_cyclic_offset": manifest["cyclic_offset_effective"],
        "interleaved_replay_order_seed": manifest["seed_effective"],
    }


def validate_exact_replay_control_contract(
    *,
    data_mode: str,
    replay_new_steps: int,
    replay_steps: int,
    source_pass_steps: int,
    replay_order: str,
    freeze_requested: bool,
    max_training_steps: int,
) -> int | None:
    """Validate controls that require one exact cached source pass and replay."""
    normalized_mode = data_mode.strip().lower()
    normalized_order = replay_order.strip().lower()
    if normalized_order not in REPLAY_ORDER_MODES:
        raise ValueError(
            f"replay_order must be one of {sorted(REPLAY_ORDER_MODES)}"
        )
    if normalized_order != "original" and normalized_mode != "interleaved_replay":
        raise ValueError("non-original replay order requires interleaved_replay")
    requires_exact_pass = freeze_requested or normalized_order != "original"
    if normalized_mode != "interleaved_replay" or not requires_exact_pass:
        return None
    if source_pass_steps <= 0:
        raise ValueError("exact cached replay controls require source_pass_steps > 0")
    if not replay_new_steps == replay_steps == source_pass_steps:
        raise ValueError(
            "exact cached replay controls require new == replay == source pass steps"
        )
    expected_boundary_step = source_pass_steps + 1
    if freeze_requested and max_training_steps < expected_boundary_step:
        raise ValueError("training must include at least one cached replay update")
    return expected_boundary_step


def interleaved_replay_windows(
    max_steps: int,
    new_steps: int,
    replay_steps: int,
    source_pass_steps: int = 0,
) -> list[dict[str, int]]:
    """Return 1-based replay and following-new windows for analysis."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    interleaved_replay_offsets(new_steps, replay_steps)
    if source_pass_steps < 0:
        raise ValueError("source_pass_steps must be non-negative")
    phases = []
    train_cursor = 0
    source_cursor = 0
    while train_cursor < max_steps:
        remaining_in_pass = (
            source_pass_steps - (source_cursor % source_pass_steps)
            if source_pass_steps
            else new_steps
        )
        actual_new_steps = min(new_steps, remaining_in_pass, max_steps - train_cursor)
        new_start = train_cursor + 1
        new_end = train_cursor + actual_new_steps
        train_cursor = new_end
        source_cursor += actual_new_steps
        actual_replay_steps = min(replay_steps, actual_new_steps, max_steps - train_cursor)
        replay_start = train_cursor + 1 if actual_replay_steps else 0
        replay_end = train_cursor + actual_replay_steps if actual_replay_steps else 0
        train_cursor = replay_end or train_cursor
        phases.append(
            {
                "new_start": new_start,
                "new_end": new_end,
                "replay_start": replay_start,
                "replay_end": replay_end,
            }
        )
    windows = []
    for index, phase in enumerate(phases):
        if not phase["replay_start"]:
            continue
        following = phases[index + 1] if index + 1 < len(phases) else None
        windows.append(
            {
                "cycle": index + 1,
                "replay_start": phase["replay_start"],
                "replay_end": phase["replay_end"],
                "new_start": following["new_start"] if following else 0,
                "new_end": following["new_end"] if following else 0,
            }
        )
    return windows


def virtual_epoch(emitted_documents: int, num_documents: int) -> int:
    if emitted_documents < 0 or num_documents <= 0:
        raise ValueError("emitted_documents must be non-negative and num_documents must be positive")
    return 1 + emitted_documents // num_documents


def dataset_equivalents(emitted_documents: int, num_documents: int) -> float:
    """Continuous sampling progress without implying an epoch boundary."""
    if emitted_documents < 0 or num_documents <= 0:
        raise ValueError("emitted_documents must be non-negative and num_documents must be positive")
    return emitted_documents / num_documents


def overlap_row_mapping(batch_size: int, overlap_fraction: float, rng: random.Random) -> list[tuple[int, int]]:
    """Map current row indices to distinct rows from the prior packed batch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= overlap_fraction <= 1.0:
        raise ValueError("overlap_fraction must be in [0, 1]")
    overlap_count = min(batch_size, int(round(batch_size * overlap_fraction)))
    targets = rng.sample(range(batch_size), k=overlap_count)
    sources = rng.sample(range(batch_size), k=overlap_count)
    return list(zip(targets, sources))


def lr_multiplier(
    *,
    step: int,
    max_steps: int,
    mode: str,
    baseline_warmdown_ratio: float,
    shifted_decay_start_step: int,
    final_lr_fraction: float,
    warmup_ratio: float = 0.0,
) -> float:
    """Return the LR multiplier for one optimizer family."""
    if mode not in LR_SCHEDULE_MODES:
        raise ValueError(f"Unknown LR schedule mode {mode!r}; expected one of {sorted(LR_SCHEDULE_MODES)}")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if not 0.0 <= final_lr_fraction <= 1.0:
        raise ValueError("final_lr_fraction must be in [0, 1]")
    if not 0.0 < baseline_warmdown_ratio <= 1.0:
        raise ValueError("baseline_warmdown_ratio must be in (0, 1]")

    clamped_step = min(max(step, 0), max_steps)
    warmup_steps = max_steps * warmup_ratio
    if warmup_steps > 0 and clamped_step < warmup_steps:
        return clamped_step / warmup_steps
    if mode == "constant":
        return 1.0

    if mode == "baseline":
        decay_start = max_steps * (1.0 - baseline_warmdown_ratio)
    else:
        if not 0 <= shifted_decay_start_step < max_steps:
            raise ValueError(
                f"shifted_decay_start_step must be in [0, {max_steps}) for shifted mode"
            )
        decay_start = float(shifted_decay_start_step)

    if clamped_step < decay_start:
        return 1.0
    decay_progress = (clamped_step - decay_start) / (max_steps - decay_start)
    return 1.0 + decay_progress * (final_lr_fraction - 1.0)
