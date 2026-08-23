"""Backward-compatible re-export.

The canonical copy now lives at ``code/gap_experiment.py`` so the mainline can
reuse it (pure functions, no torch dependency: epoch_indices,
epoch_reshuffle_indices, shuffle_buffer_stream, interleaved_replay_offsets,
ordered_replay_offsets, lr_multiplier).

This shim keeps ``from gap_experiment import ...`` working for trainer.py /
lib.py, which are synced to clusters as a self-contained package.
"""
from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

_canonical = _Path(__file__).resolve().parents[1] / "code" / "gap_experiment.py"
if not _canonical.exists():          # cluster-synced package without code/
    _canonical = _Path(__file__).resolve().parent / "_gap_experiment_vendored.py"

_spec = _ilu.spec_from_file_location("_gap_experiment_canonical", _canonical)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

globals().update({k: v for k, v in _mod.__dict__.items() if not k.startswith("__")})
