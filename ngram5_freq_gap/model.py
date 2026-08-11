"""Model entry point for the 5-gram frequency-gap experiment.

This module provides a single ``GPT`` / ``GPTConfig`` / ``MODEL_PROVENANCE``
interface to the trainer.  It tries, in order:

  1. The cluster's ``NanoGPTOriginal`` (imported from the cluster ``train.py``
     when running on the cluster — the n-gram injection tables are live).
  2. The vanilla decoder-only Transformer from
     ``nanogpt_gap_vanilla_control.vanilla_model`` (for local CPU smoke tests
     without the full cluster codebase).

When the user supplies a new architecture, replace the body of this file with
that implementation; the trainer only depends on the three names above plus a
``num_scaling_params()`` method and an optional ``architecture_manifest()``.
"""

from __future__ import annotations

import hashlib
import os
import sys
import types
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Try to import the cluster's NanoGPTOriginal.  On the cluster, train.py is in
# the repo root and exposes NanoGPTOriginal at module scope.  We attempt the
# import in a way that does not crash if the cluster deps (pyarrow, rustbpe,
# the full train.py) are unavailable — we fall back to the vanilla model.
_CLUSTER_TRAIN_PY = _PROJECT_ROOT / "train.py"
_NGRAM_GAP_EXP = Path("/data3/guoshaoyang/ngram-gap-exp/train.py")

NanoGPTOriginal = None
NanoGPTMLPTrunk = None
_cluster_model_loaded = False
_cluster_model_source = None

for _candidate in (_CLUSTER_TRAIN_PY, _NGRAM_GAP_EXP):
    if _candidate.exists():
        try:
            # ``train.py`` is a script: importing the whole file also builds a
            # second model, optimizer and dataloaders, and can even enter its
            # training loop.  Execute only the definition-only prefix ending
            # immediately before the hyperparameter/setup section instead.
            source = _candidate.read_text(encoding="utf-8")
            marker = (
                "# ---------------------------------------------------------------------------\n"
                "# Hyperparameters (edit these directly, no CLI flags needed)\n"
                "# ---------------------------------------------------------------------------"
            )
            if marker not in source:
                raise RuntimeError(f"definition boundary not found in {_candidate}")
            definition_source = source.split(marker, 1)[0]

            # The prefix imports sibling modules (observable, lib, ...), so
            # make its directory importable.  Register the temporary module as
            # dataclasses resolve annotations through ``sys.modules``.
            _train_dir = str(_candidate.parent)
            if _train_dir not in sys.path:
                sys.path.insert(0, _train_dir)
            digest = hashlib.sha256(str(_candidate).encode()).hexdigest()[:12]
            module_name = f"_ngram5_cluster_defs_{digest}"
            _mod = types.ModuleType(module_name)
            _mod.__file__ = str(_candidate)
            _mod.__package__ = ""
            sys.modules[module_name] = _mod

            import builtins
            _orig_print = builtins.print
            builtins.print = lambda *a, **k: None
            try:
                exec(compile(definition_source, str(_candidate), "exec"), _mod.__dict__)
            finally:
                builtins.print = _orig_print
            if hasattr(_mod, "NanoGPTOriginal"):
                NanoGPTOriginal = _mod.NanoGPTOriginal
                NanoGPTMLPTrunk = getattr(_mod, "NanoGPTMLPTrunk", None)
                GPTConfig = getattr(_mod, "NanoGPTConfig", None) or getattr(_mod, "GPTConfig", None)
                _cluster_model_loaded = True
                _cluster_model_source = str(_candidate)
                break
        except Exception:
            if "module_name" in locals():
                sys.modules.pop(module_name, None)
            continue

# Trunk selection: "transformer" (default, the baseline) or "mlp" (position-wise
# MLP blocks; n-gram tables, embeddings, head, optimizer grouping unchanged).
TRUNK_VARIANT = os.environ.get("NGRAM5_TRUNK", "transformer").strip().lower()

if NanoGPTOriginal is not None:
    if TRUNK_VARIANT == "mlp":
        if NanoGPTMLPTrunk is None:
            raise RuntimeError(
                "NGRAM5_TRUNK=mlp requested but the cluster model definitions "
                f"at {_cluster_model_source} do not provide NanoGPTMLPTrunk"
            )
        GPT = NanoGPTMLPTrunk
        MODEL_PROVENANCE = {
            "source": _cluster_model_source,
            "source_description": "cluster NanoGPTMLPTrunk (position-wise MLP trunk; n-gram injection tables live)",
            "note": "loaded from the definition-only prefix of cluster train.py",
            "trunk": "mlp",
        }
    elif TRUNK_VARIANT in ("transformer", "baseline"):
        GPT = NanoGPTOriginal
        MODEL_PROVENANCE = {
            "source": _cluster_model_source,
            "source_description": "cluster NanoGPTOriginal with n-gram injection tables",
            "note": "loaded from the definition-only prefix of cluster train.py",
            "trunk": "transformer",
        }
    else:
        raise ValueError(f"unknown NGRAM5_TRUNK: {TRUNK_VARIANT!r}")
else:
    # Fallback: vanilla transformer for local CPU smoke tests.
    sys.path.insert(0, str(_PROJECT_ROOT / "nanogpt_gap_vanilla_control"))
    from vanilla_model import GPT, GPTConfig, MODEL_PROVENANCE  # noqa: E402,F401
    MODEL_PROVENANCE = {
        **MODEL_PROVENANCE,
        "note": "vanilla fallback (no n-gram tables); cluster NanoGPTOriginal unavailable",
        "trunk": TRUNK_VARIANT,
    }

__all__ = ["GPT", "GPTConfig", "MODEL_PROVENANCE", "NanoGPTOriginal", "NanoGPTMLPTrunk", "TRUNK_VARIANT"]
