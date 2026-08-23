#!/usr/bin/env python3
"""Toy6: does fixed replay let a per-key lookup table memorize label noise?

This is a deliberately small binary-classification experiment.  It does not
import the OPHIS model, tokenizer, data pipeline, or optimizers.

Ground truth
------------
Each key has an observed base feature x in {-1, +1}.  The true conditional is

    P(y = 1 | key) = 1 - noise,  if x = +1
                     noise,      if x = -1.

Thus a two-parameter shared backbone, bias + weight * x, can represent the
entire population rule.  The optional lookup memory adds one scalar per key.
Its effective values are centered within each (frequency, sign) cell, so it is
an identifiable key-specific residual and cannot replace the shared base rule
or encode a frequency-level calibration offset.

Protocol
--------
Keys are assigned exact per-epoch sample counts r in configurable frequency
buckets.  ``fixed`` reuses one finite label sample every epoch; ``fresh``
draws new labels from the same P(y|key) each epoch.  Evaluation uses both an
independent, large sampled validation set and the exact population CE.

The script writes epoch/final records to ``metrics.jsonl`` and a deterministic
matrix/aggregate artifact to ``summary.json``.  Training uses per-key positive
counts (Bernoulli sufficient statistics), so it is fast and has no hidden
minibatch-order effect.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA_VERSION = 1


def _csv_ints(text: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers: {text}") from exc
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def _csv_floats(text: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated floats: {text}") from exc
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def _csv_choices(text: str, choices: set[str], label: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    bad = [item for item in values if item not in choices]
    if not values or bad:
        allowed = ",".join(sorted(choices))
        raise argparse.ArgumentTypeError(f"{label} must be a comma-list from {{{allowed}}}; bad={bad}")
    return values


def _unique(values: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def _noise_name(noise: float) -> str:
    return format(noise, ".8g").replace("-", "m").replace(".", "p")


def _derived_seed(seed: int, noise: float, purpose: int) -> int:
    """Stable seed mixer independent of Python's randomized hash()."""
    noise_code = int(round(noise * 1_000_000))
    modulus = (1 << 63) - 25
    return (seed * 1_000_003 + noise_code * 97_409 + purpose * 65_537 + 17) % modulus


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _dump_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        _json_ready(value),
        indent=indent,
        sort_keys=True,
        separators=None if indent else (",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    noise: float
    replay_mode: str
    memory: bool
    frequencies: tuple[int, ...]
    keys_per_frequency: int
    epochs: int
    val_samples_per_key: int
    backbone_lr: float
    memory_lr: float
    memory_l2: float
    dtype: str
    device: str
    log_every: int
    log_epochs: tuple[int, ...]

    @property
    def config_id(self) -> str:
        memory_name = "mem" if self.memory else "nomem"
        return f"s{self.seed}_noise{_noise_name(self.noise)}_{self.replay_mode}_{memory_name}"


class LookupReplayModel(nn.Module):
    """Two shared scalars plus an optional, centered per-key residual."""

    def __init__(self, signs: torch.Tensor, memory_groups: torch.Tensor, memory: bool) -> None:
        super().__init__()
        self.register_buffer("signs", signs)
        self.register_buffer("memory_groups", memory_groups)
        self.bias = nn.Parameter(torch.zeros((), dtype=signs.dtype, device=signs.device))
        self.weight = nn.Parameter(torch.zeros((), dtype=signs.dtype, device=signs.device))
        if memory:
            self.raw_memory = nn.Parameter(torch.zeros_like(signs))
        else:
            self.register_parameter("raw_memory", None)

    def effective_memory(self) -> torch.Tensor:
        if self.raw_memory is None:
            return torch.zeros_like(self.signs)
        # This projection makes the table an identifiable within-cell residual.
        # It also makes noise=0 a strict negative control: identical keys within
        # every (r, sign) cell cannot induce any effective lookup value.
        centered = self.raw_memory.clone()
        for group in torch.unique(self.memory_groups, sorted=True):
            mask = self.memory_groups == group
            centered[mask] = centered[mask] - centered[mask].mean()
        return centered

    def forward(self) -> torch.Tensor:
        return self.bias + self.weight * self.signs + self.effective_memory()


def _make_key_spec(
    frequencies: Sequence[int], keys_per_frequency: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frequencies_per_key: list[int] = []
    signs: list[float] = []
    memory_groups: list[int] = []
    half = keys_per_frequency // 2
    for frequency_index, frequency in enumerate(frequencies):
        frequencies_per_key.extend([frequency] * keys_per_frequency)
        signs.extend([-1.0] * half)
        signs.extend([+1.0] * half)
        memory_groups.extend([2 * frequency_index] * half)
        memory_groups.extend([2 * frequency_index + 1] * half)
    return (
        torch.tensor(frequencies_per_key, dtype=torch.int64),
        torch.tensor(signs, dtype=dtype),
        torch.tensor(memory_groups, dtype=torch.int64),
    )


def _true_probabilities(signs: torch.Tensor, noise: float) -> torch.Tensor:
    return torch.where(
        signs > 0,
        torch.full_like(signs, 1.0 - noise),
        torch.full_like(signs, noise),
    )


def _sample_positive_counts(
    totals: torch.Tensor, probabilities: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Draw independent Bernoulli samples and return positives per key."""
    counts = torch.empty_like(probabilities)
    for total in torch.unique(totals, sorted=True).tolist():
        mask = totals == total
        probs = probabilities[mask]
        draws = torch.rand((int(mask.sum()), int(total)), generator=generator)
        counts[mask] = (draws < probs[:, None].cpu()).sum(dim=1).to(probabilities.dtype)
    return counts


def _loss_sums_per_key(logits: torch.Tensor, positives: torch.Tensor, totals: torch.Tensor) -> torch.Tensor:
    positive_loss = F.softplus(-logits)
    negative_loss = F.softplus(logits)
    return positives * positive_loss + (totals - positives) * negative_loss


def _sample_ce(logits: torch.Tensor, positives: torch.Tensor, totals: torch.Tensor) -> torch.Tensor:
    return _loss_sums_per_key(logits, positives, totals).sum() / totals.sum()


def _weighted_sample_ce(
    logits: torch.Tensor,
    positives: torch.Tensor,
    totals: torch.Tensor,
    key_weights: torch.Tensor,
) -> torch.Tensor:
    per_key_ce = _loss_sums_per_key(logits, positives, totals) / totals
    return (per_key_ce * key_weights).sum() / key_weights.sum()


def _population_ce(
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    key_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    per_key_ce = probabilities * F.softplus(-logits) + (1.0 - probabilities) * F.softplus(logits)
    if key_weights is None:
        return per_key_ce.mean()
    return (per_key_ce * key_weights).sum() / key_weights.sum()


def _bayes_ce(noise: float) -> float:
    if noise == 0.0:
        return 0.0
    return float(-(1.0 - noise) * math.log(1.0 - noise) - noise * math.log(noise))


def _float(tensor: torch.Tensor) -> float:
    return float(tensor.detach().cpu().item())


@torch.no_grad()
def _measure(
    model: LookupReplayModel,
    train_positives: torch.Tensor,
    train_totals: torch.Tensor,
    val_positives: torch.Tensor,
    val_totals: torch.Tensor,
    probabilities: torch.Tensor,
    frequencies_per_key: torch.Tensor,
    frequencies: Sequence[int],
) -> dict[str, Any]:
    logits = model()
    memory = model.effective_memory()
    train_ce = _sample_ce(logits, train_positives, train_totals)
    # Validation draws the same number of examples for every key to make every
    # per-r estimate precise.  Reweight those estimates by r for the headline,
    # matching the training key marginal exactly.
    val_ce = _weighted_sample_ce(
        logits, val_positives, val_totals, key_weights=train_totals
    )
    population_ce = _population_ce(logits, probabilities, key_weights=train_totals)
    per_r: list[dict[str, Any]] = []
    for frequency in frequencies:
        mask_cpu = frequencies_per_key == frequency
        mask = mask_cpu.to(logits.device)
        r_train_totals = train_totals[mask]
        r_val_totals = val_totals[mask]
        r_train_ce = _sample_ce(logits[mask], train_positives[mask], r_train_totals)
        r_val_ce = _sample_ce(logits[mask], val_positives[mask], r_val_totals)
        r_population_ce = _population_ce(logits[mask], probabilities[mask])
        r_memory = memory[mask]
        per_r.append(
            {
                "r": int(frequency),
                "n_keys": int(mask_cpu.sum()),
                "n_train_samples": int(r_train_totals.sum().detach().cpu()),
                "n_val_samples": int(r_val_totals.sum().detach().cpu()),
                "train_ce": _float(r_train_ce),
                "val_ce": _float(r_val_ce),
                "gap": _float(r_val_ce - r_train_ce),
                "population_ce": _float(r_population_ce),
                "population_gap": _float(r_population_ce - r_train_ce),
                "table_rms": _float(torch.sqrt(torch.mean(r_memory.square()))),
                "table_abs_mean": _float(torch.mean(torch.abs(r_memory))),
            }
        )
    return {
        "train_ce": _float(train_ce),
        "val_ce": _float(val_ce),
        "gap": _float(val_ce - train_ce),
        "population_ce": _float(population_ce),
        "population_gap": _float(population_ce - train_ce),
        "backbone_bias": _float(model.bias),
        "backbone_weight": _float(model.weight),
        "table_l2": _float(torch.linalg.vector_norm(memory)),
        "table_rms": _float(torch.sqrt(torch.mean(memory.square()))),
        "table_abs_mean": _float(torch.mean(torch.abs(memory))),
        "per_r": per_r,
    }


def _build_optimizer(model: LookupReplayModel, config: ExperimentConfig) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = [
        {"params": [model.bias, model.weight], "lr": config.backbone_lr}
    ]
    if model.raw_memory is not None:
        groups.append({"params": [model.raw_memory], "lr": config.memory_lr})
    return torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8)


def _record(
    config: ExperimentConfig,
    epoch: int,
    metrics: dict[str, Any],
    *,
    final: bool,
) -> dict[str, Any]:
    return {
        "record_type": "final" if final else "epoch",
        "schema_version": SCHEMA_VERSION,
        "config_id": config.config_id,
        "seed": config.seed,
        "noise": config.noise,
        "replay_mode": config.replay_mode,
        "memory": config.memory,
        "epoch": epoch,
        "bayes_ce": _bayes_ce(config.noise),
        **metrics,
    }


def run_experiment(config: ExperimentConfig, jsonl_file: Any) -> dict[str, Any]:
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    device = torch.device(config.device)
    frequencies_cpu, signs_cpu, memory_groups_cpu = _make_key_spec(
        config.frequencies, config.keys_per_frequency, dtype
    )
    probabilities_cpu = _true_probabilities(signs_cpu, config.noise)
    train_totals_cpu = frequencies_cpu.clone()
    val_totals_cpu = torch.full_like(frequencies_cpu, config.val_samples_per_key)

    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(_derived_seed(config.seed, config.noise, purpose=1))
    val_generator = torch.Generator(device="cpu")
    val_generator.manual_seed(_derived_seed(config.seed, config.noise, purpose=2))
    initial_train_positives_cpu = _sample_positive_counts(
        train_totals_cpu, probabilities_cpu, train_generator
    )
    val_positives_cpu = _sample_positive_counts(
        val_totals_cpu, probabilities_cpu, val_generator
    )

    signs = signs_cpu.to(device)
    memory_groups = memory_groups_cpu.to(device)
    probabilities = probabilities_cpu.to(device)
    train_totals = train_totals_cpu.to(device=device, dtype=dtype)
    val_totals = val_totals_cpu.to(device=device, dtype=dtype)
    val_positives = val_positives_cpu.to(device)
    train_positives = initial_train_positives_cpu.to(device)

    model = LookupReplayModel(signs=signs, memory_groups=memory_groups, memory=config.memory)
    optimizer = _build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=0.0
    )

    initial_metrics = _measure(
        model,
        train_positives,
        train_totals,
        val_positives,
        val_totals,
        probabilities,
        frequencies_cpu,
        config.frequencies,
    )
    jsonl_file.write(_dump_json(_record(config, 0, initial_metrics, final=False)) + "\n")
    jsonl_file.flush()

    for epoch in range(1, config.epochs + 1):
        if config.replay_mode == "fresh" and epoch > 1:
            train_positives = _sample_positive_counts(
                train_totals_cpu, probabilities_cpu, train_generator
            ).to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model()
        objective = _sample_ce(logits, train_positives, train_totals)
        if model.raw_memory is not None and config.memory_l2 > 0.0:
            objective = objective + config.memory_l2 * model.effective_memory().square().mean()
        objective.backward()
        optimizer.step()
        scheduler.step()

        should_log = (
            (config.log_every > 0 and epoch % config.log_every == 0)
            or epoch in config.log_epochs
            or epoch == config.epochs
        )
        if should_log:
            metrics = _measure(
                model,
                train_positives,
                train_totals,
                val_positives,
                val_totals,
                probabilities,
                frequencies_cpu,
                config.frequencies,
            )
            jsonl_file.write(
                _dump_json(_record(config, epoch, metrics, final=epoch == config.epochs))
                + "\n"
            )
            jsonl_file.flush()

    final_metrics = _measure(
        model,
        train_positives,
        train_totals,
        val_positives,
        val_totals,
        probabilities,
        frequencies_cpu,
        config.frequencies,
    )
    return _record(config, config.epochs, final_metrics, final=True)


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)),
    }


def _aggregate(final_records: Sequence[dict[str, Any]], frequencies: Sequence[int]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, str, bool], list[dict[str, Any]]] = {}
    for record in final_records:
        key = (record["noise"], record["replay_mode"], record["memory"])
        groups.setdefault(key, []).append(record)

    aggregates: list[dict[str, Any]] = []
    for (noise, replay_mode, memory), records in groups.items():
        records = sorted(records, key=lambda item: item["seed"])
        scalar_names = [
            "train_ce",
            "val_ce",
            "gap",
            "population_ce",
            "population_gap",
            "backbone_bias",
            "backbone_weight",
            "table_l2",
            "table_rms",
            "table_abs_mean",
        ]
        aggregate: dict[str, Any] = {
            "noise": noise,
            "replay_mode": replay_mode,
            "memory": memory,
            "n_seeds": len(records),
            "seeds": [record["seed"] for record in records],
            "metrics": {
                name: _mean_std([record[name] for record in records]) for name in scalar_names
            },
            "per_r": [],
        }
        per_seed_per_r = [
            {entry["r"]: entry for entry in record["per_r"]} for record in records
        ]
        for frequency in frequencies:
            entries = [per_r[frequency] for per_r in per_seed_per_r]
            aggregate["per_r"].append(
                {
                    "r": frequency,
                    "gap": _mean_std([entry["gap"] for entry in entries]),
                    "population_gap": _mean_std(
                        [entry["population_gap"] for entry in entries]
                    ),
                    "train_ce": _mean_std([entry["train_ce"] for entry in entries]),
                    "val_ce": _mean_std([entry["val_ce"] for entry in entries]),
                    "table_rms": _mean_std([entry["table_rms"] for entry in entries]),
                }
            )
        aggregates.append(aggregate)
    return aggregates


def _validate_args(args: argparse.Namespace) -> None:
    if args.keys_per_frequency <= 0 or args.keys_per_frequency % 2:
        raise SystemExit("--keys-per-frequency must be a positive even integer for sign balance")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.val_samples_per_key <= 0:
        raise SystemExit("--val-samples-per-key must be positive")
    if args.log_every < 0:
        raise SystemExit("--log-every must be non-negative (0 disables periodic logging)")
    if any(epoch <= 0 for epoch in args.log_epochs):
        raise SystemExit("all --log-epochs must be positive")
    if args.threads <= 0:
        raise SystemExit("--threads must be positive")
    if any(frequency <= 0 for frequency in args.frequencies):
        raise SystemExit("all --frequencies must be positive")
    if any(not 0.0 <= noise < 0.5 for noise in args.noise_values):
        raise SystemExit("all --noise-values must satisfy 0 <= noise < 0.5")
    if args.backbone_lr <= 0.0 or args.memory_lr <= 0.0:
        raise SystemExit("learning rates must be positive")
    if args.memory_l2 < 0.0:
        raise SystemExit("--memory-l2 must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"requested {args.device}, but CUDA is unavailable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the independent Toy6 lookup-memory x replay matrix."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("toy/results/toy6_lookup_replay"))
    parser.add_argument(
        "--replay-modes",
        type=lambda text: _csv_choices(text, {"fixed", "fresh"}, "replay modes"),
        default=["fixed", "fresh"],
        help="comma-list: fixed,fresh",
    )
    parser.add_argument(
        "--memory-modes",
        type=lambda text: _csv_choices(text, {"on", "off"}, "memory modes"),
        default=["off", "on"],
        help="comma-list: off,on",
    )
    parser.add_argument("--noise-values", type=_csv_floats, default=[0.0, 0.2])
    parser.add_argument("--seeds", type=_csv_ints, default=[0])
    parser.add_argument(
        "--frequencies",
        type=_csv_ints,
        default=[1, 2, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument("--keys-per-frequency", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--val-samples-per-key", type=int, default=512)
    parser.add_argument("--backbone-lr", type=float, default=0.05)
    parser.add_argument("--memory-lr", type=float, default=0.08)
    parser.add_argument(
        "--memory-l2",
        type=float,
        default=0.0,
        help="optional mean-square penalty on the effective lookup residual",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="periodic epoch interval; 0 disables it (explicit log epochs still apply)",
    )
    parser.add_argument(
        "--log-epochs",
        type=_csv_ints,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 300],
        help="replay-dose readouts from one run; values beyond --epochs are ignored",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace metrics.jsonl and summary.json if they already exist",
    )
    args = parser.parse_args()
    args.replay_modes = _unique(args.replay_modes)
    args.memory_modes = _unique(args.memory_modes)
    args.noise_values = _unique(args.noise_values)
    args.seeds = _unique(args.seeds)
    args.frequencies = _unique(args.frequencies)
    args.log_epochs = sorted(_unique(args.log_epochs))
    return args


def main() -> None:
    args = parse_args()
    _validate_args(args)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    summary_path = args.output_dir / "summary.json"
    existing = [path for path in (metrics_path, summary_path) if path.exists()]
    if existing and not args.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise SystemExit(f"refusing to overwrite existing output(s): {paths}; pass --overwrite")

    configs: list[ExperimentConfig] = []
    for seed in args.seeds:
        for noise in args.noise_values:
            for replay_mode in args.replay_modes:
                for memory_mode in args.memory_modes:
                    configs.append(
                        ExperimentConfig(
                            seed=seed,
                            noise=noise,
                            replay_mode=replay_mode,
                            memory=memory_mode == "on",
                            frequencies=tuple(args.frequencies),
                            keys_per_frequency=args.keys_per_frequency,
                            epochs=args.epochs,
                            val_samples_per_key=args.val_samples_per_key,
                            backbone_lr=args.backbone_lr,
                            memory_lr=args.memory_lr,
                            memory_l2=args.memory_l2,
                            dtype=args.dtype,
                            device=args.device,
                            log_every=args.log_every,
                            log_epochs=tuple(args.log_epochs),
                        )
                    )

    final_records: list[dict[str, Any]] = []
    with metrics_path.open("w", encoding="utf-8") as jsonl_file:
        for index, config in enumerate(configs, start=1):
            print(f"[{index}/{len(configs)}] {config.config_id}", flush=True)
            final_record = run_experiment(config, jsonl_file)
            final_records.append(final_record)
            print(
                "  "
                f"train={final_record['train_ce']:.6f} "
                f"val={final_record['val_ce']:.6f} "
                f"gap={final_record['gap']:+.6f} "
                f"table_rms={final_record['table_rms']:.6f}",
                flush=True,
            )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "toy6_lookup_replay",
        "gap_definition": "sampled_validation_ce_minus_current_training_sample_ce",
        "population_gap_definition": "exact_population_ce_minus_current_training_sample_ce",
        "ground_truth": {
            "base_feature": "balanced per-frequency key sign x in {-1,+1}",
            "conditional": "P(y=1|key)=1-noise for x=+1, noise for x=-1",
            "backbone": "bias + weight*x (2 shared scalar parameters)",
            "memory": "optional one scalar per key, centered within each (frequency, sign) cell",
        },
        "protocol": {
            "fixed": "one finite per-key label sample replayed for every epoch",
            "fresh": "new independent labels from the same P(y|key) every epoch",
            "validation": (
                "one independent sampled set, val_samples_per_key for every key; "
                "headline CE is reweighted by r to match the training key marginal"
            ),
            "optimizer": "Adam with separate backbone/memory learning rates and cosine decay",
            "training_statistic": "per-key positive counts; exact Bernoulli sufficient statistic",
        },
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {"overwrite", "output_dir"}
        },
        "n_configs": len(configs),
        "configs": [asdict(config) | {"config_id": config.config_id} for config in configs],
        "final_records": final_records,
        "aggregates": _aggregate(final_records, args.frequencies),
        "artifacts": {"metrics_jsonl": "metrics.jsonl", "summary_json": "summary.json"},
    }
    summary_path.write_text(_dump_json(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {metrics_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
