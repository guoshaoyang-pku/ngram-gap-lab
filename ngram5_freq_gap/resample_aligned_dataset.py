#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path


def integer_stream(path: Path, chunk_size: int = 8 * 1024 * 1024):
    carry = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            payload = carry + chunk
            fields = payload.split()
            if payload and not payload[-1:].isspace():
                carry = fields.pop() if fields else payload
            else:
                carry = b""
            for field in fields:
                yield int(field)
    if carry:
        yield int(carry)


def row_count(path: Path, row_stride: int) -> int:
    count = sum(1 for _ in integer_stream(path))
    if count % row_stride:
        raise ValueError(f"{path} has {count} tokens, not aligned to {row_stride}")
    return count // row_stride


def selected_indices(total_rows: int, target_rows: int, seed: int) -> list[int]:
    if target_rows <= 0 or target_rows > total_rows:
        raise ValueError(
            f"target rows {target_rows} must be in [1, {total_rows}]"
        )
    return sorted(random.Random(seed).sample(range(total_rows), target_rows))


def write_selected_rows(
    source: Path,
    destination: Path,
    row_stride: int,
    selected: set[int],
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    row_index = 0
    token_index = 0
    row: list[int] = []
    written_rows = 0
    with destination.open("w", encoding="utf-8") as handle:
        for token in integer_stream(source):
            row.append(token)
            token_index += 1
            if len(row) != row_stride:
                continue
            if row_index in selected:
                handle.write(" ".join(map(str, row)))
                handle.write(" ")
                written_rows += 1
            row.clear()
            row_index += 1
    if row:
        raise ValueError(f"{source} ended with a partial row")
    with destination.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    return written_rows


def indices_digest(indices: list[int]) -> str:
    payload = ",".join(map(str, indices)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def resample_split(
    source_dir: Path,
    output_dir: Path,
    split: str,
    row_stride: int,
    target_rows: int,
    seed: int,
) -> dict:
    source = source_dir / f"{split}_tokens.txt"
    total_rows = row_count(source, row_stride)
    indices = selected_indices(total_rows, target_rows, seed)
    written_rows = write_selected_rows(
        source,
        output_dir / f"{split}_tokens.txt",
        row_stride,
        set(indices),
    )
    if written_rows != target_rows:
        raise RuntimeError(
            f"{split}: wrote {written_rows} rows, expected {target_rows}"
        )
    return {
        "source_rows": total_rows,
        "selected_rows": written_rows,
        "seed": seed,
        "selected_row_indices_sha256": indices_digest(indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select complete aligned rows from a full controlled stream. "
            "The exact train-frequency index is copied unchanged."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=285 * 72)
    parser.add_argument("--val-rows", type=int, default=288)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--row-stride", type=int, default=None)
    args = parser.parse_args()

    source_meta = json.loads((args.source_dir / "meta.json").read_text())
    row_stride = args.row_stride or int(source_meta["loader_row_stride"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "meta.json",
        "metadata.json",
        "contexts.json",
        "exact_ngram_counts.npz",
        "fivegram_counts.npz",
    ):
        source = args.source_dir / name
        if source.exists():
            shutil.copy2(source, args.output_dir / name)

    splits = {
        "train": resample_split(
            args.source_dir,
            args.output_dir,
            "train",
            row_stride,
            args.train_rows,
            args.seed ^ 0x13579,
        ),
        "val": resample_split(
            args.source_dir,
            args.output_dir,
            "val",
            row_stride,
            args.val_rows,
            args.seed ^ 0x2468A,
        ),
    }
    meta_path = args.output_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update(
        {
            "schema_version": 3,
            "train_tokens": args.train_rows * row_stride,
            "val_tokens": args.val_rows * row_stride,
            "loader_rows_train": args.train_rows,
            "loader_rows_val": args.val_rows,
            "loader_row_stride": row_stride,
            "sample_resampling": {
                "method": "uniform_random_complete_row_subsample",
                "source_dataset": str(args.source_dir),
                "source_frequency_index_unchanged": True,
                "target_steps_per_epoch": args.train_rows // 72,
                "target_device_batch_rows": 72,
                "splits": splits,
            },
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    metadata_path = args.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "schema_version": 3,
            "train_tokens": args.train_rows * row_stride,
            "val_tokens": args.val_rows * row_stride,
            "loader_row_stride": row_stride,
            "frequency_index_scope": (
                "complete upstream train epoch before row subsampling"
            ),
            "sample_resampling": meta["sample_resampling"],
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(meta["sample_resampling"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()