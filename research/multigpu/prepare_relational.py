# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy only native RelBench DB, TRAIN and VAL with fixed nested row IDs."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def prepare(
    source: Path, root: Path, task: str, target: str, kind: str
) -> None:
    """Copy selected native splits and record fixed sample identities."""
    out = root / source.name
    paths = sorted((source / "db").glob("*.parquet")) + [
        source / "tasks" / task / f"{split}.parquet"
        for split in ["train", "val"]
    ]
    files = {}
    for path in paths:
        relative = path.relative_to(source)
        dest = out / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        metadata = pq.read_schema(path).metadata or {}
        files[str(relative)] = {
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "bytes": dest.stat().st_size,
            "rows": pq.ParquetFile(dest).metadata.num_rows,
            "columns": pq.read_schema(dest).names,
            "relbench_metadata": {
                k.decode(): json.loads(v)
                for k, v in metadata.items()
                if k != b"pandas"
            },
        }
    train_rows = files[f"tasks/{task}/train.parquet"]["rows"]
    val_rows = files[f"tasks/{task}/val.parquet"]["rows"]
    for split, rows in [("train", train_rows), ("val", val_rows)]:
        # Context is a single seeded permutation of TRAIN, shared by every arm.
        ids = (
            np.random.default_rng(20261008).permutation(rows)
            if split == "train"
            else np.arange(rows)
        )
        dest = out / f"{task}-{split}-ids.npy"
        np.save(dest, ids, allow_pickle=False)
        files[dest.name] = {
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "bytes": dest.stat().st_size,
        }
    manifest = {
        "dataset": source.name,
        "task": task,
        "task_type": kind,
        "target_column": target,
        "source_cache": str(source),
        "train_rows": train_rows,
        "val_rows": val_rows,
        "seed": 20261008,
        "neighbors": [16, 16],
        "temporal_strategy": "last",
        "context_sizes": [
            n for n in [1024, 4096, 16384, 32768, 65536] if n <= train_rows
        ],
        "query_sizes": sorted(
            {min(n, val_rows) for n in [256, 2048, 4096, 16384]}
        ),
        "selection": (
            "Use first N stored IDs into native split parquet; "
            "fixed identical IDs for all arms"
        ),
        "leakage_rules": [
            "No TEST copied or read",
            "Only TRAIN labels enter model",
            "Drop VAL target before sampling/predict",
            "Use task timestamps in temporal neighbor sampler",
            "No lag features constructed from VAL labels",
            "Fit preprocessing on selected TRAIN context only",
        ],
        "source": "https://relbench.stanford.edu/",
        "license": (
            "RelBench MIT code; underlying source dataset terms "
            "remain applicable"
        ),
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(  # noqa: T201
        json.dumps({"path": str(out), "train": train_rows, "val": val_rows}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--kind", choices=["classification", "regression"], required=True
    )
    args = parser.parse_args()
    prepare(args.source, args.root, args.task, args.target, args.kind)
