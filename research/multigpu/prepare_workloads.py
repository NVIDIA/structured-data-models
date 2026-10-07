# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare fixed real-data TRAIN/VAL arrays for multi-GPU comparisons."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_covtype
from sklearn.model_selection import train_test_split


def prepare(name: str, root: Path) -> None:
    """Download a real task and store disjoint reproducible split arrays."""
    fetch = fetch_covtype if name == "covertype" else fetch_california_housing
    data = fetch(data_home=str(root / "source_cache"))
    classification = name == "covertype"
    ids = np.arange(len(data.target), dtype=np.int64)
    train, val = train_test_split(
        ids,
        test_size=0.2,
        random_state=20261008,
        stratify=data.target if classification else None,
    )
    # A separate deterministic permutation makes all context sizes nested.
    train = np.random.default_rng(20261008).permutation(train)
    val = np.random.default_rng(20261009).permutation(val)
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        "x_train": data.data[train].astype(np.float32),
        "y_train": data.target[train].astype(
            np.int64 if classification else np.float32
        ),
        "x_val": data.data[val].astype(np.float32),
        "y_val": data.target[val].astype(
            np.int64 if classification else np.float32
        ),
        "train_ids": train,
        "val_ids": val,
    }
    records = {}
    for key, array in arrays.items():
        path = out / f"{key}.npy"
        np.save(path, array, allow_pickle=False)
        records[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    manifest = {
        "name": name,
        "task": "classification" if classification else "regression",
        "seed": 20261008,
        "train_rows": len(train),
        "val_rows": len(val),
        "feature_names": list(data.feature_names),
        "context_sizes": [
            n for n in [1024, 4096, 16384, 32768, 65536] if n <= len(train)
        ],
        "query_sizes": [n for n in [256, 2048, 4096, 16384] if n <= len(val)],
        "selection": (
            "First N stored rows; fixed nested prefixes; "
            "no fitted preprocessing"
        ),
        "split": (
            "Deterministic custom 80/20 TRAIN/VAL; "
            "dataset has no official held-out TEST"
        ),
        "inference_label_rule": (
            "Only y_train may enter model/recipe; "
            "y_val is for postprediction scoring"
        ),
        "source": "https://archive.ics.uci.edu/dataset/31/covertype"
        if classification
        else "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html",
        "license": "CC BY 4.0"
        if classification
        else (
            "Source California census housing data via scikit-learn/StatLib; "
            "distribution license unspecified in fetched metadata"
        ),
        "metrics": ["accuracy", "multiclass_log_loss", "macro_f1"]
        if classification
        else ["MAE", "RMSE", "R2"],
        "files": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "source_description.txt").write_text(data.DESCR)
    print(  # noqa: T201
        json.dumps({"path": str(out), "train": len(train), "val": len(val)}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=["covertype", "california_housing"], required=True
    )
    args = parser.parse_args()
    prepare(args.dataset, args.root)
