# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an SDM tabular model on TALENT."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import TALENT
import torch

from benchmark.tabular.talent.models import (
    MODEL_CONFIGS,
    UnsupportedDatasetError,
    register_sdm_method,
)

SEED_NUM = 15
BENCHMARK_DIR = Path(__file__).parent.parent


def _write(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model", choices=tuple(MODEL_CONFIGS), default="tabiclv2"
)
parser.add_argument("--dataset-path", type=Path, required=True)
parser.add_argument("--dataset")
parser.add_argument(
    "--output-dir", type=Path, default=BENCHMARK_DIR / "talent_out"
)
args = parser.parse_args()

register_sdm_method()
root = args.dataset_path.resolve()
datasets = (
    [args.dataset]
    if args.dataset
    else sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "info.json").is_file()
    )
)
if not datasets:
    raise FileNotFoundError(f"No TALENT datasets found under {root}")

model = MODEL_CONFIGS[args.model]
method = f"[SDM] {model.name}"
config = {
    "model": {},
    "training": {"n_bins": 2},
    "general": {
        "model": args.model,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_estimators": model.num_estimators,
    },
}

failed = False
for dataset in datasets:
    path = args.output_dir / args.model / dataset / "result.json"
    cached = json.loads(path.read_text()) if path.is_file() else {}
    if (
        cached.get("status") in {"success", "unsupported"}
        and cached.get("config") == config
        and cached.get("seed_num") == SEED_NUM
    ):
        if cached.get("method") != method:
            cached["method"] = method
            _write(path, cached)
        print(f"{dataset}: cached")
        continue

    print(f"{dataset}: running")
    record: dict[str, object]
    try:
        result = TALENT.run_from_dataset(
            "sdm",
            dataset,
            str(root),
            config=config,
            seed_num=SEED_NUM,
            tune=False,
            tune_threshold=True,
            threshold_metric="f1",
        )
        record = {
            "status": "success",
            "dataset": dataset,
            "model": args.model,
            "method": method,
            "config": config,
            "seed_num": SEED_NUM,
            "result": result.to_dict(),
        }
    except UnsupportedDatasetError as error:
        record = {
            "status": "unsupported",
            "dataset": dataset,
            "model": args.model,
            "method": method,
            "config": config,
            "seed_num": SEED_NUM,
            "error": str(error),
        }
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        failed = True
        continue
    _write(path, record)

if failed:
    raise SystemExit(1)
