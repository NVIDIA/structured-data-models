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
parser.add_argument(
    "--dataset", action="append", help="Run only these datasets; repeatable."
)
parser.add_argument(
    "--seeds", type=int, default=SEED_NUM, help="TALENT seeds per dataset."
)
parser.add_argument(
    "--tune-threshold",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Tune the binary decision threshold on the validation split "
    "(TALENT default); --no-tune-threshold scores argmax predictions.",
)
parser.add_argument(
    "--output-dir", type=Path, default=BENCHMARK_DIR / "talent_out"
)
parser.add_argument(
    "--name", help="Result directory and method label (default: the model)."
)
parser.add_argument("--num-estimators", type=int)
parser.add_argument(
    "--checkpoint",
    type=Path,
    help="kumo-tabular classification checkpoint (or the only one).",
)
parser.add_argument(
    "--checkpoint-reg", type=Path, help="kumo-tabular regression checkpoint."
)
parser.add_argument("--size", choices=("small", "large"))
parser.add_argument(
    "--numerical-missing", choices=("dispatch", "nan", "mix", "impute")
)
args = parser.parse_args()
if args.model != "kumo-tabular" and (
    args.checkpoint
    or args.checkpoint_reg
    or args.size
    or args.numerical_missing
):
    parser.error(
        "checkpoint, size and missing options need --model kumo-tabular"
    )

register_sdm_method()
root = args.dataset_path.resolve()
datasets = (
    args.dataset
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
label = args.name or model.name
method = f"[SDM] {label}"
general: dict[str, object] = {
    "model": args.model,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_estimators": args.num_estimators or model.num_estimators,
}
if not args.tune_threshold:
    general["tune_threshold"] = False  # part of the cache key; default runs keep their key
checkpoints = {
    task: str(path.resolve())
    for task, path in (
        ("classification", args.checkpoint),
        ("regression", args.checkpoint_reg or args.checkpoint),
    )
    if path is not None
}
if checkpoints:
    general["checkpoints"] = checkpoints
if args.size:
    general["size"] = args.size
if args.numerical_missing:
    general["numerical_missing"] = args.numerical_missing
config = {"model": {}, "training": {"n_bins": 2}, "general": general}

failed = False
for dataset in datasets:
    path = args.output_dir / label / dataset / "result.json"
    cached = json.loads(path.read_text()) if path.is_file() else {}
    if (
        cached.get("status") in {"success", "unsupported"}
        and cached.get("config") == config
        and cached.get("seed_num") == args.seeds
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
            seed_num=args.seeds,
            tune=False,
            tune_threshold=args.tune_threshold,
            threshold_metric="f1",
        )
        record = {
            "status": "success",
            "dataset": dataset,
            "model": args.model,
            "method": method,
            "config": config,
            "seed_num": args.seeds,
            "result": result.to_dict(),
        }
    except UnsupportedDatasetError as error:
        record = {
            "status": "unsupported",
            "dataset": dataset,
            "model": args.model,
            "method": method,
            "config": config,
            "seed_num": args.seeds,
            "error": str(error),
        }
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        failed = True
        continue
    _write(path, record)

if failed:
    raise SystemExit(1)
