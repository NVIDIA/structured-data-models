# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Run an SDM forecasting model on fev-bench."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import fev
import pandas as pd
import torch

from benchmark.forecasting.fev_bench import FEV_URL
from benchmark.forecasting.fev_bench.models import SDMModel
from benchmark.forecasting.model import MODEL_CONFIGS, add_model_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument(
        "--task",
        action="append",
        help="Run only the selected fev-bench task; repeat to combine.",
    )
    args = parser.parse_args()

    model_config = MODEL_CONFIGS[args.model]
    if args.max_context_size is not None:
        model_config = replace(
            model_config, max_context_size=args.max_context_size
        )
    result_path = (
        Path(__file__).parent.parent
        / "fev_bench_out"
        / model_config.name
        / "results.csv"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    completed = (
        set(pd.read_csv(result_path)["task_name"])
        if result_path.exists()
        else set()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SDMModel(
        model_config.factory(device),
        config=model_config,
        batch_size=args.batch_size,
    )

    # Mirrors fev's official `models/evaluate.py` runner:
    benchmark = fev.Benchmark.from_yaml(
        f"{FEV_URL}/benchmarks/fev_bench/tasks.yaml"
    )
    for task in benchmark.tasks:
        if args.task is not None and task.task_name not in args.task:
            continue
        if task.task_name in completed:
            continue

        predictions = model.fit_predict(task)
        summary = task.evaluation_summary(
            predictions,
            model_name=f"[SDM] {model_config.name}",
            training_time_s=model.training_time,
            inference_time_s=model.inference_time,
            trained_on_this_dataset=(
                task.dataset_config in model.trained_on_datasets
            ),
            extra_info={"model_kwargs": json.dumps(vars(args))},
        )
        row = pd.DataFrame([summary])
        row.to_csv(
            result_path,
            mode="a",
            header=not result_path.exists(),
            index=False,
        )
        print(row[["task_name", "test_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
