# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Run an SDM time series forecasting model on fev-bench."""

import argparse
from pathlib import Path

import fev
import pandas as pd
import torch

from benchmark.timeseries.fev_bench import FEV_BENCH_URL
from benchmark.timeseries.fev_bench.predictor import SDMPredictor
from benchmark.timeseries.model import MODEL_CONFIGS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model",
    choices=tuple(MODEL_CONFIGS),
    default="timesfm3",
    help="SDM model to benchmark.",
)
parser.add_argument(
    "--mini",
    action="store_true",
    help="Run the 20-task fev-bench-mini subset.",
)
parser.add_argument(
    "--task",
    action="append",
    help="Run only the selected fev-bench task; repeat to combine.",
)
args = parser.parse_args()

config = MODEL_CONFIGS[args.model]
path = Path(__file__).parent.parent
path = path / "fev_bench_out" / config.name / "results.csv"
path.parent.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SDMPredictor(
    model=config.factory(device),
    config=config,
)

completed = set(pd.read_csv(path)["task_name"]) if path.exists() else set()
filename = "tasks_mini.yaml" if args.mini else "tasks.yaml"
benchmark = fev.Benchmark.from_yaml(
    f"{FEV_BENCH_URL}/benchmarks/fev_bench/{filename}"
)
for task in benchmark.tasks:
    if args.task is not None and task.task_name not in args.task:
        continue
    if task.task_name in completed:
        continue

    summary = task.evaluation_summary(
        model.fit_predict(task),
        model_name=f"[SDM] {config.name}",
        training_time_s=model.training_time,
        inference_time_s=model.inference_time,
    )
    print(
        f"{task.task_name}: Error={summary['test_error']:6f}, "
        f"Time={model.inference_time:.2f}s"
    )
    row = pd.DataFrame([summary])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)
    completed.add(task.task_name)
