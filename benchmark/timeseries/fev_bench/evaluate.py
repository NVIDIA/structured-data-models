# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Evaluate SDM time series forecasting results on fev-bench."""

from pathlib import Path

import fev
import pandas as pd

from benchmark.timeseries.fev_bench import FEV_BENCH_URL
from benchmark.timeseries.model import MODEL_CONFIGS

BASELINE = "Seasonal Naive"
REFERENCES = ("timesfm-3", "timesfm-2_5", "chronos-2", "seasonal_naive")
METRICS = ("SQL", "MASE")


benchmark_dir = Path(__file__).parent.parent
result_root = benchmark_dir / "fev_bench_out"
output_root = benchmark_dir / "evals" / "fev_bench"

dfs = []
for config in MODEL_CONFIGS.values():
    path = result_root / config.name / "results.csv"
    if path.exists():
        dfs.append(pd.read_csv(path))
tasks = set.intersection(*(set(df["task_name"]) for df in dfs))

for name in REFERENCES:
    url = f"{FEV_BENCH_URL}/benchmarks/fev_bench/results/{name}.csv"
    dfs.append(pd.read_csv(url))

df = pd.concat(dfs)
df = df[df["task_name"].isin(tasks)]

for i, metric in enumerate(METRICS):
    if i > 0:
        print()
    summary = fev.leaderboard(df, metric, baseline_model=BASELINE)
    print(f"fev-bench leaderboard ({metric}, {len(tasks)} tasks):")
    print(summary[["skill_score", "win_rate"]].to_string(float_format="%.4f"))
