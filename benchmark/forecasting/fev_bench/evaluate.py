# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Evaluate SDM forecasting model results on fev-bench."""

from pathlib import Path

import fev
import pandas as pd

from benchmark.forecasting.fev_bench import FEV_URL
from benchmark.forecasting.model import MODEL_CONFIGS

BASELINE = "Seasonal Naive"
REFERENCES = ("timesfm-3", "timesfm-2_5", "chronos-2", "seasonal_naive")
METRICS = ("SQL", "MASE")


def main() -> None:
    benchmark_dir = Path(__file__).parent.parent
    result_root = benchmark_dir / "fev_bench_out"
    output_root = benchmark_dir / "evals" / "fev_bench"

    summaries = []
    for model_config in MODEL_CONFIGS.values():
        path = result_root / model_config.name / "results.csv"
        if path.exists():
            summaries.append(pd.read_csv(path))

    if not summaries:
        raise FileNotFoundError(
            f"No fev-bench results found under {result_root}"
        )

    # Compare with published results on tasks run by all models:
    tasks = set.intersection(*(set(df["task_name"]) for df in summaries))
    for name in REFERENCES:
        path = f"{FEV_URL}/benchmarks/fev_bench/results/{name}.csv"
        summaries.append(pd.read_csv(path))
    summaries = pd.concat(summaries)
    summaries = summaries[summaries["task_name"].isin(tasks)]

    output_root.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        leaderboard = fev.leaderboard(
            summaries,
            metric_column=metric,
            baseline_model=BASELINE,
        )
        leaderboard.to_csv(output_root / f"leaderboard_{metric}.csv")
        print(f"\nfev-bench leaderboard ({metric}, {len(tasks)} tasks):")
        print(
            leaderboard[["skill_score", "win_rate"]].to_string(
                float_format="%.4f"
            )
        )


if __name__ == "__main__":
    main()
