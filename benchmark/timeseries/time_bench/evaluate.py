# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Evaluate SDM time series forecasting results on TIME."""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from benchmark.timeseries.model import MODEL_CONFIGS

BASELINE = "seasonal_naive"
REFERENCES = ("TimesFM-3", "TimesFM-2.5", "Chronos-2", BASELINE)
METRICS = ("MASE", "CRPS")

runs = {}
for config in MODEL_CONFIGS.values():
    path = Path(__file__).parent.parent / "time_bench_out" / config.name
    if any(path.glob("**/metrics.npz")):
        runs[f"[SDM] {config.name}"] = path

configs = set.intersection(
    *(
        {path.parent.relative_to(run) for path in run.rglob("metrics.npz")}
        for run in runs.values()
    )
)

path = snapshot_download(
    repo_id="Real-TSF/TIME-Output",
    repo_type="dataset",
    allow_patterns=[
        f"results/{name}/{config.as_posix()}/metrics.npz"
        for name in REFERENCES
        for config in sorted(configs)
    ],
)
gc.collect()
for name in REFERENCES:
    runs[name] = Path(path) / "results" / name

rows = []
for model, run in runs.items():
    for config in sorted(configs):
        with np.load(run / config / "metrics.npz") as metrics:
            rows.append(
                {
                    "model": model,
                    "dataset": config.as_posix(),
                    **{
                        metric: np.nanmean(metrics[metric])
                        for metric in METRICS
                    },
                }
            )

df = pd.DataFrame(rows).set_index(["model", "dataset"])
scores = np.log(df.div(df.loc[BASELINE], level="dataset"))
terms = scores.index.get_level_values("dataset").str.split("/").str[-1]
for i, (group, score) in enumerate([("all", scores), *scores.groupby(terms)]):
    if i > 0:
        print()
    summary = np.exp(score.groupby("model").mean()).sort_values("CRPS")
    print(f"TIME leaderboard ({group}):")
    print(summary.reset_index().to_string(index=False, float_format="%.4f"))
