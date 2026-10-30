# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Evaluate SDM time series forecasting results on GIFT-Eval."""

from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.timeseries.gift_eval import GIFT_EVAL_URL
from benchmark.timeseries.model import MODEL_CONFIGS

BASELINE = "Seasonal_Naive"
REFERENCES = ("TimesFM-3", "TimesFM-2.5", "Chronos-2", BASELINE)
METRICS = {
    "eval_metrics/MASE[0.5]": "MASE",
    "eval_metrics/mean_weighted_sum_quantile_loss": "CRPS",
}

root = Path(__file__).parent.parent / "gift_eval_out"

dfs = []
for config in MODEL_CONFIGS.values():
    path = root / config.name / "all_results.csv"
    if path.exists():
        dfs.append(pd.read_csv(path))
datasets = set.intersection(*(set(df["dataset"]) for df in dfs))

for name in REFERENCES:
    url = f"{GIFT_EVAL_URL}/results/{name}/all_results.csv"
    dfs.append(pd.read_csv(url))

df = pd.concat(dfs)
df = df[df["dataset"].isin(datasets)].set_index(["model", "dataset"])
df = df[list(METRICS)].rename(columns=METRICS)

terms = df.index.get_level_values("dataset").str.split("/").str[-1]
scores = np.log(df.div(df.loc[BASELINE], level="dataset"))
for i, (group, score) in enumerate([("all", scores), *scores.groupby(terms)]):
    if i > 0:
        print()
    summary = np.exp(score.groupby("model").mean())
    summary = summary.sort_values("CRPS")
    print(f"GIFT-Eval leaderboard ({group}):")
    print(summary.reset_index().to_string(index=False, float_format="%.4f"))
