# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Run an SDM time series forecasting model on GIFT-Eval."""

import argparse
import logging
import os
import warnings
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from gift_eval.data import Dataset
from gluonts.ev import metrics
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

from benchmark.timeseries.gift_eval import GIFT_EVAL_URL
from benchmark.timeseries.gift_eval.datasets import (
    DATASETS,
    MED_LONG_DATASETS,
    PRETTY_NAMES,
)
from benchmark.timeseries.model import MODEL_CONFIGS
from benchmark.timeseries.predictor import SDMPredictor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model",
    choices=tuple(MODEL_CONFIGS),
    default="timesfm3",
    help="SDM model to benchmark.",
)
parser.add_argument(
    "--dataset",
    action="append",
    help="Run only the selected GIFT-Eval dataset; repeat to combine.",
)
args = parser.parse_args()

warnings.filterwarnings("ignore", message=r"'.*' is deprecated.*")
logging.getLogger("gluonts.model.forecast_generator").addFilter(
    lambda r: not r.getMessage().startswith("The mean prediction")
)

METRICS = [
    metrics.MSE(forecast_type="mean"),
    metrics.MSE(forecast_type="0.5"),
    metrics.MAE(),
    metrics.MASE(),
    metrics.MAPE(),
    metrics.SMAPE(),
    metrics.MSIS(),
    metrics.RMSE(),
    metrics.NRMSE(),
    metrics.ND(),
    metrics.MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ),
]

config = MODEL_CONFIGS[args.model]
path = Path(__file__).parent.parent
path = path / "gift_eval_out" / config.name / "all_results.csv"
path.parent.mkdir(parents=True, exist_ok=True)

names = args.dataset or DATASETS
if not os.getenv("GIFT_EVAL"):
    os.environ["GIFT_EVAL"] = snapshot_download(
        repo_id="Salesforce/GiftEval",
        repo_type="dataset",
        allow_patterns=[f"{name}/*" for name in names],
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = config.factory(device)

completed = set(pd.read_csv(path)["dataset"]) if path.exists() else set()
properties = pd.read_json(f"{GIFT_EVAL_URL}/notebooks/dataset_properties.json")
for name in names:
    key, _, freq = name.partition("/")
    key = PRETTY_NAMES.get(key.lower(), key.lower())
    freq = freq or properties[key]["frequency"]
    for term in ("short", "medium", "long"):
        if term != "short" and name not in MED_LONG_DATASETS:
            continue
        dataset_id = f"{key}/{freq}/{term}"
        if dataset_id in completed:
            continue

        dataset = Dataset(name=name, term=term, to_univariate=False)
        predictor = SDMPredictor(model, config, dataset.prediction_length)
        with patch(
            "gluonts.model.evaluation.tqdm",
            partial(tqdm, total=len(dataset.test_data), desc=dataset_id),
        ):
            result = evaluate_model(
                predictor,
                test_data=dataset.test_data,
                metrics=METRICS,
                seasonality=get_seasonality(dataset.freq),
            )
        print(
            f"{dataset_id}: "
            f"MASE={result.iloc[0]['MASE[0.5]']:.4f}, "
            f"CRPS={result.iloc[0]['mean_weighted_sum_quantile_loss']:.4f}"
        )
        row = pd.DataFrame(
            [
                {
                    "dataset": dataset_id,
                    "model": f"[SDM] {config.name}",
                    **{
                        f"eval_metrics/{k}": v
                        for k, v in result.iloc[0].items()
                    },
                    "domain": properties[key]["domain"],
                    "num_variates": properties[key]["num_variates"],
                }
            ]
        )
        row.to_csv(path, mode="a", header=not path.exists(), index=False)
        completed.add(dataset_id)
