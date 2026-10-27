# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Run an SDM time series forecasting model on TIME."""

import argparse
import os
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import tqdm
from gluonts.time_feature import get_seasonality
from huggingface_hub import snapshot_download
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.utils import get_available_terms

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
    help="Run only the selected TIME dataset; repeat to combine.",
)
args = parser.parse_args()

warnings.filterwarnings("ignore", message=r"'.*' is deprecated.*")

config = MODEL_CONFIGS[args.model]
dataset_config = load_dataset_config()
path = Path(__file__).parent.parent / "time_bench_out" / config.name

names = args.dataset or dataset_config["datasets"]
if not os.getenv("TIME_DATASET"):
    os.environ["TIME_DATASET"] = snapshot_download(
        repo_id="Real-TSF/TIME",
        repo_type="dataset",
        allow_patterns=[f"{name}/*" for name in names],
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = config.factory(device)

for name in names:
    for term in get_available_terms(name, dataset_config):
        if (path / name / term / "metrics.npz").exists():
            continue

        settings = get_dataset_settings(name, term, dataset_config)
        dataset = Dataset(
            name=name,
            term=term,
            prediction_length=settings["prediction_length"],
            test_length=settings["test_length"],
        )

        predictor = SDMPredictor(model, config, dataset.prediction_length)
        outs = list(
            tqdm.tqdm(
                predictor.predict(dataset.test_data.input),
                total=len(dataset.test_data),
                desc=f"{name}/{term}",
            )
        )

        quantiles = np.stack([out.forecast_array for out in outs])
        if quantiles.ndim == 4:
            quantiles = quantiles.swapaxes(-1, -2)  # [B, Q, C, R]

        with redirect_stdout(None):
            save_window_predictions(
                dataset=dataset,
                fc_quantiles=quantiles,
                ds_config=f"{name}/{term}",
                output_base_dir=str(path),
                seasonality=get_seasonality(dataset.freq),
                quantile_levels=[float(q) for q in outs[0].forecast_keys],
            )
        with np.load(path / name / term / "metrics.npz") as metrics:
            mase = np.nanmean(metrics["MASE"])
            crps = np.nanmean(metrics["CRPS"])
        print(f"{name}/{term}: MASE={mase:.4f}, CRPS={crps:.4f}")
