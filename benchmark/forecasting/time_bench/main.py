# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Run an SDM forecasting model on the TIME benchmark."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from gluonts.time_feature import get_seasonality
from timebench.evaluation.data import (
    Dataset,
    get_dataset_settings,
    load_dataset_config,
)
from timebench.evaluation.saver import save_window_predictions
from timebench.evaluation.utils import get_available_terms

from benchmark.forecasting.gift_eval.models import (
    QUANTILE_LEVELS,
    SDMPredictor,
)
from benchmark.forecasting.model import MODEL_CONFIGS, add_model_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument(
        "--dataset",
        action="append",
        help="Run only the selected TIME dataset; repeat to combine.",
    )
    args = parser.parse_args()

    model_config = MODEL_CONFIGS[args.model]
    if args.max_context_size is not None:
        model_config = replace(
            model_config, max_context_size=args.max_context_size
        )
    result_dir = (
        Path(__file__).parent.parent / "time_bench_out" / model_config.name
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_config.factory(device)

    # Mirrors TIME's official `experiments/*.py` runners:
    config = load_dataset_config()
    for name in args.dataset or config["datasets"]:
        for term in get_available_terms(name, config):
            if (result_dir / name / term / "metrics.npz").exists():
                continue

            settings = get_dataset_settings(name, term, config)
            dataset = Dataset(
                name=name,
                term=term,
                prediction_length=settings["prediction_length"],
                test_length=settings["test_length"],
                val_length=settings.get("val_length", 0),
                # Flatten multivariate series for models without joint
                # multi-target forecasting:
                to_univariate=not model.supports_multi_target,
            )

            predictor = SDMPredictor(
                model,
                config=model_config,
                prediction_length=dataset.prediction_length,
                batch_size=args.batch_size,
            )
            forecasts = predictor.predict(dataset.test_data.input)
            # GluonTS stores quantiles as [Q, H, V] (or [Q, H] if univariate):
            quantiles = np.stack([f.forecast_array for f in forecasts])
            if quantiles.ndim == 4:
                quantiles = quantiles.swapaxes(-1, -2)  # [N, Q, V, H]

            save_window_predictions(
                dataset=dataset,
                fc_quantiles=quantiles,
                ds_config=f"{name}/{term}",
                output_base_dir=str(result_dir),
                seasonality=get_seasonality(dataset.freq),
                model_hyperparams=vars(args),
                quantile_levels=list(QUANTILE_LEVELS),
            )


if __name__ == "__main__":
    main()
