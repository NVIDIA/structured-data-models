# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Run an SDM forecasting model on GIFT-Eval."""

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
from gift_eval.data import Dataset
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality

from benchmark.forecasting.gift_eval import GIFT_EVAL_URL
from benchmark.forecasting.gift_eval.models import (
    QUANTILE_LEVELS,
    SDMPredictor,
)
from benchmark.forecasting.model import MODEL_CONFIGS, add_model_args

# Datasets and terms from GIFT-Eval's notebooks:
SHORT_DATASETS = (  # noqa: SIM905
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly "
    "electricity/15T electricity/H electricity/D electricity/W "
    "solar/10T solar/H solar/D solar/W hospital covid_deaths "
    "us_births/D us_births/M us_births/W "
    "saugeenday/D saugeenday/M saugeenday/W "
    "temperature_rain_with_missing "
    "kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D "
    "car_parts_with_missing restaurant "
    "hierarchical_sales/D hierarchical_sales/W "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H "
    "M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W "
    "ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H "
    "jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H "
    "bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
).split()
MED_LONG_DATASETS = (  # noqa: SIM905
    "electricity/15T electricity/H solar/10T solar/H "
    "kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T "
    "M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T "
    "jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T "
    "bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
).split()
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}
METRICS = [
    MSE(forecast_type="mean"),
    MSE(forecast_type="0.5"),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument(
        "--dataset",
        action="append",
        help="Run only the selected GIFT-Eval dataset; repeat to combine.",
    )
    args = parser.parse_args()

    model_config = MODEL_CONFIGS[args.model]
    if args.max_context_size is not None:
        model_config = replace(
            model_config, max_context_size=args.max_context_size
        )
    result_path = (
        Path(__file__).parent.parent
        / "gift_eval_out"
        / model_config.name
        / "all_results.csv"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    completed = (
        set(pd.read_csv(result_path)["dataset"])
        if result_path.exists()
        else set()
    )
    properties = pd.read_json(
        f"{GIFT_EVAL_URL}/notebooks/dataset_properties.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_config.factory(device)

    for name in args.dataset or SHORT_DATASETS:
        key, _, freq = name.partition("/")
        key = PRETTY_NAMES.get(key.lower(), key.lower())
        freq = freq or properties[key]["frequency"]
        for term in ("short", "medium", "long"):
            if term != "short" and name not in MED_LONG_DATASETS:
                continue
            config = f"{key}/{freq}/{term}"
            if config in completed:
                continue

            # Flatten multivariate series for models without joint multi-target
            # forecasting:
            dataset = Dataset(
                name=name,
                term=term,
                to_univariate=(
                    not model.supports_multi_target
                    and properties[key]["num_variates"] > 1
                ),
            )
            predictor = SDMPredictor(
                model,
                config=model_config,
                prediction_length=dataset.prediction_length,
                batch_size=args.batch_size,
            )
            result = evaluate_model(
                predictor,
                test_data=dataset.test_data,
                metrics=METRICS,
                seasonality=get_seasonality(dataset.freq),
            )
            row = pd.DataFrame(
                [
                    {
                        "dataset": config,
                        "model": f"[SDM] {model_config.name}",
                        **{
                            f"eval_metrics/{k}": v
                            for k, v in result.iloc[0].items()
                        },
                        "domain": properties[key]["domain"],
                        "num_variates": properties[key]["num_variates"],
                    }
                ]
            )
            row.to_csv(
                result_path,
                mode="a",
                header=not result_path.exists(),
                index=False,
            )
            print(row.to_string(index=False))


if __name__ == "__main__":
    main()
