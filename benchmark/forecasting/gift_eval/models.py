# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDM forecasting adapter for GIFT-Eval."""

from collections.abc import Iterator

import numpy as np
import pandas as pd
from gluonts.dataset import Dataset
from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast
from gluonts.model.predictor import Predictor

import sdm
from benchmark.forecasting.model import ModelConfig, forecast, select_levels

QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class SDMPredictor(Predictor):
    """GluonTS predictor forecasting batches of series with an SDM model."""

    def __init__(
        self,
        model: sdm.models.ICLModel,
        config: ModelConfig,
        prediction_length: int,
        batch_size: int,
    ) -> None:
        super().__init__(prediction_length=prediction_length)
        self.model = model
        self.config = config
        self.batch_size = batch_size

    def predict(
        self,
        dataset: Dataset,
        **kwargs: object,
    ) -> Iterator[QuantileForecast]:
        for batch in batcher(dataset, batch_size=self.batch_size):
            # GluonTS stores series as [V, T] (or [T] if univariate):
            num_variates = np.atleast_2d(batch[0]["target"]).shape[0]
            targets = [f"target_{i}" for i in range(num_variates)]
            past, future = [], []
            for entry in batch:
                target = np.atleast_2d(entry["target"])
                length = target.shape[-1]
                timestamp = pd.period_range(
                    entry["start"],
                    periods=length + self.prediction_length,
                ).to_timestamp()
                columns = {
                    "item_id": entry["item_id"],
                    "timestamp": timestamp[:length],
                }
                columns |= dict(zip(targets, target, strict=True))
                for i, x in enumerate(entry.get("past_feat_dynamic_real", [])):
                    columns[f"past_feat_dynamic_real_{i}"] = x
                past.append(pd.DataFrame(columns))
                future.append(
                    pd.DataFrame(
                        {
                            "item_id": entry["item_id"],
                            "timestamp": timestamp[length:],
                        }
                    )
                )
            values, levels = forecast(
                self.model,
                config=self.config,
                past=past,
                future=future,
                targets=targets,
            )
            values = select_levels(values, levels, QUANTILE_LEVELS)
            for entry, value in zip(batch, values, strict=True):
                value = value.transpose(2, 0, 1)  # [Q, H, V]
                if entry["target"].ndim == 1:
                    value = value[..., 0]
                yield QuantileForecast(
                    forecast_arrays=value,
                    forecast_keys=[str(q) for q in QUANTILE_LEVELS],
                    start_date=entry["start"] + entry["target"].shape[-1],
                )
