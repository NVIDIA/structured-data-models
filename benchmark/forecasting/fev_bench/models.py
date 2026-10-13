# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDM forecasting adapter for fev-bench."""

import zlib
from collections.abc import Iterator

import datasets
import fev
import numpy as np
import pandas as pd

import sdm
from benchmark.forecasting.model import ModelConfig, forecast, select_levels


class SDMModel(fev.ForecastingModel):
    """fev model forecasting batches of series with an SDM model."""

    def __init__(
        self,
        model: sdm.models.ICLModel,
        config: ModelConfig,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.batch_size = batch_size

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        predictions = []
        for window in task.iter_windows():
            past, future = window.get_input_data()

            values = []
            with self._record_inference_time():
                for past_batch, future_batch in zip(
                    _to_frames(past, self.batch_size),
                    _to_frames(future, self.batch_size),
                    strict=True,
                ):
                    out, levels = forecast(
                        self.model,
                        config=self.config,
                        past=[self._prepare(task, df) for df in past_batch],
                        future=[
                            self._prepare(task, df) for df in future_batch
                        ],
                        targets=task.target_columns,
                    )
                    # Median point forecast followed by the task quantiles:
                    wanted = [0.5, *task.quantile_levels]
                    values.append(select_levels(out, levels, wanted))
            values = np.concatenate(values)  # [N, H, V, 1 + Q]

            window_predictions = datasets.DatasetDict()
            for i, column in enumerate(task.target_columns):
                window_predictions[column] = datasets.Dataset.from_dict(
                    {
                        "predictions": values[:, :, i, 0],
                        **{
                            str(level): values[:, :, i, j + 1]
                            for j, level in enumerate(task.quantile_levels)
                        },
                    }
                )
            predictions.append(window_predictions)
        return predictions

    def _prepare(self, task: fev.Task, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the past or future data of a series for the model."""
        return df


class TimesFM3Model(SDMModel):
    """fev model for ``TimesFM3``.

    Static covariates are ignored, while targets and dynamic covariates are
    converted to numbers (hashing strings into 100 buckets) and imputed with
    the last observed value.
    """

    model_name = "sdm-timesfm3"

    def _prepare(self, task: fev.Task, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=task.static_columns)
        for column in df.columns.drop([task.id_column, task.timestamp_column]):
            x = df[column]
            if pd.api.types.is_string_dtype(x):
                x = x.fillna("").map(lambda v: zlib.crc32(v.encode()) % 100)
            df[column] = x.astype("float32").ffill().bfill().fillna(0)
        return df


MODELS: dict[str, type[SDMModel]] = {
    "timesfm3": TimesFM3Model,
}


def _to_frames(
    dataset: datasets.Dataset,
    batch_size: int,
) -> Iterator[list[pd.DataFrame]]:
    # One frame per series, with static values repeated over time steps and
    # the column types declared by the benchmark:
    dtypes = {
        name: pd.ArrowDtype(getattr(feature, "feature", feature).pa_type)
        for name, feature in dataset.features.items()
    }
    for batch in dataset.with_format("numpy").iter(batch_size):
        num_series = len(next(iter(batch.values())))
        yield [
            pd.DataFrame({name: x[i] for name, x in batch.items()}).astype(
                dtypes
            )
            for i in range(num_series)
        ]
