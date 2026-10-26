# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import torch
from gluonts.dataset import Dataset
from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast
from gluonts.model.predictor import Predictor

import sdm
from benchmark.timeseries.model import ModelConfig

QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class SDMPredictor(Predictor):
    def __init__(
        self,
        model: sdm.models.ICLModel,
        config: ModelConfig,
        prediction_length: int,
        batch_size: int = 64,
    ) -> None:
        super().__init__(prediction_length=prediction_length)
        self.model = model
        self.config = config
        self.batch_size = batch_size
        self.device = next(model.parameters()).device

    def predict(
        self,
        dataset: Dataset,
        **kwargs: object,
    ) -> Iterator[QuantileForecast]:

        for batch in batcher(dataset, batch_size=self.batch_size):
            cpu_generator = torch.Generator().manual_seed(42)

            ys = [torch.from_numpy(entry["target"]) for entry in batch]
            ys = [torch.atleast_2d(y) for y in ys]
            xs = []
            if (
                self.model.supports_context_only_columns
                and "past_feat_dynamic_real" in batch[0]
            ):
                xs = [
                    torch.from_numpy(entry["past_feat_dynamic_real"])
                    for entry in batch
                ]
                xs = [torch.atleast_2d(x) for x in xs]

            if self.config.max_context_size is not None:
                ys = [y[..., -self.config.max_context_size :] for y in ys]
                xs = [x[..., -self.config.max_context_size :] for x in xs]

            length = max(y.size(-1) for y in ys)
            y = torch.full(
                (len(ys), ys[0].size(0), length),
                fill_value=torch.nan,
            )
            for i, y_i in enumerate(ys):
                y[i, :, -y_i.size(-1) :] = y_i
            x = torch.full(
                (len(ys), xs[0].size(0) if len(xs) > 0 else 0, length),
                fill_value=torch.nan,
            )
            for i, x_i in enumerate(xs):
                x[i, :, -x_i.size(-1) :] = x_i

            if (
                self.config.max_variates is not None
                and x.size(1) > self.config.max_variates - 1
            ):  # Keep room for at least one target:
                perm = torch.randperm(x.size(1), generator=cpu_generator)
                x = x[:, perm[: self.config.max_variates - 1]]

            y = y.transpose(-1, -2).to(
                self.device,
                memory_format=torch.contiguous_format,
            )
            x = x.transpose(-1, -2).to(
                self.device,
                memory_format=torch.contiguous_format,
            )

            size = y.size(-1)
            if self.config.max_variates is not None:
                size = min(size, self.config.max_variates - x.size(-1))

            preds = []
            for start in range(0, y.size(-1), size):
                y_i = y[..., start : start + size]
                if y_i.size(-1) < size:
                    y_i = torch.cat([y_i, y[..., : size - y_i.size(-1)]], -1)

                out = self.model(
                    x_context=x,
                    y_context=y_i,
                    x_query=torch.empty(
                        (x.size(0), self.prediction_length, 0),
                        device=self.device,
                    ),
                    num_estimators=self.config.num_estimators,
                )
                pred = out.numerical
                pred = pred.view(y.size(0), self.prediction_length, size, -1)
                preds.append(pred)

            pred = torch.cat(preds, dim=-2)[..., : y.size(-1), :]
            pred = pred.permute(0, 3, 1, 2).cpu().numpy()  # [B, Q, R, C]
            levels = out.numerical_columns[: pred.shape[1]]
            levels = [str(float(f"0.{q.split('q')[-1]}")) for q in levels]

            for entry, pred_i in zip(batch, pred):
                if entry["target"].ndim == 1:
                    pred_i = pred_i[..., 0]
                yield QuantileForecast(
                    forecast_arrays=pred_i,
                    forecast_keys=levels,
                    start_date=entry["start"] + entry["target"].shape[-1],
                )
