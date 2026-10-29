# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import zlib
from collections.abc import Iterator

import datasets
import fev
import numpy as np
import torch
import tqdm

import sdm
from benchmark.timeseries.model import ModelConfig
from sdm import TableTensor


class SDMPredictor(fev.ForecastingModel):
    def __init__(
        self,
        model: sdm.models.ICLModel,
        config: ModelConfig,
        batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.batch_size = batch_size
        self.device = next(model.parameters()).device

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        predictions = []
        for window in tqdm.tqdm(
            task.iter_windows(),
            total=task.num_windows,
            desc=task.task_name,
        ):
            past, future = window.get_input_data()

            all_columns = task.known_dynamic_columns.copy()
            if self.model.supports_context_only_columns:
                all_columns += task.past_dynamic_columns

            values = []
            with self._record_inference_time():
                for ys, xs_context, xs_query in zip(
                    _from(past, task.target_columns, self.batch_size),
                    _from(past, all_columns, self.batch_size),
                    _from(future, task.known_dynamic_columns, self.batch_size),
                    strict=True,
                ):
                    generator = torch.Generator().manual_seed(42)

                    if self.config.max_context_size is not None:
                        length = self.config.max_context_size
                        ys = [y[..., -length:] for y in ys]
                        xs_context = [x[..., -length:] for x in xs_context]

                    length = max(y.size(-1) for y in ys)
                    y = torch.full(
                        (len(ys), ys[0].size(0), length),
                        fill_value=torch.nan,
                    )
                    for i, y_i in enumerate(ys):
                        y[i, :, -y_i.size(-1) :] = y_i
                    x_context = torch.full(
                        (
                            len(ys),
                            xs_context[0].size(0)
                            if len(xs_context) > 0
                            else 0,
                            length,
                        ),
                        fill_value=torch.nan,
                    )
                    for i, x_context_i in enumerate(xs_context):
                        x_context[i, :, -x_context_i.size(-1) :] = x_context_i
                    x_query = torch.empty(
                        len(ys),
                        xs_query[0].size(0) if len(xs_query) > 0 else 0,
                        task.horizon,
                    )
                    for i, x_query_i in enumerate(xs_query):
                        x_query[i, :, :] = x_query_i

                    if (  # Keep room for at least one target:
                        self.config.max_variates is not None
                        and x_query.size(1) >= self.config.max_variates
                    ):
                        perm = torch.randperm(
                            x_query.size(1), generator=generator
                        )[: self.config.max_variates - 1]
                        x_context = x_context[:, perm]
                        x_query = x_query[:, perm]
                    elif (
                        self.config.max_variates is not None
                        and x_context.size(1) >= self.config.max_variates
                    ):
                        perm = torch.randperm(
                            x_context.size(1) - x_query.size(1),
                            generator=generator,
                        )[: self.config.max_variates - 1 - x_query.size(1)]
                        x_context = torch.cat(
                            [
                                x_context[:, : x_query.size(1)],
                                x_context[:, perm + x_query.size(1)],
                            ],
                            dim=1,
                        )

                    y = y.transpose(-1, -2).to(
                        self.device,
                        memory_format=torch.contiguous_format,
                    )
                    x_context = x_context.transpose(-1, -2).to(
                        self.device,
                        memory_format=torch.contiguous_format,
                    )
                    x_query = x_query.transpose(-1, -2).to(
                        self.device,
                        memory_format=torch.contiguous_format,
                    )

                    size = y.size(-1)
                    if self.config.max_variates is not None:
                        size = min(
                            size,
                            self.config.max_variates - x_context.size(-1),
                        )

                    preds = []
                    for start in range(0, y.size(-1), size):
                        y_i = y[..., start : start + size]
                        if y_i.size(-1) < size:
                            y_i = torch.cat(
                                [y_i, y[..., : size - y_i.size(-1)]], dim=-1
                            )

                        with torch.amp.autocast(
                            self.device.type,
                            dtype=self.config.autocast_dtype,
                            enabled=x_context.is_cuda
                            and self.config.autocast_dtype != torch.float32,
                        ):
                            out = self.model(
                                x_context=TableTensor.from_tensor(x_context),
                                y_context=TableTensor.from_tensor(y_i),
                                x_query=TableTensor.from_tensor(x_query),
                                num_estimators=self.config.num_estimators,
                            )
                        pred = out.numerical
                        pred = pred.view(y.size(0), task.horizon, size, -1)
                        preds.append(pred)

                    pred = torch.cat(preds, dim=-2)[..., : y.size(-1), :]
                    pred = pred.cpu().numpy()  # [B, R, C, Q]
                    levels = out.numerical_columns[: pred.shape[-1]]
                    levels = [float(f"0.{q.split('q')[-1]}") for q in levels]
                    wanted = [0.5, *task.quantile_levels]
                    index = [levels.index(q) for q in wanted]
                    values.append(pred[..., index])

            values = np.concatenate(values)  # [B, R, C, 1 + Q]
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


def _from(
    dataset: datasets.Dataset,
    columns: list[str],
    batch_size: int,
) -> Iterator[list[torch.Tensor]]:
    string_columns = [
        column
        for column in columns
        if dataset.features[column].feature.dtype in ("string", "large_string")
    ]

    for batch in dataset.with_format("numpy").iter(batch_size):
        if not columns:
            yield []
            continue

        for column in string_columns:
            batch[column] = [
                [
                    zlib.crc32(value.encode()) % 100
                    if isinstance(value, str)
                    else 0
                    for value in values
                ]
                for values in batch[column]
            ]

        yield [
            torch.from_numpy(
                np.asarray(
                    [batch[column][i] for column in columns],
                    dtype=np.float32,
                )
            )
            for i in range(len(batch[columns[0]]))
        ]
