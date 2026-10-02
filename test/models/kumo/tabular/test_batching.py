# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Literal

import pytest
import torch

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.models import KumoTabular
from sdm.models.kumo.tabular import estimate_batch_size


@pytest.fixture(params=["small", "medium", "large"])
def size(
    request: pytest.FixtureRequest,
) -> Literal["small", "medium", "large"]:
    return request.param


@pytest.mark.parametrize("num_classes", [0, 2, 10, 11, 100])
@pytest.mark.parametrize(
    ("num_rows", "num_columns"), [(1, 1), (3000, 25), (10000, 500)]
)
def test_estimate_batch_size(
    size: Literal["small", "medium", "large"],
    num_classes: int,
    num_rows: int,
    num_columns: int,
) -> None:
    task: Literal["classification", "regression"] = (
        "classification" if num_classes else "regression"
    )
    model = KumoTabular(task=task, size=size, pretrained=False, device="meta")
    x = TableTensor.from_tensor(
        torch.empty(num_rows, num_columns, device="meta")
    )
    if num_classes:
        y = TableTensor(
            columns={Stype.categorical: ["target"]},
            categorical=CategoricalTensor(
                code=torch.empty(
                    num_rows, 1, dtype=torch.int64, device="meta"
                ),
                categories=(torch.arange(num_classes),),
            ),
        )
    else:
        y = TableTensor.from_tensor(torch.empty(num_rows, 1, device="meta"))
    cell_channels, icl_channels, num_layers = {
        "small": (128, 512, 12),
        "medium": (256, 512, 24),
        "large": (256, 1024, 24),
    }[size]
    tasks = (
        max(
            math.ceil(num_classes / 9),
            4 * math.ceil(math.log(num_classes, 10)),
        )
        if num_classes > 10
        else 1
    )
    workspace = (
        tasks
        * 2
        * (
            4 * (min(2 * num_columns, 500) + 4) * cell_channels
            + 15 * icl_channels
        )
    )
    cache = tasks * 2 * 2 * icl_channels * num_layers
    for memory_budget in (0, 2**29 + 1, 2**30):
        expected = max(
            1,
            min(16, (memory_budget // 2) // (num_rows * (workspace + cache))),
        )
        assert (
            estimate_batch_size(
                model=model,
                x=x,
                y=y,
                num_estimators=16,
                memory_budget=memory_budget,
            )
            == expected
        )
        for estimator_batch_size in (1, 4, 16, 32):
            row_bytes = workspace * estimator_batch_size + 16 * 8 * (
                8 * num_columns + 4 * (num_classes or 999)
            )
            expected = max(1, (memory_budget // 2) // row_bytes)
            assert (
                estimate_batch_size(
                    model=model,
                    x=x,
                    y=y,
                    mode="predict",
                    num_estimators=16,
                    estimator_batch_size=estimator_batch_size,
                    memory_budget=memory_budget,
                )
                == expected
            )


@pytest.mark.parametrize("mode", ["fit", "predict"])
def test_estimate_batch_size_execution_dtype(
    size: Literal["small", "medium", "large"],
    mode: Literal["fit", "predict"],
) -> None:
    model = KumoTabular(
        task="regression", size=size, pretrained=False, device="meta"
    )
    x = TableTensor.from_tensor(torch.empty(100, 30, device="meta"))
    y = TableTensor.from_tensor(torch.empty(100, 1, device="meta"))
    num_estimators = 1000 if mode == "fit" else 16
    fp16 = estimate_batch_size(
        model=model,
        x=x,
        y=y,
        memory_budget=2**30,
        num_estimators=num_estimators,
        estimator_batch_size=4,
        mode=mode,
    )
    fp32 = estimate_batch_size(
        model=model,
        x=x,
        y=y,
        memory_budget=2**30,
        num_estimators=num_estimators,
        estimator_batch_size=4,
        mode=mode,
        dtype=torch.float32,
    )
    assert 1 <= fp32 < fp16
