# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import pytest
import torch

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.models import KumoTabular
from sdm.models.kumo.tabular import (
    estimate_fit_batch_size,
    estimate_predict_batch_size,
)


def _context(
    num_columns: int, num_classes: int
) -> tuple[TableTensor, TableTensor]:
    x = TableTensor.from_tensor(torch.empty(1000, num_columns, device="meta"))
    if not num_classes:
        return x, TableTensor.from_tensor(torch.empty(1000, 1, device="meta"))
    y = TableTensor(
        columns={Stype.categorical: ["target"]},
        categorical=CategoricalTensor(
            code=torch.empty(1000, 1, dtype=torch.int64, device="meta"),
            categories=(torch.arange(num_classes),),
        ),
    )
    return x, y


@pytest.mark.parametrize(
    ("size", "num_classes", "num_columns", "fit_size", "query_size"),
    [
        ("small", 0, 30, 5, 620),
        ("medium", 3, 30, 2, 868),
        ("large", 10, 30, 2, 786),
        ("small", 11, 30, 1, 204),
        ("medium", 100, 30, 1, 75),
        ("large", 0, 500, 1, 101),
    ],
)
def test_estimate_batch_sizes(
    size: Literal["small", "medium", "large"],
    num_classes: int,
    num_columns: int,
    fit_size: int,
    query_size: int,
) -> None:
    task: Literal["classification", "regression"] = (
        "classification" if num_classes else "regression"
    )
    model = KumoTabular(task=task, size=size, pretrained=False, device="meta")
    x, y = _context(num_columns, num_classes)
    assert (
        estimate_fit_batch_size(
            model, x, y, memory_budget=2**30, num_estimators=16
        )
        == fit_size
    )
    assert (
        estimate_predict_batch_size(
            model=model,
            x=x,
            y=y,
            memory_budget=2**30,
            num_estimators=16,
            estimator_batch_size=4,
        )
        == query_size
    )


def test_estimate_batch_size_bounds() -> None:
    model = KumoTabular(task="regression", pretrained=False, device="meta")
    x, y = _context(30, 0)
    assert (
        estimate_fit_batch_size(
            model, x, y, memory_budget=0, num_estimators=16
        )
        == 1
    )
    assert (
        estimate_predict_batch_size(
            model, x, y, memory_budget=0, num_estimators=16
        )
        == 1
    )
    assert (
        estimate_fit_batch_size(
            model, x, y, memory_budget=2**60, num_estimators=16
        )
        == 16
    )
