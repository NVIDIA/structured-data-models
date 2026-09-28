# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("autogluon")
pytest.importorskip("tabarena")

from benchmark.tabular.model import (
    SDMKumoTabularSmallModel,
    SDMTabICLv2Model,
)
from sdm import CategoricalTensor, Stype, TableTensor
from sdm.models import ECOC
from sdm.models.base import _batch_slices, _query_chunks
from sdm.processing.execution import MemberContext, MemberQuery


def test_estimator_cost() -> None:
    adapter = object.__new__(SDMTabICLv2Model)
    x = TableTensor.from_tensor(torch.empty(2, 5, 7))

    assert adapter._estimator_cost(x, num_classes=0) == 10 * (7 + 32)
    assert adapter._estimator_cost(x, num_classes=5) == 10 * (7 + 32)


def test_auto_estimator_batching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = object.__new__(SDMTabICLv2Model)
    monkeypatch.setattr(
        SDMTabICLv2Model,
        "_get_model_params",
        lambda _: {"estimator_batch_size": "auto"},
    )

    size, cost, max_cost = adapter._estimator_batching()

    assert size is None
    assert cost == adapter._estimator_cost
    assert max_cost == 2**20


def test_kumo_ecoc_cost_controls_batches_and_query_chunks() -> None:
    adapter = object.__new__(SDMKumoTabularSmallModel)
    adapter.model = SimpleNamespace(ecoc=ECOC(max_classes=10))
    x = TableTensor.from_tensor(torch.empty(500, 4))

    base_cost = 500 * (4 + 32)
    assert adapter._estimator_cost(x, num_classes=0) == base_cost
    assert adapter._estimator_cost(x, num_classes=5) == base_cost
    assert adapter._estimator_cost(x, num_classes=12) == base_cost * 8

    y = TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.zeros(500, 1, dtype=torch.long),
            categories=(torch.arange(12),),
        ),
    )
    context = MemberContext(
        x=x,
        y=y,
        related_tables=None,
        input_stypes={},
    )
    batches = _batch_slices(
        contexts=[context] * 8,
        queries=None,
        class_values=[tuple(range(12))] * 8,
        estimator_batch_size=None,
        max_cost=adapter._estimator_batch_budget,
        cost=adapter._estimator_cost,
    )
    assert [(batch.start, batch.stop) for batch in batches] == [
        (0, 7),
        (7, 8),
    ]

    query = MemberQuery(
        x=TableTensor.from_tensor(torch.empty(1_000, 4)),
        related_tables=None,
    )
    chunks = _query_chunks(
        queries=[query] * 7,
        max_cost=adapter._estimator_batch_budget,
        cost=adapter._estimator_cost,
        num_classes=12,
    )
    assert [[member.x.size(-2) for member in chunk] for chunk in chunks] == [
        [520] * 7,
        [480] * 7,
    ]
