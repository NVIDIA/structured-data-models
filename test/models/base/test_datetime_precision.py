# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pyarrow as pa
import pytest
import torch

from sdm import RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.processing import AddCalendarFields, Recipe


class CalendarModel(ICLModel):
    supported_feature_stypes = frozenset({Stype.datetime})
    supported_target_stypes = frozenset({Stype.numerical})
    supports_multi_target = False
    supports_context_only_columns = False
    supports_related_tables = False

    @classmethod
    def default_recipe(cls) -> Recipe:
        return Recipe()

    def _forward(
        self,
        x_context: TableTensor | None,
        y_context: TableTensor | None,
        x_query: TableTensor | None,
        x_context_only: TableTensor | None,
        related_context_tables: RelatedTables | None,
        related_query_tables: RelatedTables | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:
        table = x_context if x_query is None else x_query
        assert table is not None
        return (
            AddCalendarFields(["month"])
            .transform(table)
            .select_columns("precise__month")
        )


@pytest.mark.parametrize("estimator_batch_size", [None, 2])
def test_eight_member_model_keeps_datetime_units_during_execution(
    estimator_batch_size: int | None,
) -> None:
    midnight = 1_704_067_200_000_000_000
    table = TableTensor.from_arrow(
        pa.table(
            {
                "legacy": pa.array(
                    [midnight // 1_000] * 2, type=pa.timestamp("us")
                ),
                "precise": pa.array(
                    [midnight - 1, midnight + 1], type=pa.timestamp("ns")
                ),
            }
        ),
        {"legacy": "datetime", "precise": "datetime"},
    )
    model = CalendarModel()
    output = model(
        x_context=table,
        y_context=torch.zeros(2, 1),
        x_query=table,
        recipe=Recipe(),
        num_estimators=8,
        estimator_batch_size=estimator_batch_size,
    )
    torch.testing.assert_close(
        output.numerical, torch.tensor([[[11.0], [0.0]]]).expand(8, -1, -1)
    )


@pytest.mark.parametrize("estimator_batch_size", [None, 2])
def test_fitted_eight_member_model_preserves_query_precision(
    estimator_batch_size: int | None,
) -> None:
    table = TableTensor.from_arrow(
        pa.table({"precise": pa.array([-1, 1], type=pa.timestamp("ns"))}),
        {"precise": "datetime"},
    )
    model = CalendarModel().eval()
    model.fit(
        table,
        torch.zeros(2, 1),
        recipe=Recipe(),
        num_estimators=8,
        estimator_batch_size=estimator_batch_size,
    )
    output = model.predict(table)
    torch.testing.assert_close(
        output.numerical, torch.tensor([[[11.0], [0.0]]]).expand(8, -1, -1)
    )
