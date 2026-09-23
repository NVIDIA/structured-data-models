# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("autogluon.tabular")
pytest.importorskip("tabarena")

from benchmark.tabular.model import _kumo_prediction_batch_size


@pytest.mark.parametrize(
    ("context_rows", "query_rows", "columns", "memory_gib", "expected"),
    [
        (1_000, 500, 30, 24, 8),
        (1_000, 500, 30, 4, 1),
        (10_000, 500, 30, 240, 1),
        (1_000, 10_000, 30, 240, 1),
        (1_000, 500, 100, 240, 1),
        (1, 1, 500, 8, 1),
    ],
)
def test_kumo_prediction_batch_size(
    context_rows: int,
    query_rows: int,
    columns: int,
    memory_gib: int,
    expected: int,
) -> None:
    assert (
        _kumo_prediction_batch_size(
            context_rows=context_rows,
            query_rows=query_rows,
            columns=columns,
            num_estimators=8,
            available_memory=memory_gib * 2**30,
        )
        == expected
    )
