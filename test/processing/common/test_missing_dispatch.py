# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import sdm.processing as sp
from sdm import EnsembleTable, TableTensor


def _sparse_table() -> TableTensor:
    # 25 rows, 60% of them holding a NaN in the first column.
    numerical = torch.arange(75, dtype=torch.float32).reshape(25, 3)
    numerical[:15, 0] = float("nan")
    return TableTensor.from_tensor(numerical, columns=("a", "b", "c"))


def _imputed(table: TableTensor) -> torch.Tensor:
    numerical = table.numerical
    mean = numerical.nanmean(dim=-2, keepdim=True)
    return torch.where(numerical.isnan(), mean, numerical)


def _assert_same(actual: TableTensor, expected: TableTensor) -> None:
    assert actual.columns == expected.columns
    torch.testing.assert_close(
        actual.numerical,
        expected.numerical,
        equal_nan=True,
    )


def test_missing_dispatch_routes_sparse_table_through_sparse() -> None:
    table = _sparse_table()
    dispatch = sp.MissingDispatch(
        sparse=sp.Identity(),
        dense=sp.ImputeMean(),
        min_rows=20,
        min_row_frac=0.5,
    )
    assert dispatch.route is None
    with pytest.raises(RuntimeError, match="fit"):
        dispatch.transform(table)

    _assert_same(dispatch.fit_transform(table), table)
    assert dispatch.route == "sparse"
    transformed = dispatch.transform(table)
    _assert_same(transformed, table)
    assert transformed.numerical.isnan().any()


def test_missing_dispatch_routes_short_table_through_dense() -> None:
    table = _sparse_table()
    dispatch = sp.MissingDispatch(
        sparse=sp.Identity(),
        dense=sp.ImputeMean(),
        min_rows=100,
        min_row_frac=0.5,
    )
    transformed = dispatch.fit_transform(table)
    assert dispatch.route == "dense"
    assert not transformed.numerical.isnan().any()
    torch.testing.assert_close(transformed.numerical, _imputed(table))
    assert dispatch.transform(table).equal(transformed)


@pytest.mark.parametrize(
    ("min_rows", "min_row_frac", "min_cell_frac", "route"),
    [
        (0, 0.6, 0.0, "sparse"),
        (0, 0.7, 0.0, "dense"),
        (0, 0.0, 0.2, "sparse"),
        (0, 0.0, 0.3, "dense"),
        (25, 0.0, 0.0, "sparse"),
        (26, 0.0, 0.0, "dense"),
    ],
)
def test_missing_dispatch_thresholds_gate_each_statistic(
    min_rows: int,
    min_row_frac: float,
    min_cell_frac: float,
    route: str,
) -> None:
    dispatch = sp.MissingDispatch(
        min_rows=min_rows,
        min_row_frac=min_row_frac,
        min_cell_frac=min_cell_frac,
    )
    dispatch.fit(_sparse_table())
    assert dispatch.route == route


@pytest.mark.parametrize("num_rows", [10, 7, 30])
def test_missing_dispatch_exact_threshold_routes_sparse(num_rows: int) -> None:
    # Fractions that are not exactly representable must still meet an
    # equal threshold, e.g. 7/10 rows against min_row_frac=0.7.
    numerical = torch.ones(num_rows, 10)
    numerical[:7, 0] = float("nan")
    table = TableTensor.from_tensor(numerical)
    dispatch = sp.MissingDispatch(
        min_row_frac=7 / num_rows,
        min_cell_frac=7 / (10 * num_rows),
    )
    dispatch.fit(table)
    assert dispatch.route == "sparse"


def test_missing_dispatch_missing_route_passes_through() -> None:
    table = _sparse_table()
    dispatch = sp.MissingDispatch(dense=sp.ImputeMean(), min_row_frac=0.5)
    _assert_same(dispatch.fit_transform(table), table)
    assert dispatch.route == "sparse"
    _assert_same(dispatch.transform(table), table)

    dispatch = sp.MissingDispatch(sparse=sp.Identity(), min_rows=100)
    _assert_same(dispatch.fit_transform(table), table)
    assert dispatch.route == "dense"


@pytest.mark.parametrize("min_rows", [20, 100])
def test_missing_dispatch_state_dict_round_trip_preserves_route(
    min_rows: int,
) -> None:
    table = _sparse_table()
    dispatch = sp.MissingDispatch(
        sparse=sp.Identity(),
        dense=sp.ImputeMean(),
        min_rows=min_rows,
        min_row_frac=0.5,
    )
    expected = dispatch.fit_transform(table)
    restored = sp.MissingDispatch(
        sparse=sp.Identity(),
        dense=sp.ImputeMean(),
        min_rows=min_rows,
        min_row_frac=0.5,
    )
    restored.load_state_dict(dispatch.state_dict())
    assert restored.route == dispatch.route
    _assert_same(restored.transform(table), expected)


@pytest.mark.parametrize("min_rows", [20, 100])
def test_missing_dispatch_routes_ensemble_members(min_rows: int) -> None:
    table = _sparse_table()
    expected = table.numerical if min_rows <= 25 else _imputed(table)
    ensemble_table = EnsembleTable.from_table(table, num_members=3)
    dispatch = sp.MissingDispatch(
        sparse=sp.Identity(),
        dense=sp.ImputeMean(),
        min_rows=min_rows,
        min_row_frac=0.5,
    )
    output = dispatch.fit_transform_ensemble(ensemble_table)
    assert output.num_members == 3
    for member_id in range(output.num_members):
        torch.testing.assert_close(
            output.table(member_id).numerical,
            expected,
            equal_nan=True,
        )
    output = dispatch.transform_ensemble(ensemble_table)
    for member_id in range(output.num_members):
        torch.testing.assert_close(
            output.table(member_id).numerical,
            expected,
            equal_nan=True,
        )


def test_missing_dispatch_repr_shows_routes_and_thresholds() -> None:
    dispatch = sp.MissingDispatch(
        sparse=sp.Identity(),
        dense=sp.ImputeMean(),
        min_rows=20,
        min_row_frac=0.5,
    )
    description = repr(dispatch)
    assert "sparse=Identity()" in description
    assert "dense=ImputeMean()" in description
    assert "min_rows=20" in description
    assert "min_row_frac=0.5" in description
    assert repr(sp.MissingDispatch()) == "MissingDispatch()"
