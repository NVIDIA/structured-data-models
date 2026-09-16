# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import sdm.processing as sp
from sdm import ColumnarTensor, EnsembleTable, TableTensor
from sdm.testing import withCUDA


@withCUDA
def test_reduce_estimators_mean(device: torch.device) -> None:
    values = torch.arange(
        2 * 3 * 4 * 2,
        dtype=torch.float32,
        device=device,
    ).reshape(2, 3, 4, 2)
    table = TableTensor.from_tensor(values)
    processor = sp.ReduceEstimators(method="mean")

    output = processor.transform(table)
    fit_output = processor.fit_transform(table)

    assert output.size() == (3, 4, 2)
    assert output.schema == table.schema
    assert output.dtype == values.dtype
    torch.testing.assert_close(output.numerical, values.mean(dim=0))
    torch.testing.assert_close(fit_output.numerical, output.numerical)
    assert repr(processor) == "ReduceEstimators(method='mean')"


def test_reduce_estimators_rejects_missing_ensemble_dimension() -> None:
    table = TableTensor.from_tensor(torch.ones(4, 2))

    with pytest.raises(ValueError, match="leading ensemble dimension"):
        sp.ReduceEstimators().transform(table)


def test_reduce_estimators_rejects_empty_ensemble_dimension() -> None:
    table = TableTensor.from_tensor(torch.empty(0, 4, 2))

    with pytest.raises(ValueError, match="at least one ensemble member"):
        sp.ReduceEstimators().transform(table)


def test_reduce_estimators_rejects_unknown_method() -> None:
    with pytest.raises(
        ValueError,
        match="method must be 'mean' or 'trimmed_mean'",
    ):
        sp.ReduceEstimators(method="median")  # type: ignore


@withCUDA
def test_reduce_estimators_trimmed_mean_rejects_outliers(
    device: torch.device,
) -> None:
    values = torch.tensor(
        [
            [[-100.0, 100.0]],
            [[1.0, 4.0]],
            [[2.0, 3.0]],
            [[3.0, 2.0]],
            [[100.0, -100.0]],
        ],
        dtype=torch.float64,
        device=device,
    )
    table = TableTensor.from_tensor(values, columns=("a", "b"))
    processor = sp.ReduceEstimators(method="trimmed_mean")

    output = processor.transform(table)

    assert output.size() == (1, 2)
    assert output.schema == table.schema
    assert output.dtype == values.dtype
    assert output.device == device
    torch.testing.assert_close(
        output.numerical,
        torch.tensor([[2.0, 3.0]], dtype=values.dtype, device=device),
    )
    assert repr(processor) == (
        "ReduceEstimators(method='trimmed_mean', proportion_to_cut=0.2)"
    )


@withCUDA
def test_reduce_estimators_trimmed_mean_floors_cut(
    device: torch.device,
) -> None:
    values = torch.arange(
        4 * 3 * 2, dtype=torch.float32, device=device
    ).reshape(4, 3, 2)
    output = sp.ReduceEstimators(method="trimmed_mean").transform(
        TableTensor.from_tensor(values)
    )

    torch.testing.assert_close(output.numerical, values.mean(dim=0))


@withCUDA
def test_reduce_estimators_trimmed_mean_non_default_proportion(
    device: torch.device,
) -> None:
    values = torch.tensor(
        [-100.0, -50.0, 1.0, 2.0, 3.0, 4.0, 50.0, 100.0],
        device=device,
    ).reshape(8, 1, 1)
    processor = sp.ReduceEstimators(
        method="trimmed_mean",
        proportion_to_cut=0.25,
    )

    output = processor.transform(TableTensor.from_tensor(values))

    torch.testing.assert_close(
        output.numerical,
        torch.tensor([[2.5]], device=device),
    )
    assert repr(processor) == (
        "ReduceEstimators(method='trimmed_mean', proportion_to_cut=0.25)"
    )


@pytest.mark.parametrize(
    "proportion_to_cut",
    [-0.1, 0.5, 1.0, float("nan")],
)
def test_reduce_estimators_rejects_invalid_proportion(
    proportion_to_cut: float,
) -> None:
    with pytest.raises(ValueError, match=r"must be in \[0, 0\.5\)"):
        sp.ReduceEstimators(
            method="trimmed_mean",
            proportion_to_cut=proportion_to_cut,
        )


def test_reduce_estimators_rejects_non_numerical_stypes() -> None:
    table = TableTensor(
        numerical=torch.ones(2, 3, 2),
        id=ColumnarTensor((torch.arange(2 * 3).reshape(2, 3),)),
    )

    with pytest.raises(ValueError, match="numerical-only output table"):
        sp.ReduceEstimators().transform(table)


def test_reduce_estimators_rejects_non_numerical_ensemble_stypes() -> None:
    member = TableTensor(
        numerical=torch.ones(2, 1),
        id=ColumnarTensor((torch.arange(2),)),
    )
    table = EnsembleTable.from_tables(
        tables=(member, member),
        member_table_ids=(0, 1),
    )

    with pytest.raises(ValueError, match="numerical-only output table"):
        sp.ReduceEstimators().transform_ensemble(table)


@withCUDA
def test_reduce_estimators_reduces_members_in_order(
    device: torch.device,
) -> None:
    first = TableTensor.from_tensor(torch.tensor([[0.0, 2.0]], device=device))
    second = TableTensor.from_tensor(torch.tensor([[3.0, 1.0]], device=device))
    table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1, 1),
    )

    output = sp.ReduceEstimators().transform_ensemble(table)

    assert output.num_members == 1
    torch.testing.assert_close(
        output.table(0).numerical,
        (first.numerical + 2 * second.numerical) / 3,
    )


@withCUDA
def test_reduce_estimators_reduces_across_storage_groups(
    device: torch.device,
) -> None:
    first = TableTensor.from_tensor(
        torch.tensor([[0.0, 2.0]], device=device),
        columns=("a", "b"),
    )
    second = TableTensor.from_tensor(
        torch.tensor([[4.0, 6.0]], device=device),
        columns=("b", "a"),
    )
    table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1, 1),
    )

    output = sp.ReduceEstimators().transform_ensemble(table)

    assert output.table(0).columns == first.columns
    torch.testing.assert_close(
        output.table(0).numerical,
        torch.tensor([[4.0, 10.0 / 3.0]], device=device),
    )


@withCUDA
def test_reduce_estimators_trimmed_mean_counts_logical_members(
    device: torch.device,
) -> None:
    first = TableTensor.from_tensor(
        torch.tensor([[0.0, 100.0]], dtype=torch.float64, device=device),
        columns=("a", "b"),
    )
    shared = TableTensor.from_tensor(
        torch.tensor([[10.0, 1.0]], dtype=torch.float64, device=device),
        columns=("b", "a"),
    )
    last = TableTensor.from_tensor(
        torch.tensor([[100.0, -100.0]], dtype=torch.float64, device=device),
        columns=("a", "b"),
    )
    table = EnsembleTable.from_tables(
        tables=(first, shared, last),
        member_table_ids=(0, 1, 1, 1, 2),
    )
    processor = sp.ReduceEstimators(method="trimmed_mean")

    output = processor.transform_ensemble(table)

    assert output.num_members == 1
    result = output.table(0)
    assert result.columns == first.columns
    assert result.schema == first.schema
    assert result.dtype == first.dtype
    assert result.device == device
    torch.testing.assert_close(
        result.numerical,
        torch.tensor([[1.0, 10.0]], dtype=first.dtype, device=device),
    )

    single = processor.transform_ensemble(
        EnsembleTable.from_table(first, num_members=1)
    )
    assert single.num_members == 1
    assert single.table(0).equal(first)


def test_reduce_estimators_rejects_empty_ensemble_table() -> None:
    table = TableTensor.from_tensor(torch.ones(4, 2))

    with pytest.raises(ValueError, match="at least one ensemble member"):
        sp.ReduceEstimators().transform_ensemble(
            EnsembleTable.from_table(table, num_members=0)
        )


@withCUDA
def test_reduce_estimators_composes_with_following_processor(
    device: torch.device,
) -> None:
    first = TableTensor.from_tensor(torch.tensor([[0.0, 2.0]], device=device))
    second = TableTensor.from_tensor(torch.tensor([[2.0, 0.0]], device=device))
    table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1),
    )

    output = sp.Sequential(
        sp.ReduceEstimators(), sp.Softmax()
    ).transform_ensemble(table)

    assert output.num_members == 1
    torch.testing.assert_close(
        output.table(0).numerical,
        torch.full((1, 2), 0.5, device=device),
    )
