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


@withCUDA
def test_reduce_estimators_trimmed_drops_one_member_per_end(
    device: torch.device,
) -> None:
    values = torch.randn(8, 3, 2, device=device)
    values[:, 0, 0] = torch.tensor(
        [5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 100.0], device=device
    )
    table = TableTensor.from_tensor(values)
    processor = sp.ReduceEstimators(method="trimmed")

    output = processor.transform(table)

    assert output.size() == (3, 2)
    assert output.schema == table.schema
    assert output.dtype == values.dtype
    expected = values.sort(dim=0).values[1:7].mean(dim=0)
    torch.testing.assert_close(output.numerical, expected)
    torch.testing.assert_close(
        output.numerical[0, 0],
        torch.tensor((2.0 + 3.0 + 5.0 + 7.0 + 8.0 + 9.0) / 6, device=device),
    )


def test_reduce_estimators_trimmed_is_mean_below_five_members() -> None:
    values = torch.randn(4, 3, 2)
    table = TableTensor.from_tensor(values)

    output = sp.ReduceEstimators(method="trimmed").transform(table)

    torch.testing.assert_close(output.numerical, values.mean(dim=0))


@withCUDA
def test_reduce_estimators_trimmed_ensemble_matches_tensor_path(
    device: torch.device,
) -> None:
    values = torch.randn(8, 3, 2, device=device)
    processor = sp.ReduceEstimators(method="trimmed")
    stacked = processor.transform(TableTensor.from_tensor(values))
    ensemble_table = EnsembleTable.from_tables(
        tables=tuple(TableTensor.from_tensor(member) for member in values),
        member_table_ids=tuple(range(8)),
    )

    output = processor.transform_ensemble(ensemble_table)

    assert output.num_members == 1
    torch.testing.assert_close(output.table(0).numerical, stacked.numerical)


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
