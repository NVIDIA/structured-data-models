# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import sdm.processing as sp
from sdm import EnsembleTable, TableTensor
from sdm.processing import AverageEstimators
from sdm.testing import withCUDA


@withCUDA
def test_average_estimators(device: torch.device) -> None:
    values = torch.arange(
        2 * 3 * 4 * 2,
        dtype=torch.float32,
        device=device,
    ).reshape(2, 3, 4, 2)
    table = TableTensor.from_tensor(values)
    processor = AverageEstimators()

    output = processor.transform(table)
    fit_output = processor.fit_transform(table)

    assert output.size() == (3, 4, 2)
    assert output.schema == table.schema
    assert output.dtype == values.dtype
    torch.testing.assert_close(output.numerical, values.mean(dim=0))
    torch.testing.assert_close(fit_output.numerical, output.numerical)
    assert repr(processor) == "AverageEstimators()"


def test_average_estimators_rejects_missing_ensemble_dimension() -> None:
    table = TableTensor.from_tensor(torch.ones(4, 2))

    with pytest.raises(ValueError, match="leading ensemble dimension"):
        AverageEstimators().transform(table)


@withCUDA
def test_acerage_estimators_trimmed(device: torch.device) -> None:
    values = torch.tensor(
        [
            [[-100.0, 100.0]],
            [[1.0, 4.0]],
            [[2.0, 3.0]],
            [[3.0, 2.0]],
            [[100.0, -100.0]],
        ],
        device=device,
    )
    processor = AverageEstimators(trim_fraction=0.2)

    output = processor.transform(TableTensor.from_tensor(values))

    torch.testing.assert_close(
        output.numerical,
        torch.tensor([[2.0, 3.0]], device=device),
    )


@withCUDA
def test_average_estimators_trimmed_reduces_members(
    device: torch.device,
) -> None:
    values = torch.tensor(
        [[[-100.0]], [[1.0]], [[2.0]], [[3.0]], [[100.0]]],
        device=device,
    )
    table = EnsembleTable.from_tables(
        tables=tuple(TableTensor.from_tensor(value) for value in values),
        member_table_ids=range(len(values)),
    )
    processor = AverageEstimators(trim_fraction=0.2)

    output = processor.transform_ensemble(table)

    assert len(output) == 1
    torch.testing.assert_close(
        output[0].numerical,
        torch.tensor([[2.0]], device=device),
    )


@withCUDA
def test_average_estimators_trimmed_keeps_all_members_when_untrimmed(
    device: torch.device,
) -> None:
    first = TableTensor.from_tensor(torch.tensor([[0.0, 2.0]], device=device))
    second = TableTensor.from_tensor(torch.tensor([[3.0, 1.0]], device=device))
    table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1, 1),
    )
    processor = AverageEstimators(trim_fraction=0.2)

    output = processor.transform_ensemble(table)

    torch.testing.assert_close(
        output[0].numerical,
        (first.numerical + 2 * second.numerical) / 3,
    )


@withCUDA
def test_average_estimators_reduces_across_storage_groups(
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

    output = AverageEstimators().transform_ensemble(table)

    assert output[0].columns == first.columns
    torch.testing.assert_close(
        output[0].numerical,
        torch.tensor([[4.0, 10.0 / 3.0]], device=device),
    )


@withCUDA
def test_average_estimators_composes_with_following_processor(
    device: torch.device,
) -> None:
    first = TableTensor.from_tensor(torch.tensor([[0.0, 2.0]], device=device))
    second = TableTensor.from_tensor(torch.tensor([[2.0, 0.0]], device=device))
    table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1),
    )

    output = sp.Sequential(
        sp.AverageEstimators(),
        sp.Softmax(),
    ).transform_ensemble(table)

    assert len(output) == 1
    torch.testing.assert_close(
        output[0].numerical,
        torch.full((1, 2), 0.5, device=device),
    )
