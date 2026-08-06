import pytest
import torch

from sdm import TableTensor
from sdm.processing import (
    ReduceEstimators,
    Sequential,
    Softmax,
)
from sdm.tensor import EnsembleTable
from sdm.testing import withCUDA


@withCUDA
def test_reduce_estimators_mean(device: torch.device) -> None:
    values = torch.arange(
        2 * 3 * 4 * 2,
        dtype=torch.float32,
        device=device,
    ).reshape(2, 3, 4, 2)
    table = TableTensor.from_tensor(values, columns=("a", "b"))
    processor = ReduceEstimators(method="mean")

    output = processor.transform(table)
    fit_output = processor.fit_transform(table)

    assert output.size() == (3, 4, 2)
    assert output.schema == table.schema
    assert output.device == device
    assert output.dtype == values.dtype
    torch.testing.assert_close(output.numerical, values.mean(dim=0))
    torch.testing.assert_close(fit_output.numerical, output.numerical)
    assert repr(processor) == "ReduceEstimators(method='mean')"


def test_reduce_estimators_rejects_missing_ensemble_dimension() -> None:
    table = TableTensor.from_tensor(torch.ones(4, 2))

    with pytest.raises(ValueError, match="leading ensemble dimension"):
        ReduceEstimators().transform(table)


def test_reduce_estimators_rejects_empty_ensemble_dimension() -> None:
    table = TableTensor.from_tensor(torch.empty(0, 4, 2))

    with pytest.raises(ValueError, match="at least one ensemble member"):
        ReduceEstimators().transform(table)


def test_reduce_estimators_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method must be 'mean'"):
        ReduceEstimators(method="median")  # type: ignore


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

    output = ReduceEstimators().transform_ensemble(table)

    assert output.num_members == 1
    assert output.table(0).device == device
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

    output = ReduceEstimators().transform_ensemble(table)

    assert output.table(0).columns == first.columns
    torch.testing.assert_close(
        output.table(0).numerical,
        torch.tensor([[4.0, 10.0 / 3.0]], device=device),
    )


def test_reduce_estimators_rejects_empty_ensemble_table() -> None:
    table = TableTensor.from_tensor(torch.ones(4, 2))

    with pytest.raises(ValueError, match="at least one ensemble member"):
        ReduceEstimators().transform_ensemble(
            EnsembleTable(table, num_members=0)
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

    output = Sequential(ReduceEstimators(), Softmax()).transform_ensemble(
        table
    )

    assert output.num_members == 1
    assert output.table(0).device == device
    torch.testing.assert_close(
        output.table(0).numerical,
        torch.full((1, 2), 0.5, device=device),
    )
