import pytest
import torch

from sdm import EnsembleTable, Stype, TableTensor
from sdm.processing import DropConstantColumns, EnsembleProcessor
from sdm.testing import withCUDA


@withCUDA
def test_unique_filter(device: torch.device) -> None:
    data = torch.tensor(
        [
            [1.0, 0.0, 3.0, 5.0, 9.0],
            [1.0, 1.0, 3.0, 6.0, 9.0],
            [1.0, 2.0, 4.0, 5.0, 9.0],
            [1.0, 3.0, 4.0, 6.0, 9.0],
        ],
        device=device,
    )
    table = TableTensor.from_tensor(
        data,
        columns=(
            "constant",
            "variable",
            "two_values",
            "two_more_values",
            "also_constant",
        ),
    )

    output = DropConstantColumns().fit_transform(table)

    assert output.columns[Stype.numerical] == (
        "variable",
        "two_values",
        "two_more_values",
    )
    assert output.numerical.equal(data[:, [1, 2, 3]])
    assert output.device == device


@withCUDA
def test_unique_filter_with_higher_threshold(device: torch.device) -> None:
    table = TableTensor.from_tensor(
        torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 2.0, 3.0],
                [1.0, 1.0, 2.0],
                [1.0, 2.0, 3.0],
            ],
            device=device,
        ),
        columns=("one", "two", "three"),
    )

    output = DropConstantColumns(threshold=2).fit_transform(table)

    assert output.columns[Stype.numerical] == ("three",)


def test_unique_filter_keeps_all_columns_with_too_few_rows() -> None:
    table = TableTensor.from_tensor(torch.tensor([[1.0, 4.0], [1.0, 4.0]]))

    assert DropConstantColumns(threshold=2).fit_transform(table) is table


@withCUDA
def test_variance_filter(device: torch.device) -> None:
    data = torch.tensor(
        [
            [1.0, 1.0, 1.0, 8.0],
            [1.0, 1.0000005, 2.0, 8.0],
            [1.0, 0.9999995, 3.0, 8.0],
            [1.0, 1.0000002, 4.0, 8.0],
        ],
        dtype=torch.float64,
        device=device,
    )
    table = TableTensor.from_tensor(
        data,
        columns=("constant", "near_constant", "variable", "also_constant"),
    )

    output = DropConstantColumns(method="variance").fit_transform(table)

    assert output.columns[Stype.numerical] == ("variable",)
    assert output.numerical.equal(data[:, [2]])
    assert output.device == device


def test_drop_constant_columns_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="method must be"):
        DropConstantColumns(method="invalid")  # type: ignore
    with pytest.raises(ValueError, match="tolerance must be None"):
        DropConstantColumns(tolerance=1e-6)
    with pytest.raises(ValueError, match="threshold must be None"):
        DropConstantColumns(method="variance", threshold=1)
    with pytest.raises(ValueError, match="threshold must be positive"):
        DropConstantColumns(threshold=0)
    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        DropConstantColumns(method="variance", tolerance=-1.0)


def test_drop_constant_columns_is_an_ensemble_processor() -> None:
    assert issubclass(DropConstantColumns, EnsembleProcessor)


def test_drop_constant_columns_splits_different_fitted_schemas() -> None:
    first = TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0], [1.0, 3.0]]),
        columns=("a", "b"),
    )
    second = TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0], [3.0, 2.0]]),
        columns=("a", "b"),
    )
    context = EnsembleTable.from_representations(
        (first, second),
        member_representation_ids=(0, 1, 0, 1),
    )
    processor = DropConstantColumns()

    transformed = processor.fit_transform_ensemble(context)

    assert transformed.representation(0).columns[Stype.numerical] == ("b",)
    assert transformed.representation(1).columns[Stype.numerical] == ("a",)
    assert transformed.representation(2).equal(transformed.representation(0))
    assert transformed.representation(3).equal(transformed.representation(1))

    query = TableTensor.from_tensor(
        torch.tensor([[4.0, 5.0], [6.0, 7.0]]),
        columns=("a", "b"),
    )
    query_output = processor.transform_ensemble(
        EnsembleTable.from_representations(
            (query, query),
            member_representation_ids=(0, 1, 0, 1),
        )
    )
    assert query_output.representation(0).columns[Stype.numerical] == ("b",)
    assert query_output.representation(1).columns[Stype.numerical] == ("a",)


def test_drop_constant_columns_keeps_shared_output_packed() -> None:
    table = TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0], [1.0, 3.0]]),
        columns=("a", "b"),
    )
    output = DropConstantColumns().fit_transform_ensemble(
        EnsembleTable(table, num_members=8)
    )

    assert (
        sum(packed.size(0) for packed in output.iter_packed_representations())
        == 1
    )
