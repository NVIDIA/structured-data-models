import pytest
import torch

from sdm import Stype, TableTensor
from sdm.processing import DropConstantColumns
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

    output = DropConstantColumns(threshold=2).fit_transform(table)

    assert output.columns == table.columns
    torch.testing.assert_close(output.numerical, table.numerical)


def test_unique_filter_batch_learns_each_representation() -> None:
    context = TableTensor.from_tensor(
        torch.tensor(
            [
                [[1.0, 1.0, 3.0], [1.0, 2.0, 3.0]],
                [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]],
            ]
        ),
        columns=("first", "second", "third"),
    )
    query = TableTensor.from_tensor(
        torch.tensor(
            [
                [[4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0]],
            ]
        ),
        columns=("first", "second", "third"),
    )

    processor = DropConstantColumns().fit_batch(context)
    first, second = processor.transform_batch(query)

    assert first.columns[Stype.numerical] == ("second",)
    assert second.columns[Stype.numerical] == ("first",)
    torch.testing.assert_close(first.numerical, query[0].numerical[:, 1:2])
    torch.testing.assert_close(second.numerical, query[1].numerical[:, :1])


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
