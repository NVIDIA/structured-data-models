import pytest
import torch
from sdm import Stype, TableTensor
from sdm.processing import ConstantFilter, Recipe, StandardScale
from sdm.testing import withCUDA


@withCUDA
def test_unique_filter(device: torch.device) -> None:
    data = torch.tensor(
        [
            [1.0, 0.0, 3.0, 5.0, torch.nan],
            [1.0, 1.0, 3.0, torch.nan, torch.nan],
            [1.0, 2.0, 4.0, 5.0, torch.nan],
            [1.0, 3.0, 4.0, torch.nan, torch.nan],
        ],
        device=device,
    )
    table = TableTensor.from_tensor(
        data,
        columns=(
            "constant",
            "variable",
            "two_values",
            "value_and_nan",
            "all_nan",
        ),
    )

    output = ConstantFilter().fit_transform(table)

    assert output.columns[Stype.numerical] == (
        "variable",
        "two_values",
        "value_and_nan",
    )
    assert output.numerical.allclose(data[:, [1, 2, 3]], equal_nan=True)
    assert output.device == device


@withCUDA
def test_unique_filter_with_higher_threshold(device: torch.device) -> None:
    table = TableTensor.from_tensor(
        torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 2.0, torch.nan],
                [1.0, 1.0, 2.0],
                [1.0, 2.0, torch.nan],
            ],
            device=device,
        ),
        columns=("one", "two", "three"),
    )

    output = ConstantFilter(threshold=2).fit_transform(table)

    assert output.columns[Stype.numerical] == ("three",)


def test_unique_filter_keeps_all_columns_with_too_few_rows() -> None:
    table = TableTensor.from_tensor(
        torch.tensor([[1.0, torch.nan], [1.0, torch.nan]])
    )

    assert ConstantFilter(threshold=2).fit_transform(table) is table


def test_fit_is_2d_and_transform_supports_batches() -> None:
    train = TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0], [1.0, 3.0]]),
        columns=("constant", "variable"),
    )
    batched = TableTensor.from_tensor(
        torch.tensor([[[4.0, 5.0]], [[6.0, 7.0]]]),
        columns=("constant", "variable"),
    )

    processor = ConstantFilter().fit(train)
    output = processor.transform(batched)

    assert output.size() == (2, 1, 1)
    assert output.columns[Stype.numerical] == ("variable",)
    with pytest.raises(ValueError, match="two-dimensional"):
        ConstantFilter().fit(batched)


@withCUDA
def test_variance_filter(device: torch.device) -> None:
    data = torch.tensor(
        [
            [1.0, 1.0, 1.0, torch.nan],
            [1.0, 1.0000005, 2.0, 2.0],
            [1.0, 0.9999995, 3.0, 3.0],
            [1.0, 1.0000002, 4.0, 4.0],
        ],
        dtype=torch.float64,
        device=device,
    )
    table = TableTensor.from_tensor(
        data,
        columns=("constant", "near_constant", "variable", "has_nan"),
    )

    output = ConstantFilter(method="variance").fit_transform(table)

    assert output.columns[Stype.numerical] == ("variable",)
    assert output.numerical.equal(data[:, [2]])
    assert output.device == device


def test_filter_is_scoped_to_recipe_fit_data() -> None:
    train = TableTensor.from_tensor(
        torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]),
        columns=("train_constant", "variable"),
    )
    later = TableTensor.from_tensor(
        torch.tensor([[10.0, 4.0], [20.0, 5.0]]),
        columns=("train_constant", "variable"),
    )
    recipe = Recipe(features=[ConstantFilter(), StandardScale()])

    recipe.features.fit(train)
    output = recipe.features.transform(later)

    assert output.columns[Stype.numerical] == ("variable",)


def test_variance_rejects_integer_input() -> None:
    table = TableTensor.from_tensor(torch.tensor([[1, 2], [1, 3]]))

    with pytest.raises(TypeError, match="floating-point"):
        ConstantFilter(method="variance").fit(table)


def test_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="method must be"):
        ConstantFilter(method="bad")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="threshold must be"):
        ConstantFilter(threshold=0)
    with pytest.raises(ValueError, match="tolerance must be"):
        ConstantFilter(tolerance=-1.0)
