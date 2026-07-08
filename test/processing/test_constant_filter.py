import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.processing import ConstantFilter, Sequential, StandardScale
from sdm.testing import withCUDA


@withCUDA
def test_unique_matches_tabicl_reference(device: torch.device) -> None:
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

    output = ConstantFilter(method="unique", threshold=1).fit_transform(table)

    assert output.columns[Stype.numerical] == (
        "variable",
        "two_values",
        "value_and_nan",
    )
    assert torch.allclose(output.numerical, data[:, [1, 2, 3]], equal_nan=True)
    assert output.device == device


def test_unique_counts_nan_as_one_category() -> None:
    table = TableTensor.from_tensor(
        torch.tensor([[7.0], [torch.nan], [7.0], [torch.nan]]),
        columns=("value_and_nan",),
    )

    kept = ConstantFilter(threshold=1).fit_transform(table)
    removed = ConstantFilter(threshold=2).fit_transform(table)

    assert kept is table
    assert removed.size(-1) == 0


@withCUDA
def test_unique_higher_threshold_uses_distinct_value_count(
    device: torch.device,
) -> None:
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


def test_unique_keeps_all_columns_for_few_samples() -> None:
    table = TableTensor.from_tensor(
        torch.tensor([[1.0, torch.nan], [1.0, torch.nan]]),
        columns=("constant", "all_nan"),
    )

    output = ConstantFilter(threshold=2).fit_transform(table)

    assert output is table


def test_unique_threshold_zero_keeps_all_columns() -> None:
    table = TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0], [1.0, 3.0]]),
        columns=("constant", "variable"),
    )

    assert ConstantFilter(threshold=0).fit_transform(table) is table


@withCUDA
def test_variance_matches_rfm_reference(device: torch.device) -> None:
    data = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0000005, 2.0],
            [1.0, 0.9999995, 3.0],
            [1.0, 1.0000002, 4.0],
        ],
        dtype=torch.float64,
        device=device,
    )
    table = TableTensor.from_tensor(
        data, columns=("constant", "near_constant", "variable")
    )
    reference_mask = data.std(dim=0) > 1e-6

    output = ConstantFilter(method="variance", tolerance=1e-6).fit_transform(
        table
    )

    assert output.columns[Stype.numerical] == ("variable",)
    assert torch.equal(output.numerical, data[:, reference_mask])
    assert output.device == device


def test_variance_nan_behavior_matches_rfm_reference() -> None:
    data = torch.tensor([[torch.nan, 1.0], [2.0, 2.0], [3.0, 3.0]])
    table = TableTensor.from_tensor(data, columns=("has_nan", "variable"))
    reference_mask = data.std(dim=0) > 1e-6

    output = ConstantFilter(method="variance").fit_transform(table)

    assert output.columns[Stype.numerical] == ("variable",)
    assert torch.equal(output.numerical, data[:, reference_mask])


def test_fit_uses_context_columns_for_later_transforms() -> None:
    train = TableTensor.from_tensor(
        torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]),
        columns=("train_constant", "variable"),
    )
    test = TableTensor.from_tensor(
        torch.tensor([[10.0, 4.0], [20.0, 5.0]]),
        columns=("train_constant", "variable"),
    )

    processor = ConstantFilter().fit(train)
    output = processor.transform(test)

    assert output.columns[Stype.numerical] == ("variable",)
    assert torch.equal(output.numerical, test.numerical[:, [1]])


def test_noop_returns_input_table() -> None:
    table = TableTensor.from_tensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    assert ConstantFilter().fit_transform(table) is table


def test_composes_before_numerical_processor() -> None:
    table = TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0], [1.0, 5.0], [1.0, 8.0]]),
        columns=("constant", "variable"),
    )

    output = Sequential(ConstantFilter(), StandardScale()).fit_transform(table)

    assert output.columns[Stype.numerical] == ("variable",)
    assert torch.allclose(
        output.numerical.mean(dim=0), torch.zeros(1), atol=1e-6
    )


def test_rejects_non_numerical_columns() -> None:
    table = TableTensor(
        columns={"categorical": ("kind",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1]]),
            categories=(StringTensor.from_list(["a", "b"]),),
        ),
    )

    with pytest.raises(ValueError, match="categorical"):
        ConstantFilter().fit(table)


def test_variance_rejects_integer_input() -> None:
    table = TableTensor.from_tensor(torch.tensor([[1, 2], [1, 3]]))

    with pytest.raises(TypeError, match="floating-point"):
        ConstantFilter(method="variance").fit(table)


def test_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match="method must be"):
        ConstantFilter(method="bad")  # ty: ignore[invalid-argument-type]


def test_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be"):
        ConstantFilter(threshold=-1)


def test_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance must be"):
        ConstantFilter(tolerance=-1.0)
