import pytest
import torch
from sdm import (
    CategoricalTensor,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.processing import FeaturePermute, Sequential, ToNumerical


def _table() -> TableTensor:
    return TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        columns=("x0", "x1", "x2"),
    )


def _mixed_table() -> TableTensor:
    return TableTensor(
        columns={
            "numerical": ("x0", "x1", "x2"),
            "categorical": ("kind", "segment"),
        },
        numerical=torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ),
        categorical=CategoricalTensor(
            data=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
            categories=(
                StringTensor.from_list(["a", "b"]),
                StringTensor.from_list(["small", "large"]),
            ),
        ),
    )


def test_feature_permute_none_is_identity() -> None:
    table = _table()

    output = FeaturePermute(method="none").transform(table)

    assert output is table


def test_feature_permute_shift_rotates_numerical_block() -> None:
    table = _table()
    torch.manual_seed(0)  # draws a cyclic offset of 1 for three columns
    processor = FeaturePermute(method="shift")

    output = processor.transform(table)

    assert isinstance(output, TableTensor)
    assert output.columns[Stype.numerical] == ("x1", "x2", "x0")
    assert torch.equal(
        output.numerical,
        table.numerical.index_select(-1, torch.tensor([1, 2, 0])),
    )


def test_feature_permute_same_global_seed_draws_same_view() -> None:
    table = _table()

    torch.manual_seed(123)
    first = FeaturePermute(method="random").transform(table)
    torch.manual_seed(123)
    second = FeaturePermute(method="random").transform(table)

    assert isinstance(first, TableTensor)
    assert isinstance(second, TableTensor)
    assert first.columns == second.columns
    assert torch.equal(first.numerical, second.numerical)


def test_feature_permute_transform_is_deterministic_per_instance() -> None:
    table = _table()
    processor = FeaturePermute(method="random")

    first = processor.transform(table)
    second = processor.transform(table)

    assert first.columns == second.columns
    assert torch.equal(first.numerical, second.numerical)


def test_feature_permute_shift_inverse_restores_numerical_block() -> None:
    table = _table()
    pipeline = Sequential(FeaturePermute(method="shift"))

    transformed = pipeline.transform(table)
    restored = pipeline.inverse_transform(transformed)

    assert isinstance(restored, TableTensor)
    assert restored.columns == table.columns
    assert torch.equal(restored.numerical, table.numerical)


def test_feature_permute_random_inverse_round_trips() -> None:
    table = _table()
    pipeline = Sequential(FeaturePermute(method="random"))

    transformed = pipeline.transform(table)
    restored = pipeline.inverse_transform(transformed)

    assert isinstance(restored, TableTensor)
    assert restored.columns == table.columns
    assert torch.equal(restored.numerical, table.numerical)


def test_feature_permute_rejects_non_numerical_columns() -> None:
    with pytest.raises(ValueError, match="categorical"):
        FeaturePermute().transform(_mixed_table())


def test_feature_permute_composes_after_to_numerical() -> None:
    table = _mixed_table()
    torch.manual_seed(0)  # draws a cyclic offset of 1 for five columns
    processor = FeaturePermute(method="shift")

    output = Sequential(ToNumerical(), processor).transform(table)
    converted = ToNumerical().transform(table)

    assert output.columns[Stype.numerical] == (
        "x1",
        "x2",
        "kind",
        "segment",
        "x0",
    )
    assert output.columns[Stype.categorical] == ()
    assert torch.equal(
        output.numerical,
        converted.numerical.index_select(-1, torch.tensor([1, 2, 3, 4, 0])),
    )


def test_feature_permute_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match="method must be"):
        FeaturePermute(method="bad")  # ty: ignore[invalid-argument-type]
