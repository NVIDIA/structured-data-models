import pytest
import torch
from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.processing import FeaturePermute, Sequential


def _table() -> TableTensor:
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


def test_feature_permute_default_single_estimator_is_identity() -> None:
    table = _table()

    output = FeaturePermute().transform(table)

    assert output is table


def test_feature_permute_resolved_shift_permutes_table_blocks() -> None:
    table = _table()
    processor = FeaturePermute(method="shift").resolve(estimator=1)

    output = processor.transform(table)

    assert isinstance(output, TableTensor)
    assert output.columns[Stype.numerical] == ("x1", "x2", "x0")
    assert output.columns[Stype.categorical] == ("segment", "kind")
    assert torch.equal(
        output.numerical,
        table.numerical.index_select(-1, torch.tensor([1, 2, 0])),
    )
    assert torch.equal(
        output.categorical.as_tensor(),
        table.categorical.as_tensor().index_select(-1, torch.tensor([1, 0])),
    )


def test_feature_permute_inverse_restores_permuted_blocks() -> None:
    table = _table()
    pipeline = Sequential(FeaturePermute(method="shift").resolve(estimator=1))

    transformed = pipeline.transform(table)
    restored = pipeline.inverse_transform(transformed)

    assert isinstance(restored, TableTensor)
    assert restored.columns == table.columns
    assert torch.equal(restored.numerical, table.numerical)
    assert torch.equal(
        restored.categorical.as_tensor(), table.categorical.as_tensor()
    )


def test_feature_permute_random_is_deterministic() -> None:
    table = _table()
    generator = torch.Generator().manual_seed(123)
    processor = FeaturePermute(method="random", generator=generator).resolve(
        estimator=1
    )

    first = processor.transform(table)
    second = processor.transform(table)

    assert isinstance(first, TableTensor)
    assert isinstance(second, TableTensor)
    assert first.columns == second.columns
    assert torch.equal(first.numerical, second.numerical)
    assert torch.equal(
        first.categorical.as_tensor(), second.categorical.as_tensor()
    )


def test_feature_permute_random_inverse_round_trips() -> None:
    table = _table()
    generator = torch.Generator().manual_seed(123)
    pipeline = Sequential(
        FeaturePermute(method="random", generator=generator).resolve(
            estimator=1
        )
    )

    transformed = pipeline.transform(table)
    restored = pipeline.inverse_transform(transformed)

    assert isinstance(restored, TableTensor)
    assert restored.columns == table.columns
    assert torch.equal(restored.numerical, table.numerical)
    assert torch.equal(
        restored.categorical.as_tensor(), table.categorical.as_tensor()
    )


def test_feature_permute_preserves_all_stype_blocks() -> None:
    table = TableTensor(
        columns={
            "datetime": ("created", "updated"),
            "id": ("user_id", "item_id"),
        },
        datetime=torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        id=ColumnarTensor(
            (
                torch.tensor([10, 20]),
                torch.tensor([30, 40]),
            )
        ),
    )

    output = (
        FeaturePermute(method="shift").resolve(estimator=1).transform(table)
    )

    assert output.columns[Stype.datetime] == ("updated", "created")
    assert output.columns[Stype.id] == ("item_id", "user_id")
    assert torch.equal(output.datetime, table.datetime[:, [1, 0]])
    assert output.id.tolist() == [[30, 10], [40, 20]]


def test_feature_permute_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match="method must be"):
        FeaturePermute(method="bad")  # ty: ignore[invalid-argument-type]
