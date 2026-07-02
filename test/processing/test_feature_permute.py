import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.processing import FeaturePermute, Pipeline


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
    pipeline = Pipeline([FeaturePermute(method="shift").resolve(estimator=1)])

    transformed = pipeline.transform(table)
    restored = pipeline.inverse_transform(transformed)

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
    pipeline = Pipeline(
        [
            FeaturePermute(method="random", generator=generator).resolve(
                estimator=1
            )
        ]
    )

    transformed = pipeline.transform(table)
    restored = pipeline.inverse_transform(transformed)

    assert restored.columns == table.columns
    assert torch.equal(restored.numerical, table.numerical)
    assert torch.equal(
        restored.categorical.as_tensor(), table.categorical.as_tensor()
    )


def test_feature_permute_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match="method must be"):
        FeaturePermute(method="bad")  # ty: ignore[invalid-argument-type]


def test_feature_permute_requires_table_input() -> None:
    with pytest.raises(TypeError, match="TableTensor"):
        FeaturePermute().transform(torch.ones(2, 3))
