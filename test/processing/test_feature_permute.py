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


def test_feature_permute_shift_rotates_numerical_block() -> None:
    table = _table()
    torch.manual_seed(3)  # draws a cyclic offset of 1 for three columns

    output = FeaturePermute(method="shift").fit_transform(table)

    assert isinstance(output, TableTensor)
    assert output.columns[Stype.numerical] == ("x1", "x2", "x0")
    assert torch.equal(
        output.numerical,
        table.numerical.index_select(-1, torch.tensor([1, 2, 0])),
    )


def test_feature_permute_random_inverse_round_trips() -> None:
    table = _table()
    pipeline = Sequential(FeaturePermute(method="random"))

    transformed = pipeline.fit_transform(table)
    restored = pipeline.inverse_transform(transformed)

    assert isinstance(restored, TableTensor)
    assert restored.columns == table.columns
    assert torch.equal(restored.numerical, table.numerical)


def test_feature_permute_composes_after_to_numerical() -> None:
    table = _mixed_table()
    torch.manual_seed(3)  # draws a cyclic offset of 1 for five columns

    output = Sequential(ToNumerical(), FeaturePermute(method="shift"))
    output = output.fit_transform(table)
    converted = ToNumerical().fit_transform(table)

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
