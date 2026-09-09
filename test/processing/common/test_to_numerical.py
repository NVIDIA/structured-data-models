import torch

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.processing import ToNumerical


def _table() -> TableTensor:
    return TableTensor(
        numerical=torch.tensor([[30.0, 100.0], [40.0, 200.0]]),
        categorical=CategoricalTensor(
            code=torch.tensor([[0, 1], [-1, 0]], dtype=torch.int64),
            categories=(
                StringTensor.from_list(["US", "DE"]),
                StringTensor.from_list(["small", "enterprise"]),
            ),
        ),
    )


def test_to_numerical_converts_categorical_stype() -> None:
    table = _table()

    categorical_ids = table.categorical.code.to(table.numerical.dtype)

    output = ToNumerical().transform(table)

    assert isinstance(output, TableTensor)
    assert output.columns[Stype.numerical] == (
        "num_0",
        "num_1",
        "cat_0",
        "cat_1",
    )
    assert output.columns[Stype.categorical] == ()
    assert torch.equal(output.numerical[..., :2], table.numerical)
    assert torch.equal(output.numerical[..., 2:], categorical_ids)
    assert output.categorical.size(-1) == 0


def test_to_numerical_is_identity_for_already_numerical_table() -> None:
    table = TableTensor.from_tensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    assert ToNumerical().transform(table) is table


def test_to_numerical_preserves_unhandled_stype() -> None:
    table = TableTensor(
        numerical=torch.tensor([[1.0], [2.0]]),
        id=ColumnarTensor((torch.arange(2),)),
    )

    output = ToNumerical().transform(table)

    assert output.columns[Stype.numerical] == ("num_0",)
    assert output.columns[Stype.id] == ("id_0",)
    assert torch.equal(output.id, table.id)


def test_to_numerical_converts_categorical_only_table() -> None:
    table = TableTensor(
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]], dtype=torch.int64),
            categories=(StringTensor.from_list(["US", "DE"]),),
        ),
    )

    output = ToNumerical().transform(table)

    assert isinstance(output, TableTensor)
    assert output.columns[Stype.numerical] == ("cat_0",)
    assert output.columns[Stype.categorical] == ()
    assert torch.equal(
        output.numerical,
        table.categorical.code.to(table.numerical.dtype),
    )


def test_to_numerical_missing_as_nan() -> None:
    output = ToNumerical(missing_as_nan=True).transform(_table())

    assert output.numerical[1, 2].isnan()
    assert output.numerical[1, 3] == 0.0
