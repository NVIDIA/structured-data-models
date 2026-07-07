import pytest
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
        columns={
            "numerical": ("age", "income"),
            "categorical": ("country", "segment"),
        },
        numerical=torch.tensor([[30.0, 100.0], [40.0, 200.0]]),
        categorical=CategoricalTensor(
            data=torch.tensor([[0, 1], [-1, 0]], dtype=torch.int64),
            categories=(
                StringTensor.from_list(["US", "DE"]),
                StringTensor.from_list(["small", "enterprise"]),
            ),
        ),
    )


def test_to_numerical_converts_categorical_stype() -> None:
    table = _table()

    categorical_ids = table.categorical.as_tensor().to(table.numerical.dtype)

    output = ToNumerical().transform(table)

    assert isinstance(output, TableTensor)
    assert output.columns[Stype.numerical] == (
        "age",
        "income",
        "country",
        "segment",
    )
    assert output.columns[Stype.categorical] == ()
    assert torch.equal(output.numerical[..., :2], table.numerical)
    assert torch.equal(output.numerical[..., 2:], categorical_ids)
    assert output.categorical.size(-1) == 0


def test_to_numerical_is_identity_for_already_numerical_table() -> None:
    table = TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        columns=("x0", "x1"),
    )

    assert ToNumerical().transform(table) is table


def test_to_numerical_rejects_unsupported_stype() -> None:
    table = TableTensor(
        columns={"id": ("row_id",)},
        id=ColumnarTensor((torch.arange(2),)),
    )

    with pytest.raises(ValueError, match="id"):
        ToNumerical().transform(table)


def test_to_numerical_keeps_source_category_vocabulary_available() -> None:
    table = _table()

    output = ToNumerical().transform(table)

    assert isinstance(output, TableTensor)
    assert table.categorical.categories[0].tolist() == ["US", "DE"]
    assert table.categorical.categories[1].tolist() == [
        "small",
        "enterprise",
    ]
    assert output.columns[Stype.categorical] == ()


def test_to_numerical_converts_categorical_only_table() -> None:
    table = TableTensor(
        columns={"categorical": ("country",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1]], dtype=torch.int64),
            categories=(StringTensor.from_list(["US", "DE"]),),
        ),
    )

    output = ToNumerical().transform(table)

    assert isinstance(output, TableTensor)
    assert output.columns[Stype.numerical] == ("country",)
    assert output.columns[Stype.categorical] == ()
    assert torch.equal(
        output.numerical,
        table.categorical.as_tensor().to(table.numerical.dtype),
    )
