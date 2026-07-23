import pytest
import torch
from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.processing import DropConstantColumns
from sdm.testing import withCUDA


@withCUDA
def test_drop_constant_columns_exact_values(device: torch.device) -> None:
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
def test_drop_constant_columns_uses_tolerance(
    device: torch.device,
) -> None:
    data = torch.tensor(
        [
            [1.0, 1.0, 8.0],
            [1.0, 1.0000005, 8.0],
            [1.0, 0.9999995, 8.0],
            [1.0, 1.0000002, 9.0],
        ],
        dtype=torch.float64,
        device=device,
    )
    table = TableTensor.from_tensor(
        data,
        columns=("constant", "near_constant", "variable"),
    )

    output = DropConstantColumns(rtol=1e-5, atol=1e-8).fit_transform(table)

    assert output.columns[Stype.numerical] == ("variable",)
    assert output.numerical.equal(data[:, [2]])
    assert output.device == device


def test_drop_constant_columns_honors_equal_nan() -> None:
    table = TableTensor.from_tensor(
        torch.tensor(
            [
                [float("nan"), float("nan"), 1.0],
                [float("nan"), 2.0, float("nan")],
            ]
        ),
        columns=("all_nan", "mixed_nan", "different"),
    )

    output = DropConstantColumns(equal_nan=True).fit_transform(table)

    assert output.columns[Stype.numerical] == ("mixed_nan", "different")


def test_drop_constant_columns_drops_exact_columns_across_stypes() -> None:
    table = TableTensor(
        columns={
            "numerical": ("x_const", "x"),
            "categorical": ("cat_const", "cat"),
            "datetime": ("time_const", "time"),
            "text": ("text_const", "text"),
            "id": ("id_const", "id"),
        },
        numerical=torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]),
        categorical=CategoricalTensor(
            data=torch.tensor([[0, 0], [0, 1], [0, 0]], dtype=torch.int32),
            categories=(
                StringTensor.from_list(["a"]),
                StringTensor.from_list(["x", "y"]),
            ),
        ),
        datetime=torch.tensor([[7, 1], [7, 2], [7, 3]]),
        text=StringTensor.from_list(
            [["same", "a"], ["same", "b"], ["same", "a"]]
        ),
        id=ColumnarTensor(
            (
                torch.tensor([10, 10, 10]),
                StringTensor.from_list(["u1", "u2", "u3"]),
            )
        ),
    )

    output = DropConstantColumns().fit_transform(table)

    assert output.columns[Stype.numerical] == ("x",)
    assert output.columns[Stype.categorical] == ("cat",)
    assert output.columns[Stype.datetime] == ("time",)
    assert output.columns[Stype.text] == ("text",)
    assert output.columns[Stype.id] == ("id",)


def test_drop_constant_columns_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="rtol must be non-negative"):
        DropConstantColumns(rtol=-1.0)
    with pytest.raises(ValueError, match="atol must be non-negative"):
        DropConstantColumns(atol=-1.0)
