from typing import Literal

import pytest
import torch

from sdm import (
    CategoricalTensor,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.processing import ShuffleColumns


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
            code=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
            categories=(
                StringTensor.from_list(["a", "b"]),
                StringTensor.from_list(["small", "large"]),
            ),
        ),
    )


def test_shuffle_columns_shift_rotates_numerical_block() -> None:
    table = _table()
    torch.manual_seed(3)  # draws a cyclic offset of 1 for three columns

    output = ShuffleColumns(method="shift").fit_transform(table)

    assert isinstance(output, TableTensor)
    assert output.columns[Stype.numerical] == ("x1", "x2", "x0")
    assert torch.equal(
        output.numerical,
        table.numerical.index_select(-1, torch.tensor([1, 2, 0])),
    )


@pytest.mark.parametrize("method", ["shift", "random"])
def test_shuffle_columns_is_reproducible_with_generator(
    method: Literal["shift", "random"],
) -> None:
    table = _table()

    first = ShuffleColumns(method=method)
    first_output = first.fit_transform(
        table,
        generator=torch.Generator().manual_seed(0),
    )
    second = ShuffleColumns(method=method)
    second_output = second.fit_transform(
        table,
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.equal(first.permutation, second.permutation)
    assert torch.equal(first_output.numerical, second_output.numerical)
