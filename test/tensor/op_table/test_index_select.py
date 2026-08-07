from typing import cast

import pytest
import torch
from torch import Tensor

from sdm import TableTensor

aten = torch.ops.aten


def make_table() -> TableTensor:
    return TableTensor(
        columns={"numerical": ("first", "second")},
        numerical=torch.arange(48).view(2, 3, 4, 2),
    )


def assert_matches_dense(out: TableTensor, expected: Tensor) -> None:
    assert type(out) is TableTensor
    assert out.size() == expected.size()
    assert out.stride() == expected.stride()
    assert out.storage_offset() == expected.storage_offset()
    assert out.numerical.equal(expected)


def test_index_select_preserves_layout() -> None:
    inp = cast(TableTensor, make_table().transpose(0, 1))
    index = torch.tensor([2, 0])

    out = aten.index_select.default(inp, 0, index)

    assert_matches_dense(
        out,
        aten.index_select.default(inp.numerical, 0, index),
    )
    assert out.columns == inp.columns


def test_index_select_rejects_column_dimension() -> None:
    inp = make_table()

    with pytest.raises(RuntimeError, match="column dimension"):
        aten.index_select.default(inp, -1, torch.tensor([0]))
