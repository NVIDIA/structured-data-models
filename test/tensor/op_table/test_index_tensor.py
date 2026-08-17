from typing import cast

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


def test_advanced_index_preserves_layout() -> None:
    inp = cast(TableTensor, make_table().transpose(0, 1))
    index = torch.tensor([2, 0])

    out = aten.index.Tensor(inp, (index,))

    assert_matches_dense(
        out,
        aten.index.Tensor(inp.numerical, (index,)),
    )
    assert out.columns == inp.columns
