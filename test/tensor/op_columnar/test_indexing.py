from typing import cast

import torch
from torch import Tensor

from sdm import ColumnarTensor

aten = torch.ops.aten


def make_columnar() -> ColumnarTensor:
    return ColumnarTensor(
        (
            torch.arange(24).view(2, 3, 4),
            torch.arange(100, 124).view(2, 3, 4),
        ),
    )


def dense(inp: ColumnarTensor) -> Tensor:
    return torch.tensor(inp.tolist())


def assert_matches_dense(out: ColumnarTensor, expected: Tensor) -> None:
    assert type(out) is ColumnarTensor
    assert out.size() == expected.size()
    assert out.stride() == expected.stride()
    assert out.storage_offset() == expected.storage_offset()
    assert out.tolist() == expected.tolist()


def test_index_overloads() -> None:
    inp = cast(ColumnarTensor, make_columnar().transpose(0, 1))
    reference = dense(inp)
    index = torch.tensor([2, 0])

    selected = aten.index_select.default(inp, 0, index)
    advanced = aten.index.Tensor(inp, (index,))

    assert_matches_dense(
        selected,
        aten.index_select.default(reference, 0, index),
    )
    assert_matches_dense(
        advanced,
        aten.index.Tensor(reference, (index,)),
    )


def test_column_index() -> None:
    inp = make_columnar()
    index = torch.tensor([1, 0])

    out = aten.index_select.default(inp, -1, index)

    assert_matches_dense(
        out,
        aten.index_select.default(dense(inp), -1, index),
    )
