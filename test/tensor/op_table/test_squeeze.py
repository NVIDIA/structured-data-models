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


def test_squeeze_overloads() -> None:
    inp = cast(TableTensor, aten.unsqueeze.default(make_table(), 0))
    reference = inp.numerical

    outputs = (
        aten.squeeze.default(inp),
        aten.squeeze.dim(inp, 0),
        aten.squeeze.dims(inp, (0,)),
    )
    expected = (
        aten.squeeze.default(reference),
        aten.squeeze.dim(reference, 0),
        aten.squeeze.dims(reference, (0,)),
    )

    for out, dense_out in zip(outputs, expected):
        assert_matches_dense(out, dense_out)
        assert out.columns == inp.columns
        assert (
            out.numerical.untyped_storage().data_ptr()
            == inp.numerical.untyped_storage().data_ptr()
        )
