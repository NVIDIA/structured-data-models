from typing import cast

import torch
from torch import Tensor

from sdm import TableTensor

aten = torch.ops.aten


def make_table() -> TableTensor:
    return TableTensor(
        columns={"numerical": ("first", "second")},
        numerical=torch.arange(48.0).view(2, 3, 4, 2),
    )


def assert_matches_dense(out: TableTensor, expected: Tensor) -> None:
    assert type(out) is TableTensor
    assert out.size() == expected.size()
    assert out.stride() == expected.stride()
    assert out.storage_offset() == expected.storage_offset()
    assert out.layout == expected.layout
    assert out.numerical.equal(expected)


def test_clone_preserves_layout_and_copies_storage() -> None:
    inp = cast(TableTensor, make_table().transpose(0, 1))

    out = aten.clone.default(inp)

    assert_matches_dense(out, aten.clone.default(inp.numerical))
    assert not out.is_contiguous()
    assert (
        out.numerical.untyped_storage().data_ptr()
        != inp.numerical.untyped_storage().data_ptr()
    )
