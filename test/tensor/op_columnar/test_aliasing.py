from typing import cast

import torch

from sdm import ColumnarTensor

aten = torch.ops.aten


def test_alias_and_detach_preserve_aliases() -> None:
    sources = (
        torch.arange(24.0).view(2, 3, 4).requires_grad_(),
        torch.arange(100.0, 124.0).view(2, 3, 4).requires_grad_(),
    )
    inp = cast(ColumnarTensor, ColumnarTensor(sources).transpose(0, 1))

    detached = aten.detach.default(inp)
    alias = aten.alias.default(detached)

    assert alias.tolist() == inp.tolist()
    assert detached.tolist() == inp.tolist()
    assert alias.stride() == inp.stride()
    assert detached.stride() == inp.stride()
    for source, alias_column, detached_column in zip(
        sources,
        alias.unbind(-1),
        detached.unbind(-1),
    ):
        assert alias_column.data_ptr() == source.data_ptr()
        assert detached_column.data_ptr() == source.data_ptr()
        assert not detached_column.requires_grad
