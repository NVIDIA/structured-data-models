from typing import cast

import torch
from torch import Tensor

from sdm import VarLenTensor


def _is_alias(left: Tensor, right: Tensor) -> bool:
    return torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        left,
        right,
    )


def test_empty_view_with_nonzero_storage_offset() -> None:
    tensor = VarLenTensor(
        data=torch.empty(0),
        offset=torch.zeros(1, dtype=torch.int64),
        valid=torch.empty(0, dtype=torch.bool),
        size=(2, 3, 4, 0),
    )
    out = cast(VarLenTensor, tensor[:, :, 1:3])

    assert out.size() == (2, 3, 2, 0)
    assert out.stride() == (12, 4, 1, 1)
    assert out.storage_offset() == 1
    assert _is_alias(out, tensor)
    data, offset = out.data_offset
    assert data.numel() == 0
    assert offset.equal(torch.zeros(1, dtype=torch.int64))
    assert out.to_arrow().to_pylist() == []
