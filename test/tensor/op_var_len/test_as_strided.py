from typing import cast

import torch
from torch import Tensor

from sdm import VarLenTensor

aten = torch.ops.aten


def _is_alias(left: Tensor, right: Tensor) -> bool:
    return torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        left,
        right,
    )


def test_as_strided_preserves_logical_alias() -> None:
    dense = torch.arange(6).view(2, 3)
    tensor = VarLenTensor.from_tensor(dense)
    out = aten.as_strided.default(tensor, [2, 2], [3, 1], 1)
    expected = aten.as_strided.default(dense, [2, 2], [3, 1], 1)

    assert isinstance(out, VarLenTensor)
    assert out.size() == expected.size()
    assert out.stride() == expected.stride()
    assert out.storage_offset() == expected.storage_offset()
    assert out.tolist() == VarLenTensor.from_tensor(expected).tolist()
    assert _is_alias(cast(Tensor, out), tensor)
