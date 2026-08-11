from typing import cast

import torch
from torch import Tensor

from sdm import ColumnarTensor

aten = torch.ops.aten


def make_columnar() -> ColumnarTensor:
    return ColumnarTensor(
        (
            torch.arange(24.0).view(2, 3, 4),
            torch.arange(100.0, 124.0).view(2, 3, 4),
        ),
    )


def dense(inp: ColumnarTensor) -> Tensor:
    return torch.tensor(inp.tolist())


def assert_matches_dense(out: ColumnarTensor, expected: Tensor) -> None:
    assert type(out) is ColumnarTensor
    assert out.size() == expected.size()
    assert out.stride() == expected.stride()
    assert out.storage_offset() == expected.storage_offset()
    assert out.layout == expected.layout
    assert out.tolist() == expected.tolist()


def test_clone_layout() -> None:
    base = make_columnar()
    inp = cast(ColumnarTensor, base.transpose(0, 1))
    reference = dense(base).transpose(0, 1)

    clone = aten.clone.default(inp)

    assert_matches_dense(
        clone,
        aten.clone.default(reference),
    )
    assert not clone.is_contiguous()
    assert clone.unbind(-1)[0].data_ptr() != inp.unbind(-1)[0].data_ptr()


def test_empty_clone_preserves_device() -> None:
    inp = ColumnarTensor((), size=(2, 3), device="meta")

    clone = inp.clone()

    assert type(clone) is ColumnarTensor
    assert clone.size() == inp.size()
    assert clone.device == inp.device
