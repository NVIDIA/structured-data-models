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


def test_contiguous_layout() -> None:
    base = make_columnar()
    inp = cast(ColumnarTensor, base.transpose(0, 1))
    reference = dense(base).transpose(0, 1)

    contiguous = aten.contiguous.default(inp)

    assert_matches_dense(
        contiguous,
        aten.contiguous.default(reference),
    )
    assert contiguous.is_contiguous()


def test_contiguous_input_preserves_identity() -> None:
    inp = make_columnar()

    assert aten.contiguous.default(inp) is inp
