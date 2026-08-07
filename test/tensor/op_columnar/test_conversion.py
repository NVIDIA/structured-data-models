from collections.abc import Callable
from typing import cast

import pytest
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


@pytest.mark.parametrize(
    "copy_op",
    [
        pytest.param(
            lambda inp: aten.to.dtype_layout(
                inp,
                dtype=inp.dtype,
                device=inp.device,
                copy=True,
            ),
            id="to.dtype_layout",
        ),
        pytest.param(
            lambda inp: aten.to.dtype(inp, inp.dtype, False, True),
            id="to.dtype",
        ),
        pytest.param(
            lambda inp: aten.to.device(
                inp,
                inp.device,
                inp.dtype,
                False,
                True,
            ),
            id="to.device",
        ),
        pytest.param(
            lambda inp: aten.to.other(
                inp,
                torch.empty(0, dtype=inp.dtype),
                False,
                True,
            ),
            id="to.other",
        ),
        pytest.param(
            lambda inp: aten._to_copy.default(inp, device=inp.device),
            id="_to_copy.default",
        ),
    ],
)
def test_copy_overloads(
    copy_op: Callable[[ColumnarTensor], ColumnarTensor],
) -> None:
    base = make_columnar()
    inp = cast(ColumnarTensor, base.transpose(0, 1))
    expected = (
        dense(base)
        .transpose(0, 1)
        .clone(
            memory_format=torch.preserve_format,
        )
    )

    out = copy_op(inp)

    assert_matches_dense(out, expected)
    for actual, source in zip(out.unbind(-1), inp.unbind(-1)):
        assert actual.data_ptr() != source.data_ptr()


def test_noop_conversion_preserves_identity() -> None:
    inp = make_columnar()

    assert aten.to.dtype_layout(inp) is inp
    assert aten.to.dtype(inp, inp.dtype) is inp
    assert aten.to.device(inp, inp.device, inp.dtype) is inp
    assert aten.to.other(inp, torch.empty(0, dtype=inp.dtype)) is inp


def test_dtype_conversion_is_unsupported() -> None:
    with pytest.raises(TypeError, match="convert"):
        aten.to.dtype(make_columnar(), torch.float64)
