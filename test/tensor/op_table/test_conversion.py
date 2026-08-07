from collections.abc import Callable
from typing import cast

import pytest
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
    copy_op: Callable[[TableTensor], TableTensor],
) -> None:
    inp = cast(TableTensor, make_table().transpose(0, 1))
    expected = inp.numerical.clone(memory_format=torch.preserve_format)

    out = copy_op(inp)

    assert_matches_dense(out, expected)
    assert out.numerical.data_ptr() != inp.numerical.data_ptr()


def test_noop_conversion_preserves_identity() -> None:
    inp = make_table()

    assert aten.to.dtype_layout(inp) is inp
    assert aten.to.dtype(inp, inp.dtype) is inp
    assert aten.to.device(inp, inp.device, inp.dtype) is inp
    assert aten.to.other(inp, torch.empty(0, dtype=inp.dtype)) is inp


def test_dtype_conversion_preserves_semantic_dtypes() -> None:
    inp = make_table()

    out = aten.to.dtype(inp, torch.float64)

    assert out.dtype == torch.float64
    assert out.numerical.dtype == torch.float64
    assert out.datetime.dtype == torch.int64
